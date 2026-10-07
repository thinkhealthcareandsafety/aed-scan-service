"""
Gemini Vision Service — checklist mode.

One multimodal Gemini call per uploaded photo/video, scoped to a single
checklist item (see `checklist_items.py`). This replaces the old
continuous per-frame WebSocket state machine: the inspector captures or
uploads one piece of media per item, and gets one AI verdict back.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from datetime import date, datetime, timezone
from typing import List, Optional

import structlog
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.checklist_items import ChecklistItem, get_item
from app.services import readiness_frames
from app.services.device_profiles import DeviceProfile, get_profile
from app.utils import validators
from app.utils.date_parser import expiry_last_valid_day, parse_all_expiry_dates, parse_gs1_dates

logger = structlog.get_logger(__name__)

# Chosen by measurement (python-cv/eval, 1 Oct 2026), not by name:
#
#   photos, 47 cases   gemini-3.5-flash-lite  42 correct, 4 retakes on
#                      unreadable shots, 0 wrong readings; p50 2.1 s,
#                      p95 3.6 s. Needed no correction from the checks
#                      below beyond the ones they always make.
#                      gemini-3.1-flash-lite (the previous choice) slower,
#                      and read in-date pads as expired until the service
#                      took the date decision away from it.
#                      gemini-2.5-flash-lite: withdrawn by Google (404).
#   readiness, 7 clips gemini-3.5-flash-lite  21/21 over three runs; p50
#   x 3 runs           3.4 s. gemini-3.8-flash 4/7, the other three timed
#                      out at 50 s. gemini-3.6-flash (the previous choice)
#                      was 21/21 the day before but slower (6-14 s), and
#                      on a free-tier key it is capped at 20 requests a day.
#
# Each list is a fallback chain: the next model is asked when the one
# before it is slow, failing or out of quota. Google counts quota per
# model, so a second model is a second allowance as well as a second chance.
#
# AEDSmartX: each chain can be overridden from the environment (comma-
# separated, first is primary) so a slow or withdrawn model can be swapped
# on Render without a deploy. Unset keeps the measured defaults above.
def _models_from_env(name: str, default: tuple) -> tuple:
    raw = os.environ.get(name, "")
    chosen = tuple(m.strip() for m in raw.split(",") if m.strip())
    return chosen or default


_IMAGE_CHAIN = _models_from_env("GEMINI_IMAGE_MODELS", ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"))
GEMINI_IMAGE_MODEL = _IMAGE_CHAIN[0]
GEMINI_IMAGE_FALLBACKS: tuple = _IMAGE_CHAIN[1:]
GEMINI_VIDEO_MODELS = _models_from_env("GEMINI_VIDEO_MODELS", ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"))

# Google's own SDK retries internally, but has been observed giving up
# within a few seconds even on a transient "model experiencing high
# demand" 503 — too eager for a spike that usually clears in seconds. Each
# model keeps retrying with short backoff inside the overall budget (see
# _first_answer), which absorbs that without surfacing a failure to the
# inspector mid-checklist. Only Google-side (5xx) errors and stalled calls
# are retried — a bad request or auth failure (4xx) won't fix itself.
RETRY_BACKOFF_SECONDS = (1.0, 2.5)
OVERALL_TIMEOUT_SECONDS = 35.0
# A video is one check per inspection, and a required one: it gets a longer
# budget so hedged retries can finish. Still inside the browser's upload
# timeout (60 s of analysis on top of the upload itself).
VIDEO_TIMEOUT_SECONDS = 50.0
# No single call may hold the budget hostage: an overloaded model's refusal
# alone can take 12-16 s to arrive.
PER_ATTEMPT_TIMEOUT_SECONDS = 18.0
# How long the first model in a chain runs alone before the next is asked
# too: past the slowest normal answer, so a healthy call isn't paid for
# twice. Readiness clips answer in 2.5-6 s; one in fourteen stalled to 42 s
# with no backup, and a backup at 9 s turns that into ~14 s. A photo
# answers in ~2 s. A failure — an outage, a quota refusal — starts the
# backup at once anyway, without waiting for these.
HEDGE_DELAY_SECONDS = 9.0
IMAGE_HEDGE_DELAY_SECONDS = 7.0

# Labels are read twice, by two different models, and the readings must
# agree. A model reading fine print is occasionally confidently wrong —
# measured: one serial in about seventy, an 8 read as a B on a dim photo —
# and a wrong serial or date on an inspection record is worse than a
# retake. Two models rarely make the same mistake on the same character.
SECOND_READ_ITEMS = frozenset({"serial_number", "pads_expiry", "battery_expiry"})
#: How long, after the first reading arrives, to wait for the second. If it
#: isn't back by then (or failed), the first reading stands on its own.
#: AEDSmartX: overridable, since how long the second model takes depends on
#: the key's tier — on a free-tier key it was measured at 11-23 s.
SECOND_READ_GRACE_SECONDS = float(os.environ.get("SECOND_READ_GRACE_SECONDS", "4.0"))

# How the video reaches the model — every frame scanned for the Ready
# light's flashes, then labelled frames sent — lives in readiness_frames.py.


# A ceiling on AI calls per day (UTC), so a flood of uploads — a bug, a bot,
# someone looping retakes — can't spend the prepaid credit in an afternoon.
# ~2,500 calls is ~200 inspections. Raise it on Render, no code change.
DAILY_AI_CALL_LIMIT = int(os.environ.get("DAILY_AI_CALL_LIMIT", "2500"))
_calls_today = {"day": None, "count": 0}


class DailyLimitReached(RuntimeError):
    """Today's AI-call ceiling is spent."""


def _spend_call() -> None:
    today = datetime.now(timezone.utc).date()
    if _calls_today["day"] != today:
        _calls_today.update(day=today, count=0)
    if _calls_today["count"] >= DAILY_AI_CALL_LIMIT:
        logger.error("checklist.daily_limit_reached", limit=DAILY_AI_CALL_LIMIT)
        raise DailyLimitReached(f"Daily AI call limit of {DAILY_AI_CALL_LIMIT} reached")
    _calls_today["count"] += 1


def ai_calls_today() -> dict:
    today = datetime.now(timezone.utc).date()
    used = _calls_today["count"] if _calls_today["day"] == today else 0
    return {"used": used, "limit": DAILY_AI_CALL_LIMIT}


# Settings the evaluation harness (python-cv/eval) varies to compare
# options on real labels; production runs with these values.
TEMPERATURE = 0.1
#: How finely each image is tokenised. None leaves it to the model.
IMAGE_MEDIA_RESOLUTION: Optional[types.MediaResolution] = None
#: "low" / "high" on models that think; None leaves it to the model.
THINKING_LEVEL: Optional[str] = None
#: Bumped by hand when the shared prompt template's wording changes; the
#: per-item and per-device wording is hashed in automatically.
PROMPT_REVISION = "2026-10-05"  # the serial label's manufacture date, for the AED's age


class ChecklistVerdict(BaseModel):
    """What the model fills in — the response schema it is held to."""

    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str
    serial_number: Optional[str] = None
    expiry_date: Optional[str] = None
    expiry_raw_text: Optional[str] = None
    # Captured so the manufacture date can never be mistaken for the expiry:
    # a Philips battery label carries both, and reading the wrong one reports
    # a five-year-old-stock battery as already dead.
    manufacture_date: Optional[str] = None
    # A date someone wrote on the battery when fitting it. Read only where a
    # battery carries no expiry of its own (Powerheart), and dated from.
    install_date: Optional[str] = None
    lot_number: Optional[str] = None
    battery_serial_number: Optional[str] = None
    present: Optional[bool] = None
    status: Optional[str] = None  # readiness_indicator only: ready | fault | unclear
    # readiness_indicator only: the numbered frames (or, for a raw video,
    # the seconds) in which the model can see the ready signal. A "ready"
    # with none is not accepted — see _check_readiness.
    ready_frames: Optional[List[int]] = None
    # The brand of AED the model can see. A photo of a different maker's
    # unit can't pass a check for this one — see _different_brand.
    brand_seen: Optional[str] = None
    # Expiry items: the model reports what it saw and the service decides.
    # date_legible — it read the date with certainty; damage_seen — the
    # pads or battery look damaged, opened, swollen, leaking or corroded.
    date_legible: Optional[bool] = None
    damage_seen: Optional[bool] = None
    # The same message as `notes`, in Hindi, when the inspector is using the
    # app in Hindi. `notes` stays English: it is what the PDF report and the
    # sales team read.
    notes_hi: Optional[str] = None


class AnalysisMeta(BaseModel):
    """How a verdict was reached — written by this service, never by the
    model. The backend stores the whole result, so any answer can be traced
    later: which model gave it, under which version of the instructions, how
    long it took, what it cost, and which safety rules changed it. Without
    this, "why did this AED pass?" had no answer beyond the verdict itself."""

    model: Optional[str] = None
    prompt_version: Optional[str] = None
    latency_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None
    frames_sent: Optional[int] = None
    flashes_found: Optional[int] = None
    #: Labels: the model that read it a second time, and how that went —
    #: agrees, disagrees, unread (it couldn't read it), unavailable.
    second_model: Optional[str] = None
    second_read: Optional[str] = None
    #: Each rule that overruled or corrected the model, by name.
    overrides: List[str] = Field(default_factory=list)


class ChecklistAnalysisResult(ChecklistVerdict):
    meta: Optional[AnalysisMeta] = None


def _prompt_version(item: ChecklistItem, profile: DeviceProfile) -> str:
    """A short fingerprint of every instruction this verdict was given."""
    text = "\n".join(
        [PROMPT_REVISION, item.prompt, profile.name, profile.appearance, profile.guidance.get(item.id, "")]
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _generation_config(item: ChecklistItem) -> types.GenerateContentConfig:
    extra = {}
    if item.media_type == "image" and IMAGE_MEDIA_RESOLUTION is not None:
        extra["media_resolution"] = IMAGE_MEDIA_RESOLUTION
    if THINKING_LEVEL:
        extra["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ChecklistVerdict,
        temperature=TEMPERATURE,
        **extra,
    )


# Languages the inspector's feedback can be written in, besides English.
SUPPORTED_LANGUAGES = {"hi"}

_HINDI_MONTHS = (
    "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
)

# The fixed messages this module writes when it overrules the model. Each
# replaces the model's own notes — so each needs its Hindi form too, or a
# Hindi reader would be shown the model's now-contradicted verdict.
_OVERRIDE_NOTES = {
    "implausible_serial": (
        "Serial number reading looked implausible — please recapture with the label centred and in focus.",
        "सीरियल नंबर ठीक से नहीं पढ़ा जा सका। लेबल को बीच में रखकर, साफ़ और फ़ोकस में फिर से फ़ोटो लें।",
    ),
    "manufacture_as_expiry": (
        "That date is the manufacture date, not the expiry date. Look for the date marked "
        "'Install before' or with an hourglass symbol and photograph that part of the label.",
        "यह बनने की तारीख है, एक्सपायरी की नहीं। 'Install before' या रेत-घड़ी (⌛) वाले निशान के पास "
        "लिखी तारीख ढूँढें और लेबल के उस हिस्से की फ़ोटो लें।",
    ),
    "readiness_no_blink": (
        "We couldn't see the green Ready light flash. Film only the small Ready light (not the "
        "On/Off button), up close and steady, for at least 10 seconds. If it never flashes, "
        "the AED needs attention.",
        "हमें हरी Ready लाइट जलती हुई नहीं दिखी। सिर्फ़ छोटी Ready लाइट (On/Off बटन नहीं) का पास से, "
        "फ़ोन स्थिर रखकर, कम से कम 10 सेकंड का वीडियो बनाएँ। अगर यह कभी नहीं जलती, तो AED को जाँच की ज़रूरत है।",
    ),
    "readiness_no_check": (
        "We couldn't clearly see a green check in the status window. Film the window on the "
        "handle up close and steady for about 5 seconds. A red X means the AED needs attention.",
        "हमें स्टेटस विंडो में हरा ✓ साफ़ नहीं दिखा। हैंडल पर लगी विंडो का पास से, फ़ोन स्थिर रखकर, "
        "करीब 5 सेकंड का वीडियो बनाएँ। लाल ✗ का मतलब है कि AED को जाँच की ज़रूरत है।",
    ),
    "readiness_no_check_aed3": (
        "We couldn't clearly see a green check in the status window. Film the small window just "
        "right of the On/Off button, up close and steady, for about 5 seconds. If the window stays "
        "blank, the AED needs attention.",
        "हमें स्टेटस विंडो में हरा ✓ साफ़ नहीं दिखा। On/Off बटन के ठीक दाईं ओर की छोटी विंडो का पास से, "
        "फ़ोन स्थिर रखकर, करीब 5 सेकंड का वीडियो बनाएँ। अगर विंडो खाली ही रहे, तो AED को जाँच की ज़रूरत है।",
    ),
    "readiness_no_rescue_ready": (
        "We couldn't clearly see the Rescue Ready indicator showing green. With the lid closed, film "
        "the round indicator beside the handle, up close and steady, for about 10 seconds. If it "
        "stays red, the AED needs attention.",
        "हमें Rescue Ready इंडिकेटर हरा नहीं दिखा। ढक्कन बंद रखकर, हैंडल के पास वाले गोल इंडिकेटर का पास से, "
        "फ़ोन स्थिर रखकर, करीब 10 सेकंड का वीडियो बनाएँ। अगर यह लाल ही रहे, तो AED को जाँच की ज़रूरत है।",
    ),
    "readiness_no_asi_lifeline": (
        "We couldn't see the status light flash green. Film the small light at the top-right "
        "corner, at the end of the handle (not the On/Off button), up close and steady, for at "
        "least 10 seconds — it flashes about every 5 seconds. Red, or no flash at all, means the "
        "AED needs attention.",
        "हमें स्टेटस लाइट हरी जलती नहीं दिखी। ऊपर दाएँ कोने में, हैंडल के सिरे पर लगी छोटी लाइट (On/Off बटन नहीं) "
        "का पास से, फ़ोन स्थिर रखकर, कम से कम 10 सेकंड का वीडियो बनाएँ — यह करीब हर 5 सेकंड में जलती है। लाल "
        "लाइट, या बिल्कुल न जलना, मतलब AED को जाँच की ज़रूरत है।",
    ),
    "readiness_no_asi_view": (
        "We couldn't see the status light flash green. Film the small light just right of the "
        "On/Off button (not the button or the screen), up close and steady, for at least 10 "
        "seconds. Red, or no flash at all, means the AED needs attention.",
        "हमें स्टेटस लाइट हरी जलती नहीं दिखी। On/Off बटन के ठीक दाईं ओर की छोटी लाइट (बटन या स्क्रीन नहीं) का "
        "पास से, फ़ोन स्थिर रखकर, कम से कम 10 सेकंड का वीडियो बनाएँ। लाल लाइट, या बिल्कुल न जलना, मतलब AED "
        "को जाँच की ज़रूरत है।",
    ),
    "serial_defibtech_back": (
        "That isn't this AED's serial number — a Defibtech serial is 9 digits. Slide the pads "
        "package out of its holder on the back (keep it plugged in), photograph the label behind "
        "it, then slide the pads back.",
        "यह इस AED का सीरियल नंबर नहीं है — Defibtech का सीरियल नंबर 9 अंकों का होता है। पीछे के होल्डर से पैड्स "
        "का पैकेट बाहर खिसकाएँ (प्लग लगा रहने दें), उसके पीछे के लेबल की फ़ोटो लें, फिर पैड्स वापस रख दें।",
    ),
    "serial_defibtech_view": (
        "That isn't this AED's serial number — it is 9 digits starting with 4. With the AED off, "
        "press the middle button beside the screen and photograph the “AED S/N” line (not "
        "“Battery S/N”).",
        "यह इस AED का सीरियल नंबर नहीं है — यह 4 से शुरू होने वाला 9 अंकों का नंबर होता है। AED बंद रखकर "
        "स्क्रीन के पास वाला बीच का बटन दबाएँ और “AED S/N” वाली लाइन की फ़ोटो लें (“Battery S/N” नहीं)।",
    ),
    "battery_made_unread": (
        "We couldn't read the date on the battery. This battery shows only the date it was made, "
        "beside the factory symbol — photograph that part of its label, close and in focus.",
        "बैटरी पर लिखी तारीख नहीं पढ़ी जा सकी। इस बैटरी पर सिर्फ़ उसके बनने की तारीख होती है, फ़ैक्टरी वाले "
        "निशान के पास — लेबल के उस हिस्से की पास से, साफ़ फ़ोटो लें।",
    ),
    "readiness_no_evidence": (
        "We couldn't clearly see the AED's ready signal. Film the readiness indicator up close "
        "and steady for at least 10 seconds.",
        "हमें AED का रेडी सिग्नल साफ़ नहीं दिखा। रेडीनेस इंडिकेटर का पास से, फ़ोन स्थिर रखकर, "
        "कम से कम 10 सेकंड का वीडियो बनाएँ।",
    ),
    "reads_disagree_serial": (
        "We couldn't read the serial number with certainty. Retake it closer, holding steady, in good light.",
        "सीरियल नंबर पक्के तौर पर नहीं पढ़ा जा सका। पास से, फ़ोन स्थिर रखकर, अच्छी रोशनी में फिर से फ़ोटो लें।",
    ),
    "reads_disagree_date": (
        "We couldn't read the expiry date with certainty. Retake it closer, holding steady, in good light.",
        "एक्सपायरी डेट पक्के तौर पर नहीं पढ़ी जा सकी। पास से, फ़ोन स्थिर रखकर, अच्छी रोशनी में फिर से फ़ोटो लें।",
    ),
    "implausible_expiry": (
        "Expiry date reading looked implausible — please recapture with the label centred, well lit, and in focus.",
        "एक्सपायरी डेट ठीक से नहीं पढ़ी जा सकी। लेबल को बीच में रखकर, अच्छी रोशनी में और फ़ोकस में फिर से फ़ोटो लें।",
    ),
}


def _override_notes(key: str, language: Optional[str]) -> dict:
    english, hindi = _OVERRIDE_NOTES[key]
    return {"notes": english, "notes_hi": hindi if language == "hi" else None}


def _expired_notes(item_id: str, last_valid: date, language: Optional[str]) -> dict:
    what = "pads" if item_id == "pads_expiry" else "battery"
    english = (
        f"The {what} expired on {last_valid.strftime('%d %b %Y')}. "
        f"Replace the {what} before this AED is relied on in an emergency."
    )
    hindi = None
    if language == "hi":
        when = f"{last_valid.day} {_HINDI_MONTHS[last_valid.month - 1]} {last_valid.year}"
        hindi = (
            f"पैड्स {when} को एक्सपायर हो चुके हैं। किसी इमरजेंसी में इस AED पर भरोसा करने से पहले पैड्स बदलें।"
            if what == "pads"
            else f"बैटरी {when} को एक्सपायर हो चुकी है। किसी इमरजेंसी में इस AED पर भरोसा करने से पहले बैटरी बदलें।"
        )
    return {"notes": english, "notes_hi": hindi}


def _hindi_date(day: date) -> str:
    return f"{day.day} {_HINDI_MONTHS[day.month - 1]} {day.year}"


def _in_date_notes(item_id: str, last_valid: date, today: date, language: Optional[str]) -> dict:
    """The verdict on an in-date consumable, written by the service so it
    can never contradict the date (the model has called pads in date until
    next month "expired")."""
    pads = item_id == "pads_expiry"
    soon = (last_valid - today).days <= 90
    when = last_valid.strftime("%d %b %Y")
    english = f"{'Pads' if pads else 'Battery'} in date until {when}." + (
        f" That's within three months — order {'replacement pads' if pads else 'a replacement battery'} soon."
        if soon
        else ""
    )
    hindi = None
    if language == "hi":
        hindi = (
            f"पैड्स {_hindi_date(last_valid)} तक वैध हैं।" + (" जल्द नए पैड्स मँगवा लें।" if soon else "")
            if pads
            else f"बैटरी {_hindi_date(last_valid)} तक वैध है।" + (" जल्द नई बैटरी मँगवा लें।" if soon else "")
        )
    return {"notes": english, "notes_hi": hindi}


# Makers of AEDs, by the words a model uses for them, so a photo of one
# maker's unit is recognised whatever it calls it. Anything else (a battery
# brand such as Duracell, a word it guessed) is ignored rather than risk
# failing a good photo.
_AED_BRANDS = {
    "Philips": ("philips", "heartstart", "laerdal"),
    "ZOLL": ("zoll",),
    "Mindray": ("mindray", "beneheart"),
    "Schiller": ("schiller", "fred easy"),
    "HeartSine": ("heartsine", "samaritan"),
    "Physio-Control": ("physio-control", "physio control", "lifepak", "stryker"),
    "Cardiac Science": ("cardiac science", "powerheart"),
    "Nihon Kohden": ("nihon kohden",),
    "Defibtech": ("defibtech", "lifeline"),
    "CU Medical": ("cu medical", "i-pad"),
}


def _aed_brand(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    for brand, words in _AED_BRANDS.items():
        if any(w in lowered for w in words):
            return brand
    return None


# Checks whose photo shows the AED itself (or its own pads/battery/key). The
# cabinet, rescue kit and contacts sticker may show no AED at all.
_BRAND_CHECKED_ITEMS = {
    "serial_number", "pads_expiry", "battery_expiry", "battery_attached",
    "pads_connected", "readiness_indicator", "child_key_pad",
}


def _different_brand(item: ChecklistItem, result, profile: Optional[DeviceProfile]) -> Optional[str]:
    """The brand the photo shows, when it is clearly a different maker's
    AED from the one being inspected; otherwise None."""
    if profile is None or not profile.brand or item.id not in _BRAND_CHECKED_ITEMS:
        return None
    seen = _aed_brand(result.brand_seen)
    return seen if seen and seen != profile.brand and seen not in profile.brand_aliases else None


def _brand_notes(seen: str, profile: DeviceProfile, language: Optional[str]) -> dict:
    english = (
        f"This looks like a {seen} AED, not the {profile.name} you chose. If you picked the wrong "
        "model, switch it from the ⋯ menu; if not, retake the photo of this unit."
    )
    hindi = (
        f"यह {seen} का AED लगता है, आपका चुना हुआ {profile.name} नहीं। अगर गलत मॉडल चुना है तो ⋯ मेनू "
        "से बदलें, वरना इसी मशीन की फ़ोटो फिर से लें।"
        if language == "hi"
        else None
    )
    return {"notes": english, "notes_hi": hindi}


_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _build_prompt(
    item: ChecklistItem,
    profile: DeviceProfile,
    frame_count: Optional[int] = None,
    language: Optional[str] = None,
    duration: Optional[float] = None,
    timed: bool = True,
) -> str:
    clip = f"a {duration:.0f}-second video clip" if duration else "a short video clip"
    labels = (
        "each preceded by its label (\"Frame 3 — 1.2 s\"). They are not "
        "evenly spaced."
        if timed
        else "each preceded by its label (\"Frame 3\"), spread evenly across the clip."
    )
    sequence_note = (
        (
            f"You are given {frame_count} still frames from {clip}, in strict "
            f"chronological order, {labels} Treat them as one "
            "continuous observation of the same scene over time, not as "
            "separate photos: a change seen in only one or two frames (a "
            "light turning on, then off again) is exactly the kind of brief "
            "event to look for. Refer to frames by their numbers.\n\n"
        )
        if frame_count
        else ""
    )
    guidance = profile.guidance.get(item.id)
    device_notes = f"Device notes for this item ({profile.name}):\n{guidance}\n\n" if guidance else ""
    language_rule = (
        "Also fill notes_hi with the same message as notes, in simple, "
        "everyday Hindi in Devanagari script, the way site staff in India "
        "speak: keep common product words in their usual Hindi form (AED, "
        "पैड्स, बैटरी, सीरियल नंबर, एक्सपायरी डेट, फ़ोटो, वीडियो). Keep notes "
        "itself in English."
        if language == "hi"
        else "Leave notes_hi null."
    )
    return (
        "You are the vision engine for an AED (defibrillator) inspection "
        "checklist app. You receive photo(s) for exactly one checklist "
        "item and must return a single structured verdict.\n\n"
        # The model has no reliable sense of the current date. Without this it
        # judged a 2026 expiry as "still valid" in late 2026. The server makes
        # the final expired/not-expired call regardless; this keeps the notes
        # it writes consistent with that verdict.
        f"Today's date is {datetime.now(timezone.utc).date().isoformat()}. "
        "A date earlier than today is in the past.\n\n"
        # Which unit this is. Every device-specific fact the model is given
        # comes from this profile, so it never judges one brand by another's
        # layout. If the photo shows something else, it says so.
        f"DEVICE: {profile.name}. {profile.appearance} Set brand_seen to "
        "the maker's name as printed on the AED itself — not the brand of "
        "batteries or other parts inside it — or null if no AED is visible "
        "or you can't read its maker. If the image clearly shows a different "
        "device, say so in notes.\n\n"
        f"{sequence_note}"
        f"Checklist item: {item.title}\n"
        f"Task: {item.prompt}\n\n"
        f"{device_notes}"
        "Always set confidence (0.0-1.0) to your own honest certainty in "
        "this specific media — a blurry, distant, dark, or ambiguous capture "
        "should score low even if you still produced a best-effort answer. "
        "notes is one short, friendly sentence: if passed=false, tell the "
        "inspector exactly what to fix or recapture; if passed=true, briefly "
        "confirm what you saw. Leave any data field you cannot determine as "
        f"null — never guess. {language_rule}"
    )


def _is_video_upload(content_type: Optional[str]) -> bool:
    """A single still sent for the readiness item is judged as a photo, not
    handed to the video decoder."""
    return not content_type or not content_type.startswith("image/")


def _mime_type_for(item: ChecklistItem, declared_content_type: Optional[str]) -> str:
    if declared_content_type and "/" in declared_content_type:
        return declared_content_type
    return "video/mp4" if item.media_type == "video" else "image/jpeg"


def _parse(response) -> ChecklistVerdict:
    parsed = response.parsed
    return parsed if isinstance(parsed, ChecklistVerdict) else ChecklistVerdict.model_validate_json(response.text)


def _reading(item: ChecklistItem, verdict, profile: Optional[DeviceProfile] = None) -> Optional[str]:
    """What a verdict read off the label, in a form two readings can be
    compared in: the serial's letters and digits, or the expiry's month —
    or, on a battery dated by its age, the month it is dated from."""
    if item.id == "serial_number":
        cleaned = validators.normalise_serial(verdict.serial_number or "") or ""
        cleaned = "".join(ch for ch in cleaned.upper() if ch.isalnum())
        return cleaned or None
    if item.id == "battery_expiry" and profile is not None and profile.battery_life_months:
        value = (verdict.install_date or verdict.manufacture_date or verdict.expiry_date or "").strip()
    else:
        value = (verdict.expiry_date or "").strip()
    return value[:7] if len(value) >= 7 else None


def _compare_readings(item: ChecklistItem, first, second, profile: Optional[DeviceProfile] = None) -> str:
    """agrees / disagrees / unread. A second model that couldn't read the
    label doesn't contradict one that could — it only fails to confirm."""
    a, b = _reading(item, first, profile), _reading(item, second, profile)
    if not b:
        return "unread"
    if not a:
        return "unread"
    return "agrees" if a == b else "disagrees"


def _quota_refusal(exc: errors.ClientError) -> tuple:
    """(daily, retry_after_seconds) from a 429. A per-day refusal won't
    clear for hours, so that model is done for this request; a per-minute
    one clears in seconds."""
    details = []
    try:
        details = (exc.details or {}).get("error", {}).get("details", []) or []
    except AttributeError:
        details = []
    daily, retry_after = False, None
    for d in details:
        kind = str(d.get("@type", ""))
        if kind.endswith("QuotaFailure"):
            daily = daily or any("PerDay" in str(v.get("quotaId", "")) for v in d.get("violations", []))
        elif kind.endswith("RetryInfo"):
            try:
                retry_after = float(str(d.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                retry_after = None
    return daily, retry_after


async def analyze_checklist_item(
    item_id: str,
    media_bytes: bytes,
    content_type: Optional[str] = None,
    *,
    aed_model: Optional[str] = None,
    language: Optional[str] = None,
    frames: Optional[List[bytes]] = None,
) -> ChecklistAnalysisResult:
    """Analyse one uploaded photo/video against its checklist item prompt,
    for the given AED model (profile) and inspector language.

    frames: AEDSmartX only — JPEG stills a browser already pulled out of a
    readiness clip, in time order, sent instead of the video when the clip
    was too large to upload. They are numbered for the model like decoded
    frames, but there is no flash scan behind them (evenly spaced stills
    can fall between a Philips flash), so a ready verdict on them rests on
    the frames the model cites."""
    item = get_item(item_id)
    if item is None:
        raise ValueError(f"Unknown checklist item: {item_id}")

    profile = get_profile(aed_model)
    lang = language if language in SUPPORTED_LANGUAGES else None
    logger.info("checklist.analyze", item_id=item_id, profile=profile.id, language=lang or "en")

    client = _get_client()

    video = (
        readiness_frames.prepare(media_bytes)
        if item.media_type == "video" and not frames and _is_video_upload(content_type)
        else None
    )

    if frames and item.media_type == "video":
        logger.info("checklist.frames_supplied", item_id=item_id, count=len(frames))
        contents = []
        for number, frame in enumerate(frames, start=1):
            contents.append(types.Part(text=f"Frame {number}"))
            contents.append(types.Part.from_bytes(data=frame, mime_type="image/jpeg"))
        contents.append(_build_prompt(item, profile, frame_count=len(frames), language=lang, timed=False))
        models = list(GEMINI_VIDEO_MODELS)
    elif video:
        logger.info(
            "checklist.frames_prepared",
            item_id=item_id,
            count=len(video.frames),
            flashes=video.flash_count,
            duration=video.duration,
        )
        contents = []
        for number, (frame, at) in enumerate(zip(video.frames, video.times), start=1):
            contents.append(types.Part(text=f"Frame {number} — {at:.1f} s"))
            contents.append(types.Part.from_bytes(data=frame, mime_type="image/jpeg"))
        contents.append(
            _build_prompt(item, profile, frame_count=len(video.frames), language=lang, duration=video.duration)
        )
        models = list(GEMINI_VIDEO_MODELS)
    else:
        mime_type = _mime_type_for(item, content_type)
        contents = [
            types.Part.from_bytes(data=media_bytes, mime_type=mime_type),
            _build_prompt(item, profile, language=lang),
        ]
        models = (
            list(GEMINI_VIDEO_MODELS)
            if item.media_type == "video"
            else [GEMINI_IMAGE_MODEL, *[m for m in GEMINI_IMAGE_FALLBACKS if m != GEMINI_IMAGE_MODEL]]
        )

    config = _generation_config(item)

    async def _call_once(model: str):
        _spend_call()
        return await client.aio.models.generate_content(model=model, contents=contents, config=config)

    hedge = HEDGE_DELAY_SECONDS if item.media_type == "video" else IMAGE_HEDGE_DELAY_SECONDS
    # Set when a lane's model fails, to start the next model straight away
    # rather than at its scheduled hedge time.
    wake = [asyncio.Event() for _ in models]

    def _fail_over(index: int) -> None:
        if index + 1 < len(wake):
            wake[index + 1].set()

    async def _keep_trying(index: int, model: str, head_start: float):
        """One model's lane: wait out its head start (or until the lane
        before it fails), then call it — and, on a Google-side error, a
        stalled call or a per-minute quota refusal, call it again — until
        it answers or the overall budget cancels it. Out of DAILY quota, it
        hands over to the next model and stops."""
        if head_start > 0:
            try:
                await asyncio.wait_for(wake[index].wait(), timeout=head_start)
            except asyncio.TimeoutError:
                pass
        attempt = 0
        while True:
            attempt += 1
            if attempt > 1 or index > 0:
                logger.info("checklist.gemini_attempt", item_id=item_id, model=model, attempt=attempt)
            try:
                return model, await asyncio.wait_for(_call_once(model), timeout=PER_ATTEMPT_TIMEOUT_SECONDS)
            except (errors.ServerError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "checklist.gemini_server_error",
                    item_id=item_id,
                    model=model,
                    attempt=attempt,
                    error=str(exc) or type(exc).__name__,
                )
                _fail_over(index)
                await asyncio.sleep(RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)])
            except errors.ClientError as exc:
                if getattr(exc, "code", None) != 429:
                    raise
                daily, retry_after = _quota_refusal(exc)
                logger.warning(
                    "checklist.gemini_quota",
                    item_id=item_id,
                    model=model,
                    daily=daily,
                    retry_after=retry_after,
                )
                _fail_over(index)
                if daily:
                    raise
                await asyncio.sleep(min(retry_after or 2.0, 10.0))

    async def _first_answer():
        """Hedged calls: each model in the chain gets its own lane, started
        HEDGE_DELAY_SECONDS after the one before it, and the first answer
        wins. Measured against production, an overloaded model can take
        12-16 s just to say "503 high demand"; walking the chain in order
        spent the whole budget on those refusals before the healthy model
        was ever asked, and the video check timed out for everyone."""
        lanes = [
            asyncio.create_task(_keep_trying(index, model, index * hedge))
            for index, model in enumerate(models)
        ]
        try:
            last_error: Optional[BaseException] = None
            for finished in asyncio.as_completed(lanes):
                try:
                    return await finished
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # a 4xx: this lane gave up for good
                    last_error = exc
            raise last_error  # type: ignore[misc]
        finally:
            for lane in lanes:
                lane.cancel()

    budget = VIDEO_TIMEOUT_SECONDS if item.media_type == "video" else OVERALL_TIMEOUT_SECONDS
    started = time.monotonic()

    second_model = None
    second: Optional[asyncio.Task] = None
    if item.id in SECOND_READ_ITEMS and item.media_type == "image":
        second_model = next((m for m in models[1:] if m != models[0]), None)
        if second_model:

            async def _second_reading():
                response = await asyncio.wait_for(_call_once(second_model), timeout=PER_ATTEMPT_TIMEOUT_SECONDS)
                return _parse(response)

            second = asyncio.create_task(_second_reading())

    try:
        winner, response = await asyncio.wait_for(_first_answer(), timeout=budget)
    except asyncio.TimeoutError as exc:
        if second:
            second.cancel()
        raise TimeoutError(f"Gemini call for checklist item={item_id} exceeded {budget}s") from exc
    except BaseException:
        if second:
            second.cancel()
        raise
    latency_ms = int((time.monotonic() - started) * 1000)

    result = ChecklistAnalysisResult.model_validate(_parse(response).model_dump())

    overrides: List[str] = []
    second_read = None
    if second is not None:
        remaining = max(0.0, budget - (time.monotonic() - started))
        try:
            other = await asyncio.wait_for(second, timeout=min(SECOND_READ_GRACE_SECONDS, remaining))
            second_read = _compare_readings(item, result, other, profile)
        except Exception as exc:  # noqa: BLE001 — slow, refused or unparseable: read once
            second.cancel()
            second_read = "unavailable"
            logger.info("checklist.second_read_unavailable", item_id=item_id, model=second_model,
                        error=str(exc) or type(exc).__name__)

    if second_read == "disagrees":
        overrides.append("second_read_disagrees")
        logger.warning("checklist.readings_disagree", item_id=item_id, first=winner, second=second_model)
        result = result.model_copy(
            update={
                "passed": False,
                "serial_number": None,
                "expiry_date": None,
                "confidence": min(result.confidence, 0.4),
                **_override_notes(
                    "reads_disagree_serial" if item.id == "serial_number" else "reads_disagree_date", lang
                ),
            }
        )
    else:
        result = _apply_deterministic_checks(
            item, result, language=lang, profile=profile, video=video, overrides=overrides
        )

    usage = getattr(response, "usage_metadata", None)

    def _tokens(name: str) -> Optional[int]:
        value = getattr(usage, name, None)
        return value if isinstance(value, int) else None

    meta = AnalysisMeta(
        model=winner,
        prompt_version=_prompt_version(item, profile),
        latency_ms=latency_ms,
        input_tokens=_tokens("prompt_token_count"),
        output_tokens=_tokens("candidates_token_count"),
        thinking_tokens=_tokens("thoughts_token_count"),
        frames_sent=len(video.frames) if video else (len(frames) if frames else None),
        flashes_found=video.flash_count if video else None,
        second_model=second_model if second is not None else None,
        second_read=second_read,
        overrides=overrides,
    )
    logger.info(
        "checklist.verdict",
        item_id=item_id,
        profile=profile.id,
        passed=result.passed,
        **meta.model_dump(exclude_none=True),
    )
    return result.model_copy(update={"meta": meta})


def _check_readiness(
    result: ChecklistAnalysisResult,
    profile: Optional[DeviceProfile],
    video: Optional[readiness_frames.ReadinessFrames],
    language: Optional[str],
    overrides: Optional[List[str]] = None,
) -> ChecklistAnalysisResult:
    """A readiness pass has to be backed by something checkable.

    - The model must name the frames (or, for a raw video, the seconds) in
      which it sees the ready signal. A "ready" it can't point to is not
      accepted — that is how a unit whose light never came on used to pass.
    - A unit that proves readiness by blinking (Philips) must also have
      blinked: if the scan of every frame found no flash at all, no reading
      of the frames by the model can pass it.
    - passed follows status, never the other way round, so a "passed" with
      no status, or a status of fault with passed=true, can't slip through.
    It only ever turns a pass into "unclear" — a retake, with exactly what
    to film — and never turns a fail into a pass."""
    status = (result.status or "").strip().lower()
    if status not in ("ready", "fault", "unclear"):
        status = "ready" if result.passed else "unclear"

    reason = None
    if status == "ready":
        cited = [n for n in (result.ready_frames or []) if n >= 0]
        if video is not None:
            cited = [n for n in cited if 1 <= n <= len(video.frames)]
        if not cited:
            reason = "no evidence"
        elif profile is not None and profile.blinking_ready and video is not None and video.flash_count == 0:
            reason = "no flash in video"

    if reason:
        if overrides is not None:
            overrides.append("readiness_" + reason.replace(" ", "_"))
        logger.info(
            "checklist.readiness_overruled",
            reason=reason,
            profile=getattr(profile, "id", None),
            cited=result.ready_frames,
            flashes=getattr(video, "flash_count", None),
        )
        key = profile.readiness_retake if profile is not None else "readiness_no_evidence"
        return result.model_copy(
            update={
                "status": "unclear",
                "passed": False,
                "confidence": min(result.confidence, 0.5),
                **_override_notes(key, language),
            }
        )
    return result.model_copy(update={"status": status, "passed": status == "ready"})


def _apply_deterministic_checks(
    item: ChecklistItem,
    result: ChecklistAnalysisResult,
    *,
    today: Optional[date] = None,
    language: Optional[str] = None,
    profile: Optional[DeviceProfile] = None,
    video: Optional[readiness_frames.ReadinessFrames] = None,
    overrides: Optional[List[str]] = None,
) -> ChecklistAnalysisResult:
    """Downgrade an implausible read the same way the old state machine did
    — cheap, deterministic sanity checks independent of Gemini's own
    confidence score. Never upgrades a verdict, only vetoes bad ones; the
    only values it changes are ones a standard defines (a GS1 field code
    copied into a serial, a GS1 expiry the model read differently)."""
    # Hindi feedback only when it was asked for — never a stray one.
    if language != "hi" and result.notes_hi:
        result = result.model_copy(update={"notes_hi": None})

    def overruled(rule: str) -> None:
        if overrides is not None:
            overrides.append(rule)

    # A different maker's AED in the photo can't pass this unit's check —
    # most often the wrong model was picked. Said plainly, with the way out.
    other = _different_brand(item, result, profile)
    if other:
        overruled("different_brand")
        logger.warning("checklist.different_brand", item=item.id, seen=result.brand_seen, expected=profile.brand)
        return result.model_copy(
            update={
                "passed": False,
                "status": "unclear" if item.id == "readiness_indicator" else result.status,
                **_brand_notes(other, profile, language),
            }
        )

    if item.id == "readiness_indicator":
        return _check_readiness(result, profile, video, language, overrides)

    if item.id == "serial_number" and result.serial_number:
        # "(21) X14K718292" on a ZOLL label is field code + serial; "SN: ..."
        # on a Philips one is caption + serial. Only the serial is stored.
        cleaned = validators.normalise_serial(result.serial_number)
        if cleaned != result.serial_number:
            overruled("serial_label_text_removed")
            result = result.model_copy(update={"serial_number": cleaned or None})
        if not validators.is_plausible_serial(result.serial_number):
            overruled("serial_implausible")
            logger.warning("checklist.implausible_serial", value=result.serial_number)
            return result.model_copy(
                update={
                    "passed": False,
                    "serial_number": None,
                    **_override_notes("implausible_serial", language),
                }
            )
        # A maker that publishes its serial's form (Defibtech: 9 digits) lets
        # a misread, or the battery's serial read off the status screen,
        # be caught here rather than stored as the AED's.
        if profile is not None and profile.serial_pattern:
            compact = re.sub(r"[\s\-]", "", result.serial_number)
            if not re.fullmatch(profile.serial_pattern, compact):
                overruled("serial_wrong_format")
                logger.warning("checklist.serial_wrong_format", profile=profile.id, value=result.serial_number)
                return result.model_copy(
                    update={
                        "passed": False,
                        "serial_number": None,
                        **_override_notes(profile.serial_retake, language),
                    }
                )
            if compact != result.serial_number:
                result = result.model_copy(update={"serial_number": compact})

    if item.id == "battery_expiry" and profile is not None and profile.battery_life_months:
        return _date_battery_by_age(result, profile, today=today, language=language, overruled=overruled)

    if item.id in ("pads_expiry", "battery_expiry") and result.expiry_date:
        gs1 = parse_gs1_dates(result.expiry_raw_text or "")

        # A manufacture date reported as the expiry is the worst failure mode
        # here — it makes fresh stock look years dead, raises a false alarm
        # with the customer, and puts a bogus entry in the replacement
        # pipeline. If the model handed back the same date for both, or the
        # date the barcode itself labels as production, it read one date and
        # guessed at its meaning; refuse it rather than publish it.
        gs1_production = gs1.get("11")
        # '(17)' is the expiry; '(15)' the best-before, which a ZOLL AED 3
        # battery uses for its install-by date.
        gs1_expiry = gs1.get("17") or gs1.get("15")
        if (result.manufacture_date and result.manufacture_date == result.expiry_date) or (
            gs1_production and gs1_production == result.expiry_date and gs1_expiry != result.expiry_date
        ):
            overruled("expiry_was_manufacture_date")
            logger.warning(
                "checklist.expiry_equals_manufacture",
                item=item.id,
                value=result.expiry_date,
                raw=result.expiry_raw_text,
            )
            return result.model_copy(
                update={
                    "passed": False,
                    "expiry_date": None,
                    **_override_notes("manufacture_as_expiry", language),
                }
            )

        # A GS1 '(17)' (or '(15)') field is the expiry by definition of the
        # standard — no symbol to interpret, no date order to guess. When the
        # model's reading of the printed dates disagrees with it, the barcode
        # wins.
        if gs1_expiry and gs1_expiry[:7] != result.expiry_date[:7]:
            overruled("expiry_corrected_from_gs1")
            logger.warning(
                "checklist.expiry_corrected_from_gs1",
                item=item.id,
                model_said=result.expiry_date,
                gs1=gs1_expiry,
            )
            result = result.model_copy(update={"expiry_date": gs1_expiry})

        plausible = validators.is_plausible_expiry(result.expiry_date)
        agrees = validators.expiry_cross_check_agrees(result.expiry_date, result.expiry_raw_text)
        if not plausible or not agrees:
            overruled("expiry_implausible")
            logger.warning(
                "checklist.implausible_expiry",
                item=item.id,
                value=result.expiry_date,
                raw=result.expiry_raw_text,
            )
            return result.model_copy(
                update={
                    "passed": False,
                    "expiry_date": None,
                    **_override_notes("implausible_expiry", language),
                }
            )
        # Keep the deterministic parser's month/day if Gemini normalised to
        # month-only but the raw text had a day — free precision upgrade.
        # Only ever upgrade to a more precise form of the SAME month. Taking the
        # first date in the raw text would, on a Philips battery, swap a correct
        # "2031-11" for the manufacture date "2026-01-13" simply because it has
        # a day in it.
        if result.expiry_raw_text and len(result.expiry_date) == 7:
            for candidate in parse_all_expiry_dates(result.expiry_raw_text):
                if len(candidate) == 10 and candidate.startswith(result.expiry_date):
                    result = result.model_copy(update={"expiry_date": candidate})
                    break

        # The verdict on an expiry is decided here, against the server's own
        # clock — never by the model. The model has no grounded sense of the
        # current date: in production it failed pads marked 2024-06 but PASSED
        # pads marked 2026-03 and 2026-08, both already expired, because to it
        # 2026 did not yet look like the past. An AED inspection that calls
        # expired pads "ready" is failing at the single thing it exists to do.
        reference_day = today or datetime.now(timezone.utc).date()
        last_valid = expiry_last_valid_day(result.expiry_date)
        if last_valid is not None and last_valid < reference_day:
            # Named by whether the model got there itself: how often it calls
            # an expired date "fine" is a measure of how far to trust it.
            overruled("expired_model_said_pass" if result.passed else "expired")
            logger.warning(
                "checklist.expired_consumable",
                item=item.id,
                expiry=result.expiry_date,
                model_said_passed=result.passed,
            )
            return result.model_copy(
                update={"passed": False, **_expired_notes(item.id, last_valid, language)}
            )

        # In date. The model only reports what it saw; the verdict is ours.
        # Damage fails whatever the date. A date it couldn't read with
        # certainty is a retake. A date it read with certainty, on a unit
        # that looks intact, passes — even when the model, misjudging the
        # calendar, said otherwise.
        if last_valid is not None:
            if result.damage_seen:
                if result.passed:
                    overruled("damage_seen")
                return result.model_copy(update={"passed": False})
            if result.date_legible is False:
                if result.passed:
                    overruled("date_not_certain")
                return result.model_copy(update={"passed": False})
            if result.date_legible and result.confidence >= 0.5:
                if not result.passed:
                    overruled("in_date_model_said_fail")
                return result.model_copy(
                    update={"passed": True, **_in_date_notes(item.id, last_valid, reference_day, language)}
                )

    return result


def _add_months(value: str, months: int) -> Optional[str]:
    """'YYYY-MM' or 'YYYY-MM-DD' moved on by whole months, at the same
    precision. A day the target month lacks (29 Feb) falls to its last day."""
    try:
        parts = [int(p) for p in value.strip().split("-")]
    except ValueError:
        return None
    if len(parts) not in (2, 3):
        return None
    total = parts[0] * 12 + (parts[1] - 1) + months
    year, month = divmod(total, 12)
    month += 1
    if len(parts) == 2:
        return f"{year:04d}-{month:02d}"
    last = expiry_last_valid_day(f"{year:04d}-{month:02d}")
    return date(year, month, min(parts[2], last.day)).isoformat()


def _english_date(value: str) -> str:
    day = expiry_last_valid_day(value)
    return day.strftime("%b %Y") if len(value) == 7 else day.strftime("%d %b %Y")


def _hindi_short_date(value: str) -> str:
    day = expiry_last_valid_day(value)
    return f"{_HINDI_MONTHS[day.month - 1]} {day.year}" if len(value) == 7 else _hindi_date(day)


def _battery_age_notes(
    dated_from: str, base: str, due: date, years: int, today: date, language: Optional[str]
) -> dict:
    """The verdict on a battery that carries no expiry, in words that say
    how the date was reached — a guess dressed up as a printed expiry would
    be a lie to the person relying on it."""
    expired = due < today
    soon = not expired and (due - today).days <= 90
    when = due.strftime("%d %b %Y")
    if dated_from == "install":
        english = (
            f"Battery installed {_english_date(base)}; its {years}-year life "
            + (f"ended on {when}. Replace the battery before this AED is relied on in an emergency."
               if expired else f"runs to {when}.")
        )
    elif expired:
        english = (
            f"This battery has no expiry date printed, only the date it was made ({_english_date(base)}). "
            f"Counting ZOLL's {years}-year battery life from then, it was due for replacement by {when}. "
            "Replace it, unless you have a record that it was installed later."
        )
    else:
        english = (
            f"Battery made {_english_date(base)}, with no expiry date printed. Counting ZOLL's {years}-year "
            f"battery life from then, plan to replace it by {when}."
        )
    if soon:
        english += " That's within three months — order a replacement battery soon."

    hindi = None
    if language == "hi":
        due_hi = _hindi_date(due)
        if dated_from == "install":
            hindi = f"बैटरी {_hindi_short_date(base)} को लगाई गई थी; इसकी {years} साल की लाइफ़ " + (
                f"{due_hi} को पूरी हो चुकी है। किसी इमरजेंसी में इस AED पर भरोसा करने से पहले बैटरी बदलें।"
                if expired
                else f"{due_hi} तक है।"
            )
        elif expired:
            hindi = (
                f"इस बैटरी पर एक्सपायरी डेट नहीं छपी, सिर्फ़ बनने की तारीख ({_hindi_short_date(base)}) है। उससे "
                f"{years} साल की बैटरी लाइफ़ गिनें तो इसे {due_hi} तक बदल देना था। अगर इसे बाद में लगाने का "
                "रिकॉर्ड नहीं है, तो बैटरी बदलें।"
            )
        else:
            hindi = (
                f"बैटरी {_hindi_short_date(base)} में बनी है, इस पर एक्सपायरी डेट नहीं छपी। {years} साल की "
                f"बैटरी लाइफ़ गिनें तो इसे {due_hi} तक बदल दें।"
            )
        if soon:
            hindi += " जल्द नई बैटरी मँगवा लें।"
    return {"notes": english, "notes_hi": hindi}


def _date_battery_by_age(
    result: ChecklistAnalysisResult,
    profile: DeviceProfile,
    *,
    today: Optional[date],
    language: Optional[str],
    overruled,
) -> ChecklistAnalysisResult:
    """Date a battery that carries no expiry of its own (Powerheart G3/G5).

    Its maker prints only the manufacture date and guarantees the battery
    for a fixed life from installation. So: from the installation date when
    one is written on it, else from manufacture. Counting from manufacture
    can only come out EARLIER than the truth — a battery is never fitted
    before it is made — which is the safe direction for a safety device:
    it may ask for a battery a little early, never pass a dead one."""
    reference_day = today or datetime.now(timezone.utc).date()

    def usable(value: Optional[str]) -> Optional[str]:
        value = (value or "").strip()
        if not validators.is_plausible_expiry(value, today=reference_day):
            return None
        # A date in the future can't be when a battery was made or fitted.
        first_day = date.fromisoformat(value if len(value) == 10 else f"{value}-01")
        return value if first_day <= reference_day else None

    made = usable(result.manufacture_date)
    if made is None and not result.manufacture_date and not result.install_date and result.expiry_date:
        # This battery has no expiry, so a lone date read as one is the
        # manufacture date the model was told to look for.
        made = usable(result.expiry_date)
        if made:
            overruled("battery_date_read_as_manufacture")
    installed = usable(result.install_date)
    if installed and made and installed[:7] < made[:7]:
        installed = None  # fitted before it was made: a misread

    base, dated_from = (installed, "install") if installed else (made, "manufacture")
    if base is None:
        overruled("battery_date_unread")
        return result.model_copy(
            update={
                "passed": False,
                "expiry_date": None,
                "confidence": min(result.confidence, 0.4),
                **_override_notes("battery_made_unread", language),
            }
        )

    due_str = _add_months(base, profile.battery_life_months)
    due = expiry_last_valid_day(due_str)
    overruled(f"battery_dated_from_{dated_from}")
    logger.info("checklist.battery_dated_by_age", profile=profile.id, base=base, dated_from=dated_from, due=due_str)
    dated = result.model_copy(
        update={
            "expiry_date": due_str,
            "manufacture_date": made or result.manufacture_date,
            "install_date": installed,
        }
    )
    years = profile.battery_life_months // 12

    if due < reference_day:
        overruled("expired_model_said_pass" if result.passed else "expired")
        return dated.model_copy(
            update={"passed": False, **_battery_age_notes(dated_from, base, due, years, reference_day, language)}
        )
    if result.damage_seen:
        return dated.model_copy(update={"passed": False})
    if result.date_legible is False or result.confidence < 0.5:
        return dated.model_copy(update={"passed": False})
    return dated.model_copy(
        update={"passed": True, **_battery_age_notes(dated_from, base, due, years, reference_day, language)}
    )
