"""
AED Inspection Checklist — item catalogue.

10 discrete checklist items in 3 sections. Each is satisfied by a single
uploaded photo (or, for the readiness indicator, a short video) and one
Gemini call.

The prompts here are the device-NEUTRAL half of each instruction: the task
and how to judge it. Where things are on a particular unit, and what "ready"
or "fitted" looks like on it, comes from that unit's profile in
device_profiles.py. They used to be one Philips-only text, which is how a
ZOLL came to be judged against a Philips.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

MediaType = Literal["image", "video"]

# Medical consumables carry SEVERAL dates, and picking the wrong one is the
# most damaging mistake this system can make: reading a manufacture date as an
# expiry told a customer that a battery built in January 2026 had already
# died, when it had five years of life left.
#
# The rule is semantic, not "take the latest date". For a safety device the
# unsafe direction is reporting an expiry LATER than the truth — that tells
# someone a dead battery is fine. So: identify dates by their ISO 15223-1
# symbols, exclude manufacture outright, and where two genuine expiry-type
# dates disagree, take the EARLIER one and say so.
_DATE_SYMBOLS = """Medical device labels print more than one date, each tagged with a standard ISO 15223-1 symbol. Identify them by symbol and wording, never by which number is largest:
- A FACTORY building icon marks the DATE OF MANUFACTURE. This is NOT an expiry. Never report it as one. Put it in manufacture_date.
- An HOURGLASS marks the USE BY / EXPIRY date.
- An ARROW POINTING INTO A BRACKET, or the words 'Install before', mark the INSTALL-BEFORE date. On AED batteries this is usually the date that matters, and the one to report.
- Printed words such as 'EXP', 'Use by', 'Install before' or 'Replace ... on or before' override any symbol.
- Barcode labels often repeat dates in GS1 form: '(17)YYMMDD' is the expiry/use-by date, '(15)YYMMDD' a best-before or install-by date, and '(11)YYMMDD' the production date; '(10)' introduces the LOT and '(21)' the serial number. A '(17)' or '(15)' date confirms the expiry; a '(11)' date is a manufacture date.
- A date written as NN/NN/YYYY is ambiguous when both numbers are 12 or less (12/05/2029 could be 12 May or 5 December): report the EARLIER of the two readings and mention the ambiguity in notes. When one number is over 12, it is the day.

Set expiry_date to the USE BY / INSTALL BEFORE / REPLACE-BY date. Set manufacture_date to the factory date when one is visible. If two genuine expiry-type dates are present and you cannot tell which governs, report the EARLIER one and explain the ambiguity in notes — never the later one. If the only date you can read is a manufacture date, set expiry_date to null, passed=false, and say the expiry date was not visible.

Copy every date you can see, verbatim and with whatever labels it carries, into expiry_raw_text (for example: 'factory 2026-01-13 / install-before 2031-11-30', or '(17)221228 / 2022-12-28')."""

_FINE_PRINT = (
    "This is fine print — if it is too small, angled, glared or blurry to "
    "read with certainty, do not guess: leave the field null, set "
    "passed=false, and say in notes what the inspector should change (move "
    "closer, reduce glare, hold steady)."
)

# The model is good at reading a label and poor at calendars: it has failed
# in-date pads as "expiring next month" and passed expired ones it thought
# were in the future. So it only reports what it sees; the service decides
# expired or not, against today's date.
_READ_DONT_JUDGE = (
    "Set date_legible=true only if you read the expiry date with certainty "
    "(false if you are unsure of any digit). Set damage_seen=true if the "
    "item is visibly damaged, opened, swollen, leaking or corroded, false if "
    "it looks intact, null if you cannot see it. Do NOT decide whether the "
    "date has passed or is coming up — the app checks the date against "
    "today itself. Set passed=true if you read the expiry date with "
    "certainty and see no damage."
)


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    section: int
    order: int
    title: str
    description: str
    media_type: MediaType
    required: bool
    prompt: str


CHECKLIST_ITEMS: list[ChecklistItem] = [
    # ── Section 1 — Consumables & identification ────────────────────────
    ChecklistItem(
        id="serial_number",
        section=1,
        order=2,
        title="Serial number",
        description="Photo of the manufacturer serial number label.",
        media_type="image",
        required=True,
        prompt=(
            "Find the AED manufacturer's serial number label and read the "
            "serial number exactly as printed (letters, digits and hyphens; "
            "keep leading zeros). Do not report a REF/model number, a LOT "
            "number or a service number as the serial. On GS1 barcode labels "
            "the serial follows '(21)' — the '(21)' is a field code, not part "
            "of the serial. If the same label prints the unit's date of "
            "manufacture (beside the factory symbol), copy it into "
            "manufacture_date as YYYY-MM or YYYY-MM-DD — it tells the owner "
            f"how old the AED is. {_FINE_PRINT}"
        ),
    ),
    ChecklistItem(
        id="pads_expiry",
        section=2,
        order=3,
        title="Pads expiry",
        description="Photo of the electrode pads packaging expiry date.",
        media_type="image",
        required=True,
        prompt=(
            "Find the expiry date of the electrode pads, on their cartridge, "
            "case or sealed pack.\n\n"
            f"{_DATE_SYMBOLS}\n\n"
            f"Normalise expiry_date to YYYY-MM or YYYY-MM-DD. {_FINE_PRINT}\n\n"
            f"{_READ_DONT_JUDGE}"
        ),
    ),
    ChecklistItem(
        id="battery_expiry",
        section=2,
        order=4,
        title="Battery expiry",
        description="Photo of the battery label expiry date. Lot & serial number are optional.",
        media_type="image",
        required=True,
        prompt=(
            "Find the date by which the AED's battery must be installed or "
            "replaced.\n\n"
            f"{_DATE_SYMBOLS}\n\n"
            f"Normalise expiry_date to YYYY-MM or YYYY-MM-DD. {_FINE_PRINT}\n\n"
            "Also try to read the battery's LOT number and serial number if "
            "visible on the same label — set lot_number and "
            "battery_serial_number if legible. These two fields are "
            "optional: never fail the check or lower pass/confidence just "
            "because they are missing or unreadable — only the expiry date "
            "is mandatory for this item.\n\n"
            f"{_READ_DONT_JUDGE}"
        ),
    ),
    # ── Section 2 — Physical status ──────────────────────────────────────
    ChecklistItem(
        id="battery_attached",
        section=2,
        order=5,
        title="Battery attached",
        description="Photo confirming the battery is fully seated in the machine.",
        media_type="image",
        required=True,
        prompt=(
            "Determine whether the battery is fully and correctly installed "
            "in the AED — seated and latched, with no visible gap, tilt or "
            "raised edge. Set passed=true only if the battery is clearly, "
            "fully installed. If the photo doesn't show the battery area "
            "clearly enough to judge, set passed=false and ask for a clearer "
            "angle in notes."
        ),
    ),
    ChecklistItem(
        id="pads_connected",
        section=2,
        order=6,
        title="Pads connected",
        description="Photo confirming the pads are connected to the machine.",
        media_type="image",
        required=True,
        prompt=(
            "Determine whether the electrode pads are connected to the AED, "
            "so it could deliver a shock straight away. Set passed=true only "
            "if you can actually SEE the connection made — the plug or "
            "cartridge the device notes describe, seated in place. Never "
            "assume a connection you cannot see. If the connector or "
            "cartridge well is visible and empty, the pads are NOT connected: "
            "set passed=false and say plainly in notes that the pads are not "
            "connected and must be plugged in before the AED can be relied "
            "on. If the photo doesn't show the connector clearly enough to "
            "judge, set passed=false and ask for a closer photo of it."
        ),
    ),
    ChecklistItem(
        id="readiness_indicator",
        section=1,
        order=1,
        title="Readiness indicator",
        description="Short video of the readiness indicator.",
        media_type="video",
        required=True,
        # A "ready" is only as good as the frame it can point to. The old
        # wording told the model flashes could fall between frames and
        # that gaps were normal — so a clip where the light never came on
        # could still pass. Now it must name the frames that show the
        # ready signal, and the service checks them (and, for units that
        # blink, the flashes found in the video itself).
        prompt=(
            "Decide whether this AED's readiness indicator shows it is ready "
            "for use. The device notes below say where the indicator is on "
            "THIS model and what 'ready' and 'fault' look like on it — find "
            "that indicator and judge only it, never any other light, "
            "button or reflection.\n\n"
            "Look at every frame. In ready_frames, list the numbers of the "
            "frames in which you can actually SEE the ready signal: for a "
            "light that blinks, the frames where that light is visibly lit; "
            "for a symbol, the frames where the ready symbol is clearly "
            "readable. (If you were given a video rather than numbered "
            "frames, list the whole seconds instead.) List nothing you would "
            "have to assume.\n"
            "- status='ready', passed=true: ONLY if ready_frames is not "
            "empty. Never conclude that a light blinked between frames — if "
            "no frame shows it lit, readiness is not confirmed.\n"
            "- status='fault', passed=false: a fault signal is visible (see "
            "the device notes), or the indicator is clearly in view and in "
            "focus through the whole clip and never shows the ready signal.\n"
            "- status='unclear', passed=false: the indicator can't be judged "
            "— out of frame, too far away, too dark, or blurred.\n\n"
            "In notes, say what you saw; if it is not ready, say exactly what "
            "to do next (for example: film just the indicator, up close and "
            "steady, for at least 10 seconds — or, if it shows a fault, that "
            "the unit needs service)."
        ),
    ),
    # ── Section 3 — Accessories & signage (all optional) ─────────────────
    ChecklistItem(
        id="child_key_pad",
        section=3,
        order=7,
        title="Child key / child pads",
        description="Photo of the infant/child key or child pads, if present.",
        media_type="image",
        required=False,
        prompt=(
            "Determine whether the AED's infant/child accessory is present "
            "and stored correctly — the device notes below say what it is on "
            "this unit. Set present=true/false, and passed=true if it is "
            "present and in good condition. This accessory is optional on "
            "many deployments — if it is genuinely absent, set present=false, "
            "passed=false and note 'not present' rather than treating the "
            "photo as unreadable."
        ),
    ),
    ChecklistItem(
        id="aed_cabinet",
        section=3,
        order=8,
        title="AED cabinet",
        description="Photo of the wall cabinet/case housing the AED.",
        media_type="image",
        required=False,
        prompt=(
            "Assess the AED wall cabinet or carry case: the door or lid "
            "closes properly, no cracked glass or broken latch, and it is "
            "visible and unobstructed. Set passed=true if it looks intact and "
            "usable, false with a reason in notes otherwise."
        ),
    ),
    ChecklistItem(
        id="first_response_kit",
        section=3,
        order=9,
        title="Fast response kit",
        description="Photo of the accompanying rescue kit (gloves, razor, scissors, mask).",
        media_type="image",
        required=False,
        prompt=(
            "Check whether a fast/first response kit (gloves, razor, "
            "scissors, CPR face shield or pocket mask) is present alongside "
            "the AED. Set present=true/false and passed=true if present and "
            "apparently complete or unopened. If absent, set present=false, "
            "passed=false and note 'not present'."
        ),
    ),
    ChecklistItem(
        id="emergency_contacts",
        section=3,
        order=10,
        title="Emergency contacts sticker",
        description="Photo confirming an emergency contact sticker is on the machine or cabinet.",
        media_type="image",
        required=False,
        prompt=(
            "Check whether an emergency contact sticker or label (a local "
            "emergency number such as 112 or 108 in India, 911 or 999 "
            "elsewhere, or a site contact) is fixed to the AED or its cabinet "
            "and legible. Set present=true/false and passed=true if present "
            "and legible. If absent or illegible, set present=false, "
            "passed=false and say why in notes."
        ),
    ),
]

_BY_ID = {item.id: item for item in CHECKLIST_ITEMS}


def get_item(item_id: str) -> Optional[ChecklistItem]:
    return _BY_ID.get(item_id)
