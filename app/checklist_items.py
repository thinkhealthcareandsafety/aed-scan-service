"""
AED inspection-scan — item catalogue.

Phase 1 beta covered the 3 items AEDSmartX's manual inspection screen
already tracks (AED serial number, pads expiry, battery expiry), scoped to
Philips HeartStart FRx / HS1 — the pilot account's actual devices. Adapted
from github.com/thinkhealthcareandsafety/Aed-inspection-platform.

readiness_indicator (the status-light blink check) was deliberately left
out of that first phase — checked its real usage history in the sibling
aed-readiness-campaign project before adding it here and found zero
confirmed real-world results, only dev/test submissions. Its prompt and the
per-brand device-context guidance below are ported verbatim from that
project anyway, since the prompt engineering itself has been iterated on
(tried, dropped, and deliberately restored) even though field validation
hasn't happened yet. Client-side treats this item's result as a suggestion
the inspector still has to confirm, never an auto-decision — see
AiScanPanel.jsx.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_PHILIPS_CONTEXT = (
    "The device being inspected is a Philips HeartStart FRx or HeartStart "
    "HS1 AED — both bright orange/yellow rugged cases with a green "
    "flashing status light on the front and a single push-button "
    "operation. Use this knowledge of Philips FRx/HS1 label placement and "
    "part appearance to read the image accurately, but do not assume a "
    "device is a Philips unit if the image clearly shows otherwise — "
    "flag that in `notes` instead of guessing."
)

# Ported from the sibling aed-readiness-campaign project's
# lib/inspectionChecklist.js MODEL_CONTEXT map — brand-specific part-layout
# guidance so the readiness-indicator prompt looks in the right place
# instead of assuming every AED is a Philips unit.
MODEL_CONTEXT: dict[str, str] = {
    "frx": (
        "The device is a Philips HeartStart FRx — bright orange/yellow rugged case, single "
        "push-button operation, small green flashing status light on the front. The battery "
        "pack (model M5070A, black/dark-grey) slides into the back/bottom of the unit. The "
        "pads connector is a small socket on the top edge."
    ),
    "hs1": (
        "The device is a Philips HeartStart HS1 — bright orange/yellow case. The battery pack "
        "(model M5071A, black/dark-grey) slides into the back/bottom of the unit. The pads are "
        "a sealed cartridge that slots into the top of the case as one piece (pads + connector "
        "combined). The small status light/window is near the carry-handle end of the case."
    ),
    "zollPlus": (
        "The device is a ZOLL AED Plus — rugged case, usually yellow or blue-grey. It takes 10 "
        "standard lithium batteries in a battery door on the back, not a slide-in proprietary "
        "pack. The pads are the pre-connected CPR-D-padz type — the padz packet's cable plugs "
        "into a port on top of the unit, so 'pads connected' means that cable is seated in that "
        "port, not that a separate connector clicks in. The status window on the front shows a "
        "check mark (ready) or a replace icon (needs service)."
    ),
    "g5": (
        "The device is a Cardiac Science Powerheart G5 — case is typically white/grey with a "
        "green carry handle. The battery is a rectangular pack that slides into a compartment "
        "on the back. The pads connector is a cable that plugs into a port on the front face. "
        "The status indicator is an icon-based screen or a simple light near the handle."
    ),
    "defibtech": (
        "The device is a Defibtech Lifeline/Lifeline AUTO — white case with an orange accent "
        "stripe. The battery pack slides into the bottom of the unit. The pads connector plugs "
        "into a port on the top. The status indicator is a small light near the carry handle: "
        "green means ready, red means it needs service."
    ),
    "crPlus": (
        "The device is a Physio-Control/Stryker LIFEPAK CR Plus — compact case, usually "
        "blue-grey. The battery is a rectangular pack in a compartment on the back. The pads "
        "are pre-connected (QUIK-COMBO), permanently wired to the unit — 'pads connected' means "
        "the pads packet's own connector is seated in the unit's port. The status indicator is "
        "a small screen or light on the front."
    ),
    "lp1000": (
        "The device is a Physio-Control/Stryker LIFEPAK 1000 — rugged case, usually orange or "
        "yellow. The battery is a rectangular pack that slides into the back. The pads "
        "(QUIK-COMBO) connector plugs into a port on the front. The status indicator is a "
        "status light and/or small LCD screen on the front panel."
    ),
}
DEFAULT_MODEL_CONTEXT = (
    "This unit's exact AED brand/model wasn't specified, or doesn't match a known preset — do "
    "not assume any particular brand's part layout. Identify the battery compartment, pads "
    "connector, cabinet/case, and status light generically from what's actually visible in the "
    "photo, and if the device's make/model would help give a more precise reading, say so in "
    "notes."
)


def model_context_for(model: Optional[str]) -> str:
    return MODEL_CONTEXT.get(model or "", DEFAULT_MODEL_CONTEXT)


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    title: str
    prompt: str
    media_type: str = "image"  # "image" or "video" (submitted as a browser-extracted frame sequence)


CHECKLIST_ITEMS: list[ChecklistItem] = [
    ChecklistItem(
        id="serial_number",
        title="Serial number",
        prompt=(
            f"{_PHILIPS_CONTEXT}\n\n"
            "Find the manufacturer serial number label. On the FRx it is on "
            "the back of the case; on the HS1 it is on the underside/back "
            "near the battery compartment, often printed near a barcode. "
            "Read the serial number exactly as printed (letters and digits, "
            "keep leading zeros). This is fine print — if it is too small, "
            "angled, glared, or blurry to read with certainty, do not "
            "guess: leave serial_number null, set passed=false, and explain "
            "in notes what the inspector should change (move closer, "
            "reduce glare, hold steady)."
        ),
    ),
    ChecklistItem(
        id="pads_expiry",
        title="Pads expiry",
        prompt=(
            f"{_PHILIPS_CONTEXT}\n\n"
            "Find the electrode pads label — either on the sealed pads "
            "cartridge/pouch itself or the pads connector cassette. Read "
            "the expiry date exactly as printed into expiry_raw_text, and "
            "also set expiry_date normalised to YYYY-MM or YYYY-MM-DD. "
            "This is fine print — if unclear, leave both null, set "
            "passed=false, and explain what to fix in notes."
        ),
    ),
    ChecklistItem(
        id="battery_expiry",
        title="Battery expiry",
        prompt=(
            f"{_PHILIPS_CONTEXT}\n\n"
            "Find the battery label (Philips FRx/HS1 batteries are usually "
            "a black or dark-grey pack, model M5070A or similar, that slides "
            "into the back/bottom of the unit). Read the expiry date "
            "exactly as printed into expiry_raw_text, and also set "
            "expiry_date normalised to YYYY-MM or YYYY-MM-DD. This is fine "
            "print — if unclear, leave both null, set passed=false, and "
            "explain what to fix in notes.\n\n"
            "Also try to read the battery's LOT number and serial number if "
            "visible on the same label — set lot_number and "
            "battery_serial_number if legible. These two fields are "
            "optional: never fail the check or lower pass/confidence just "
            "because they are missing or unreadable — only the expiry date "
            "is mandatory for this item."
        ),
    ),
    ChecklistItem(
        id="readiness_indicator",
        title="Readiness indicator",
        media_type="video",
        prompt=(
            "WHERE TO LOOK — this is the single most common mistake, avoid it: the readiness "
            "indicator is a SMALL round status LED or status window, not a large power button. "
            "Locate that small indicator specifically before judging anything — ignore any "
            "large ON/OFF button's own color/state entirely, even if it is lit or green.\n\n"
            "HOW TO JUDGE IT — a healthy, ready-to-use unit blinks that small LED green on a "
            "slow cycle; the gap between flashes varies by unit and can be as long as 4-5 "
            "seconds, so a short clip may only catch ONE flash, or catch it right at the very "
            "start or end of the clip — that is completely normal and is NOT a fault. Watch "
            "every frame carefully, start to finish. If you see even ONE distinct green flash "
            "(or a steady green ready icon on a status window) anywhere in the sequence, that "
            "alone is sufficient evidence: set status='ready', passed=true. Only set "
            "status='fault' if that small indicator is clearly, unambiguously showing a "
            "red/service-needed state, or you can positively confirm it stays completely "
            "dark/off the whole time with the indicator plainly in frame and in focus "
            "throughout. Do not require seeing a full on-off-on cycle — one confirmed green "
            "flash is enough to pass.\n\n"
            "If you cannot clearly identify the small indicator's position or color at all "
            "(e.g. it's completely out of frame, far too dark to make out any color, or too "
            "blurry/shaky to tell), set status='unclear', passed=false, and say specifically in "
            "notes what to fix — e.g. 'record at least 10 seconds since blinks can be up to 5 "
            "seconds apart', 'move closer to the small status indicator, not the power button', "
            "or 'hold the camera steady and well lit'. Reserve 'unclear' for genuinely unusable "
            "footage — if the indicator is visible at all, prefer making a 'ready'/'fault' call "
            "over 'unclear'."
        ),
    ),
]

_BY_ID = {item.id: item for item in CHECKLIST_ITEMS}


def get_item(item_id: str) -> Optional[ChecklistItem]:
    return _BY_ID.get(item_id)
