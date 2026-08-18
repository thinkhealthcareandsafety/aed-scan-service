"""
AED inspection-scan — item catalogue.

Phase 1 beta: only the 3 items AEDSmartX's manual inspection screen already
tracks (AED serial number, pads expiry, battery expiry). Scoped specifically
to the two devices the pilot account (Medic Assist Gmbh) actually has on
file: Philips HeartStart FRx / HS1. Adapted from the item catalogue at
github.com/thinkhealthcareandsafety/Aed-inspection-platform, trimmed from
10 items down to these 3, and with LED/status detection intentionally left
out of this phase — it's the least proven part of that system.
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


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    title: str
    prompt: str


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
]

_BY_ID = {item.id: item for item in CHECKLIST_ITEMS}


def get_item(item_id: str) -> Optional[ChecklistItem]:
    return _BY_ID.get(item_id)
