"""
Plausibility checks for AI-read AED data.

Gemini can misread fine print (a "5" for an "S", a stray digit, a
transposed date). These checks don't verify a read is *correct* — only
that it isn't obvious garbage — before it's allowed to count toward
consensus. Deliberately generic (no per-manufacturer format rules), in
keeping with this app's "any brand, no plugin" design.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

_SERIAL_MIN_LEN = 4
_SERIAL_MAX_LEN = 30
_SERIAL_ALNUM_RE = re.compile(r"[A-Za-z0-9]")

_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")

# How far from today an expiry date can be and still be believed.
# Backwards: far. A neglected AED with pads that ran out years ago is exactly
# what an inspection exists to find — the window used to stop at five years,
# so pads that expired in 2019 or 2020 were told their photo "looked
# implausible" and to retake it, instead of being told to replace them.
# Forwards: near. No AED pads or battery is sold with more than about five
# years of life, so a date further out than this is a misread digit (an
# 8 for a 3), and believing it would pass a dead consumable.
_MIN_PLAUSIBLE_YEAR_OFFSET = -20
_MAX_PLAUSIBLE_YEAR_OFFSET = 8


# What the model sometimes copies along with the serial: the GS1 field code
# "(21)" printed in front of it on barcode labels (a ZOLL AED Plus reads
# "(21) X14K718292"), or an "SN:" / "S/N" / "Serial No." caption.
#
# A caption is only stripped when it is punctuated as one ("SN:", "S/N",
# "SN "): some serials genuinely begin with the letters SN.
_SERIAL_PREFIX_RE = re.compile(
    r"^\s*(?:\(21\)|serial\s*(?:no\.?|number)?\s*[:#]|s\s*/\s*n\s*[:#.]?|sn\s*[:#.]|sn\s+)\s*",
    re.IGNORECASE,
)


def normalise_serial(serial: Optional[str]) -> Optional[str]:
    """The serial alone, without a GS1 '(21)' code or an 'SN:' caption."""
    if serial is None:
        return None
    cleaned = serial.strip()
    for _ in range(2):  # "(21) SN: X14..." is two prefixes deep
        stripped = _SERIAL_PREFIX_RE.sub("", cleaned, count=1).strip()
        if stripped == cleaned or not stripped:
            break
        cleaned = stripped
    return cleaned


def is_plausible_serial(serial: Optional[str]) -> bool:
    """Reject empty, too-short/long, or non-alphanumeric "serial" reads."""
    if not serial:
        return False
    trimmed = serial.strip()
    if not (_SERIAL_MIN_LEN <= len(trimmed) <= _SERIAL_MAX_LEN):
        return False
    return bool(_SERIAL_ALNUM_RE.search(trimmed))


def is_plausible_expiry(date_str: Optional[str], *, today: Optional[date] = None) -> bool:
    """Reject a normalised expiry string that isn't a real, plausible date.

    Accepts "YYYY-MM" or "YYYY-MM-DD". Checks the month is 1-12, the day
    (if present) is a real day for that month/year, and the year falls in a
    plausible window around today rather than decades off (a strong signal
    of a misread digit rather than a real expiry).
    """
    if not date_str:
        return False
    match = _ISO_DATE_RE.match(date_str.strip())
    if not match:
        return False

    year, month, day = match.groups()
    year_i, month_i = int(year), int(month)
    if not (1 <= month_i <= 12):
        return False

    try:
        if day is not None:
            date(year_i, month_i, int(day))
        else:
            date(year_i, month_i, 1)
    except ValueError:
        return False

    reference_year = (today or date.today()).year
    return (
        reference_year + _MIN_PLAUSIBLE_YEAR_OFFSET
        <= year_i
        <= reference_year + _MAX_PLAUSIBLE_YEAR_OFFSET
    )


def _to_year_month(date_str: str) -> Optional[str]:
    """Truncate a validated 'YYYY-MM' or 'YYYY-MM-DD' string to 'YYYY-MM'."""
    match = _ISO_DATE_RE.match(date_str.strip())
    if not match:
        return None
    year, month, _day = match.groups()
    return f"{year}-{month}"


def expiry_cross_check_agrees(
    gemini_normalised: Optional[str], raw_label_text: Optional[str]
) -> bool:
    """Independently re-derive a date from the raw label text Gemini
    transcribed and compare it against Gemini's own normalised value.

    Uses `date_parser.parse_expiry_date` — a deterministic regex parser,
    unrelated to whatever reasoning Gemini used to normalise its answer —
    as a second opinion. Returns True (agrees / no basis to disagree)
    whenever there's nothing to cross-check against; only returns False on
    an *active* disagreement between the two independent reads, since the
    regex parser failing to extract anything from a rough transcription is
    expected and shouldn't block an otherwise good Gemini read.
    """
    if not gemini_normalised or not raw_label_text:
        return True

    from app.utils.date_parser import parse_all_expiry_dates

    # The transcription lists every date on the label (manufacture as well as
    # expiry), so agreement means the chosen date appears among them — not
    # that it happens to be the first one written down.
    parsed_all = parse_all_expiry_dates(raw_label_text)
    if not parsed_all:
        return True

    gemini_ym = _to_year_month(gemini_normalised)
    if gemini_ym is None:
        return True

    return any((_to_year_month(p) or p) == gemini_ym for p in parsed_all)
