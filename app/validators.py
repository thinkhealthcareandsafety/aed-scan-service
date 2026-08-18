"""
Plausibility checks for AI-read AED data.

Gemini can misread fine print (a "5" for an "S", a stray digit, a
transposed date). These checks don't verify a read is *correct* — only
that it isn't obvious garbage — before it's trusted. Adapted unchanged
from github.com/thinkhealthcareandsafety/Aed-inspection-platform.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

_SERIAL_MIN_LEN = 4
_SERIAL_MAX_LEN = 30
_SERIAL_ALNUM_RE = re.compile(r"[A-Za-z0-9]")

_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")

_MIN_PLAUSIBLE_YEAR_OFFSET = -5
_MAX_PLAUSIBLE_YEAR_OFFSET = 20


def is_plausible_serial(serial: Optional[str]) -> bool:
    if not serial:
        return False
    trimmed = serial.strip()
    if not (_SERIAL_MIN_LEN <= len(trimmed) <= _SERIAL_MAX_LEN):
        return False
    return bool(_SERIAL_ALNUM_RE.search(trimmed))


def is_plausible_expiry(date_str: Optional[str], *, today: Optional[date] = None) -> bool:
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
    match = _ISO_DATE_RE.match(date_str.strip())
    if not match:
        return None
    year, month, _day = match.groups()
    return f"{year}-{month}"


def expiry_cross_check_agrees(
    gemini_normalised: Optional[str], raw_label_text: Optional[str]
) -> bool:
    if not gemini_normalised or not raw_label_text:
        return True

    from app.date_parser import parse_expiry_date

    parsed = parse_expiry_date(raw_label_text)
    if not parsed:
        return True

    gemini_ym = _to_year_month(gemini_normalised)
    parsed_ym = _to_year_month(parsed) or parsed
    if gemini_ym is None:
        return True

    return gemini_ym == parsed_ym
