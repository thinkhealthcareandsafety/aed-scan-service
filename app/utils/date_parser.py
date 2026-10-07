"""
Expiry date parser for AED label text.
Handles many date formats found on AED labels worldwide.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import List, Optional

# Ordered list of patterns, most-specific first. Full Y-M-D / M-D-Y dates
# MUST be tried before the bare Y-M / M-Y patterns below, otherwise the
# shorter pattern matches the leading "2025/06" of "2025/06/15" and the day
# is silently dropped.
_PATTERNS = [
    # 2025-06-15  |  2025/06/15
    (r"\b(20\d{2})[-/\.](0[1-9]|1[0-2])[-/\.](\d{1,2})\b", "%Y-%m-%d"),
    # 06/15/2025  |  06-15-2025
    (r"\b(0[1-9]|1[0-2])[-/\.](\d{1,2})[-/\.](20\d{2})\b", "%m-%d-%Y"),
    # 2025-06  |  2025/06  |  2025.06
    (r"\b(20\d{2})[-/\.](0[1-9]|1[0-2])\b", "%Y-%m"),
    # 06/2025  |  06-2025  |  06.2025
    (r"\b(0[1-9]|1[0-2])[-/\.](20\d{2})\b", "%m-%Y"),
    # JUN 2025  |  Jun-2025  |  JUN/2025
    (r"\b([A-Za-z]{3})[-/ ]*(20\d{2})\b", "%b %Y"),
    # USE BY 2025-06
    (r"[Uu][Ss][Ee][\s]+[Bb][Yy][\s]+(20\d{2})[-/\.](0[1-9]|1[0-2])", "%Y-%m"),
    # EXP 2025-06
    (r"[Ee][Xx][Pp][:\s]*(20\d{2})[-/\.](0[1-9]|1[0-2])", "%Y-%m"),
]

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_expiry_date(text: str) -> Optional[str]:
    """
    Extract and normalise expiry date from OCR text.
    Returns ISO string "YYYY-MM" or "YYYY-MM-DD", or None.
    """
    if not text:
        return None

    text = text.strip()

    for pattern, fmt in _PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        groups = match.groups()
        try:
            if fmt in ("%Y-%m", "%m-%Y"):
                if fmt == "%Y-%m":
                    year, month = int(groups[0]), int(groups[1])
                else:
                    month, year = int(groups[0]), int(groups[1])
                return f"{year:04d}-{month:02d}"

            elif fmt == "%b %Y":
                month_str = groups[0].lower()[:3]
                month = _MONTH_ABBR.get(month_str)
                if not month:
                    continue
                year = int(groups[1])
                return f"{year:04d}-{month:02d}"

            elif fmt == "%Y-%m-%d":
                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                return f"{year:04d}-{month:02d}-{day:02d}"

            elif fmt == "%m-%d-%Y":
                month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                return f"{year:04d}-{month:02d}-{day:02d}"

        except (ValueError, IndexError):
            continue

    return None


# GS1 human-readable element strings, as printed under barcodes on pads boxes
# and device labels: "(17)221228" is the expiry and "(11)230301" the
# production date, both YYMMDD. "(15)" is the best-before date, which is
# where a ZOLL AED 3 battery label prints its install-by date. A day of "00"
# means the end of that month. Unlike a printed date, a GS1 field says
# unambiguously what it is.
_GS1_DATE_RE = re.compile(r"\((1[157])\)\s?(\d{2})(\d{2})(\d{2})")


def parse_gs1_dates(text: str) -> dict:
    """{'17': expiry, '15': best-before, '11': production} from GS1 text,
    normalised like the other parsers ('YYYY-MM-DD', or 'YYYY-MM' when the
    day is '00')."""
    found: dict = {}
    if not text:
        return found
    for match in _GS1_DATE_RE.finditer(text):
        field_code, yy, mm, dd = match.groups()
        month = int(mm)
        if not 1 <= month <= 12:
            continue
        year = 2000 + int(yy)
        if dd == "00":
            value = f"{year:04d}-{month:02d}"
        else:
            try:
                value = date(year, month, int(dd)).isoformat()
            except ValueError:
                continue
        found.setdefault(field_code, value)
    return found


def parse_all_expiry_dates(text: str) -> List[str]:
    """Every date in `text`, normalised, in the order they appear.

    Labels carry several dates (manufacture, install-before, use-by), and the
    transcription now lists all of them. `parse_expiry_date` returns only the
    first match, which on a Philips battery is the manufacture date — so a
    cross-check against it vetoed correct readings. Anything comparing against
    the raw text must consider every date in it.
    """
    if not text:
        return []
    found: List[tuple] = []
    claimed: List[range] = []
    for pattern, fmt in _PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            span = range(match.start(), match.end())
            # A span already claimed by a more specific pattern (a full
            # Y-M-D) must not be re-read by a shorter one (its leading Y-M).
            if any(span.start < c.stop and c.start < span.stop for c in claimed):
                continue
            value = parse_expiry_date(match.group(0))
            if value:
                claimed.append(span)
                found.append((match.start(), value))
    # GS1 fields have no separators, so the patterns above never see them.
    for match in _GS1_DATE_RE.finditer(text):
        gs1 = parse_gs1_dates(match.group(0))
        for value in gs1.values():
            found.append((match.start(), value))
    found.sort(key=lambda item: item[0])
    seen: List[str] = []
    for _, value in found:
        if value not in seen:
            seen.append(value)
    return seen


def expiry_last_valid_day(date_str: Optional[str]) -> Optional[date]:
    """The last day on which a consumable with this expiry may still be used.

    A label printed 'YYYY-MM' means usable THROUGH that month, so a month-only
    expiry resolves to the month's final day. The previous implementation used
    the first day instead, which would have failed pads on 1 September that
    are legitimately good until the 30th.
    """
    if not date_str:
        return None
    try:
        parts = date_str.strip().split("-")
        if len(parts) == 2:
            year, month = int(parts[0]), int(parts[1])
            # The day before the first of next month.
            first_of_next = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
            return first_of_next - timedelta(days=1)
        return date.fromisoformat(date_str.strip())
    except ValueError:
        return None


def is_expired(date_str: Optional[str], *, today: Optional[date] = None) -> bool:
    """True once the last valid day of this expiry has passed."""
    last_day = expiry_last_valid_day(date_str)
    if last_day is None:
        return False
    return last_day < (today or date.today())
