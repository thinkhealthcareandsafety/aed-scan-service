"""
Expiry date parser for AED label text — deterministic second opinion used to
cross-check Gemini's own date reading. Adapted unchanged from
github.com/thinkhealthcareandsafety/Aed-inspection-platform.
"""
from __future__ import annotations

import re
from typing import Optional

_PATTERNS = [
    (r"\b(20\d{2})[-/\.](0[1-9]|1[0-2])[-/\.](\d{1,2})\b", "%Y-%m-%d"),
    (r"\b(0[1-9]|1[0-2])[-/\.](\d{1,2})[-/\.](20\d{2})\b", "%m-%d-%Y"),
    (r"\b(20\d{2})[-/\.](0[1-9]|1[0-2])\b", "%Y-%m"),
    (r"\b(0[1-9]|1[0-2])[-/\.](20\d{2})\b", "%m-%Y"),
    (r"\b([A-Za-z]{3})[-/ ]*(20\d{2})\b", "%b %Y"),
    (r"[Uu][Ss][Ee][\s]+[Bb][Yy][\s]+(20\d{2})[-/\.](0[1-9]|1[0-2])", "%Y-%m"),
    (r"[Ee][Xx][Pp][:\s]*(20\d{2})[-/\.](0[1-9]|1[0-2])", "%Y-%m"),
]

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_expiry_date(text: str) -> Optional[str]:
    """Extract and normalise expiry date from raw label text. Returns ISO
    string "YYYY-MM" or "YYYY-MM-DD", or None."""
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
