"""An expired consumable must FAIL, decided by the server's clock.

Production passed pads marked 2026-03 and 2026-08 in late September 2026,
because the model has no grounded sense of the current date. These tests pin
the deterministic verdict so it can never depend on the model again. The date
is injected, so they don't start failing as the real calendar moves on.
"""
from datetime import date

from app.services.checklist_items import get_item
from app.services.gemini_checklist_service import (
    ChecklistAnalysisResult,
    _apply_deterministic_checks,
)
from app.utils.date_parser import expiry_last_valid_day, is_expired

TODAY = date(2026, 9, 28)


def verdict(item_id, expiry, model_passed=True, raw=None):
    result = ChecklistAnalysisResult(
        passed=model_passed,
        confidence=0.95,
        notes="model notes",
        expiry_date=expiry,
        expiry_raw_text=raw or f"USE BY {expiry}",
    )
    return _apply_deterministic_checks(get_item(item_id), result, today=TODAY)


# ── the production failures ─────────────────────────────────────────────────

def test_pads_expired_six_months_ago_fail_even_if_the_model_says_pass():
    r = verdict("pads_expiry", "2026-03", model_passed=True)
    assert r.passed is False
    assert "expired" in r.notes.lower()


def test_pads_expired_last_month_fail():
    assert verdict("pads_expiry", "2026-08").passed is False


def test_expired_battery_fails_too():
    assert verdict("battery_expiry", "2025-12-31").passed is False


# ── month-only labels mean "usable through that month" ─────────────────────

def test_current_month_is_still_valid():
    # USE BY 2026-09 on 28 Sep 2026 is good until the 30th — not a false alarm.
    assert verdict("pads_expiry", "2026-09").passed is True


def test_month_only_resolves_to_its_last_day():
    assert expiry_last_valid_day("2026-09") == date(2026, 9, 30)
    assert expiry_last_valid_day("2026-12") == date(2026, 12, 31)
    assert expiry_last_valid_day("2028-02") == date(2028, 2, 29)  # leap year
    assert expiry_last_valid_day("2027-02") == date(2027, 2, 28)


# ── full dates and the day boundary ────────────────────────────────────────

def test_full_date_yesterday_is_expired_today_is_not():
    assert verdict("pads_expiry", "2026-09-27").passed is False
    assert verdict("pads_expiry", "2026-09-28").passed is True


def test_future_expiry_passes():
    assert verdict("battery_expiry", "2031-11-30",
                   raw="factory 2026-01-13 / install-before 2031-11-30").passed is True


# ── the server never upgrades a model FAIL to a PASS ───────────────────────

def test_valid_date_does_not_override_a_model_fail():
    # A legible, in-date label on a physically damaged pouch is still a fail.
    assert verdict("pads_expiry", "2031-01", model_passed=False).passed is False


def test_is_expired_uses_end_of_month():
    assert is_expired("2026-09", today=TODAY) is False
    assert is_expired("2026-08", today=TODAY) is True
