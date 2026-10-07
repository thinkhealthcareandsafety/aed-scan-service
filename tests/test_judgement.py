"""The model reports what it sees; the service decides.

Found by the evaluation run of 1 Oct 2026 (python-cv/eval):
  - pads that expired in 2019 and 2020 were read correctly, then refused as
    an "implausible" date and sent back for a retake — instead of being
    reported as expired, which is what the inspection is for;
  - in-date pads (Nov 2026) were read correctly and failed anyway, the
    model deciding they had "expired" or were "next month";
  - a ZOLL serial label passed a Philips HS1 inspection.
"""
from __future__ import annotations

from datetime import date

import pytest

from app.services import gemini_checklist_service as svc
from app.services.checklist_items import get_item
from app.services.device_profiles import GENERIC, PHILIPS_FRX, PHILIPS_HS1, ZOLL_AED_PLUS

TODAY = date(2026, 10, 1)


def _expiry(item="pads_expiry", expiry="2026-11", passed=False, legible=True, damage=False, confidence=0.9,
            raw=None, language=None, overrides=None):
    result = svc.ChecklistAnalysisResult(
        passed=passed, confidence=confidence, notes="The model's own words.", expiry_date=expiry,
        expiry_raw_text=raw, date_legible=legible, damage_seen=damage,
    )
    return svc._apply_deterministic_checks(
        get_item(item), result, today=TODAY, language=language, overrides=overrides
    )


# ── Old expiries are expiries, not misreads ────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("expiry", ["2019-05", "2020-09", "2010-01"])
def test_pads_that_expired_years_ago_are_reported_as_expired(expiry):
    checked = _expiry(expiry=expiry, passed=False)
    assert checked.passed is False
    assert checked.expiry_date == expiry  # kept: it feeds the replacement quote
    assert "expired" in checked.notes
    assert "implausible" not in checked.notes


@pytest.mark.unit
@pytest.mark.parametrize("expiry", ["2048-03", "2039-11"])
def test_a_date_no_consumable_could_last_to_is_a_misread(expiry):
    # No AED pads or battery lasts beyond ~5 years: a far-future date is a
    # misread digit, and believing it would pass a dead consumable.
    checked = _expiry(expiry=expiry, passed=True)
    assert checked.passed is False
    assert checked.expiry_date is None


# ── In date: the service decides, not the model's sense of the calendar ────

@pytest.mark.unit
def test_in_date_pads_the_model_misjudged_pass():
    overrides = []
    checked = _expiry(expiry="2027-08", passed=False, overrides=overrides)
    assert checked.passed is True
    assert checked.notes == "Pads in date until 31 Aug 2027."
    assert overrides == ["in_date_model_said_fail"]


@pytest.mark.unit
def test_pads_expiring_within_three_months_pass_with_a_reminder():
    checked = _expiry(expiry="2026-11", passed=False)
    assert checked.passed is True
    assert "within three months" in checked.notes


@pytest.mark.unit
def test_the_reminder_is_in_hindi_for_a_hindi_reader():
    checked = _expiry(item="battery_expiry", expiry="2026-11", language="hi")
    assert checked.notes_hi == "बैटरी 30 नवंबर 2026 तक वैध है। जल्द नई बैटरी मँगवा लें।"


@pytest.mark.unit
def test_damaged_pads_fail_whatever_the_date():
    overrides = []
    checked = _expiry(expiry="2028-01", passed=True, damage=True, overrides=overrides)
    assert checked.passed is False
    assert overrides == ["damage_seen"]


@pytest.mark.unit
def test_a_date_the_model_could_not_read_with_certainty_is_a_retake():
    checked = _expiry(expiry="2028-01", passed=True, legible=False)
    assert checked.passed is False


@pytest.mark.unit
def test_a_low_confidence_read_is_not_promoted_to_a_pass():
    checked = _expiry(expiry="2028-01", passed=False, confidence=0.3)
    assert checked.passed is False


@pytest.mark.unit
def test_with_no_legibility_answer_the_model_fail_stands():
    # Older answers (or a model that skips the field) keep the old rule:
    # the service never promotes a fail it has no evidence about.
    checked = _expiry(expiry="2028-01", passed=False, legible=None)
    assert checked.passed is False


@pytest.mark.unit
def test_an_expired_date_still_fails_even_if_the_model_passed_it():
    overrides = []
    checked = _expiry(expiry="2026-08", passed=True, overrides=overrides)
    assert checked.passed is False
    assert overrides == ["expired_model_said_pass"]


# ── A different maker's AED can't pass this unit's check ───────────────────

def _seen(item, profile, brand, passed=True, language=None, overrides=None):
    result = svc.ChecklistAnalysisResult(
        passed=passed, confidence=0.9, notes="Looks fine.", brand_seen=brand, serial_number="X14K718292"
    )
    return svc._apply_deterministic_checks(
        get_item(item), result, today=TODAY, profile=profile, language=language, overrides=overrides
    )


@pytest.mark.unit
def test_a_zoll_label_fails_a_philips_inspection_and_says_what_to_do():
    overrides = []
    checked = _seen("serial_number", PHILIPS_HS1, "ZOLL", overrides=overrides)
    assert checked.passed is False
    assert "ZOLL AED, not the Philips HeartStart HS1" in checked.notes
    assert "⋯ menu" in checked.notes
    assert overrides == ["different_brand"]


@pytest.mark.unit
def test_the_brand_advice_is_in_hindi_for_a_hindi_reader():
    checked = _seen("pads_connected", ZOLL_AED_PLUS, "Philips HeartStart", language="hi")
    assert checked.notes_hi and "मेनू" in checked.notes_hi


@pytest.mark.unit
@pytest.mark.parametrize(
    "item, profile, brand",
    [
        ("serial_number", PHILIPS_FRX, "Philips"),
        ("serial_number", PHILIPS_FRX, "HeartStart"),  # Philips' product line
        ("battery_attached", ZOLL_AED_PLUS, "Duracell"),  # the cells in a ZOLL, not an AED maker
        ("serial_number", PHILIPS_HS1, None),
        ("serial_number", GENERIC, "ZOLL"),  # no model chosen: nothing to contradict
        ("aed_cabinet", PHILIPS_HS1, "ZOLL"),  # a cabinet isn't the unit
    ],
)
def test_a_brand_check_never_fails_a_good_photo(item, profile, brand):
    assert _seen(item, profile, brand).passed is True


@pytest.mark.unit
def test_a_readiness_video_of_another_brand_is_unclear_not_a_fault():
    result = svc.ChecklistAnalysisResult(
        passed=True, confidence=0.9, notes="Green check.", status="ready", ready_frames=[2], brand_seen="ZOLL"
    )
    checked = svc._apply_deterministic_checks(get_item("readiness_indicator"), result, profile=PHILIPS_HS1)
    assert checked.passed is False
    assert checked.status == "unclear"


@pytest.mark.unit
def test_the_prompt_asks_for_the_brand_without_naming_one():
    prompt = svc._build_prompt(get_item("serial_number"), PHILIPS_HS1)
    assert "brand_seen" in prompt
    assert "ZOLL" not in prompt.upper()


@pytest.mark.unit
def test_date_prompts_ask_the_model_to_read_not_judge():
    prompt = svc._build_prompt(get_item("battery_expiry"), PHILIPS_HS1)
    assert "date_legible" in prompt and "damage_seen" in prompt
    assert "Do NOT decide whether the date has passed" in prompt
