"""The ZOLL AED 3 and the Powerheart G3 and G5 are judged as themselves.

Each differs from the units the app already knew in a way that would fail a
healthy unit, or pass a broken one, if it were described like them:

- AED 3: a status window that shows a green check or goes BLANK — no red X.
- G3/G5: a Rescue Ready indicator that is green or red, and that turns red
  for a few seconds whenever the lid is opened or closed.
- G3/G5 batteries print no expiry at all, only the date they were made;
  Cardiac Science/ZOLL guarantee them 4 years from installation. Asking for
  an expiry date that isn't there would be a retake forever.
- AED 3 batteries print their install-by date as GS1 '(15)YYMMDD'.
"""
from __future__ import annotations

from datetime import date

import pytest

from app.services import gemini_checklist_service as svc
from app.services import readiness_frames
from app.services.checklist_items import CHECKLIST_ITEMS, get_item
from app.services.device_profiles import POWERHEART_G3, POWERHEART_G5, ZOLL_AED_3, get_profile
from app.utils.date_parser import parse_gs1_dates

TODAY = date(2026, 10, 3)
POWERHEART = (POWERHEART_G3, POWERHEART_G5)
BATTERY = get_item("battery_expiry")
READINESS = get_item("readiness_indicator")


def _prompt(item_id: str, profile, frames=None) -> str:
    return svc._build_prompt(get_item(item_id), profile, frame_count=frames)


def _video(frames: int = 24) -> readiness_frames.ReadinessFrames:
    return readiness_frames.ReadinessFrames(frames=[b"x"] * frames, times=[i * 0.4 for i in range(frames)])


def _battery(profile, language=None, **fields):
    defaults = dict(passed=False, confidence=0.9, notes="Only a manufacture date.", date_legible=True)
    result = svc.ChecklistAnalysisResult(**{**defaults, **fields})
    return svc._apply_deterministic_checks(BATTERY, result, today=TODAY, profile=profile, language=language)


# ── The app's ids reach the right profile ───────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "model, profile",
    [("Zoll AED 3", ZOLL_AED_3), ("Zoll Powerheart G3", POWERHEART_G3), ("Zoll Powerheart G5", POWERHEART_G5)],
)
def test_each_new_model_id_gets_its_own_profile(model, profile):
    assert get_profile(model) is profile


@pytest.mark.unit
@pytest.mark.parametrize("profile", (ZOLL_AED_3, *POWERHEART), ids=lambda p: p.id)
@pytest.mark.parametrize("item", [item.id for item in CHECKLIST_ITEMS])
def test_no_prompt_for_these_units_describes_a_philips(profile, item):
    assert "Philips" not in _prompt(item, profile)


@pytest.mark.unit
@pytest.mark.parametrize("item", [item.id for item in CHECKLIST_ITEMS])
def test_no_aed3_prompt_borrows_the_aed_plus_layout(item):
    prompt = _prompt(item, ZOLL_AED_3)
    assert "123" not in prompt  # the AED Plus's ten photo cells
    assert "REPLACE BATTERIES ON OR BEFORE" not in prompt


# ── Readiness ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_aed3_readiness_is_a_green_check_and_a_blank_window_is_a_fault():
    prompt = _prompt("readiness_indicator", ZOLL_AED_3, frames=20)
    assert "right of the blue On/Off button" in prompt
    assert "GREEN CHECK MARK" in prompt
    assert "BLANK window" in prompt and "status='fault'" in prompt
    assert "never fail it for not blinking" in prompt
    assert "red X" not in prompt.replace("Red X", "")


@pytest.mark.unit
@pytest.mark.parametrize("profile", POWERHEART, ids=lambda p: p.id)
def test_powerheart_readiness_is_green_or_red_and_the_lid_self_test_is_normal(profile):
    prompt = _prompt("readiness_indicator", profile, frames=20)
    assert "Rescue Ready" in prompt
    assert "GREEN" in prompt and "RED" in prompt
    assert "red that turns green within the clip is normal" in prompt
    assert "never fail it for not blinking" in prompt


@pytest.mark.unit
@pytest.mark.parametrize("profile", (ZOLL_AED_3, *POWERHEART), ids=lambda p: p.id)
def test_a_steady_indicator_seen_in_frames_passes_without_any_flash(profile):
    assert not profile.blinking_ready
    result = svc.ChecklistAnalysisResult(
        passed=True, confidence=0.9, notes="Green.", status="ready", ready_frames=[3, 8]
    )
    checked = svc._apply_deterministic_checks(READINESS, result, profile=profile, video=_video())
    assert checked.passed is True
    assert checked.status == "ready"


@pytest.mark.unit
@pytest.mark.parametrize(
    "profile, says",
    [(ZOLL_AED_3, "right of the On/Off button"), (POWERHEART_G3, "Rescue Ready"), (POWERHEART_G5, "Rescue Ready")],
    ids=lambda v: getattr(v, "id", v),
)
def test_an_unbacked_ready_asks_to_film_this_units_own_indicator(profile, says):
    result = svc.ChecklistAnalysisResult(passed=True, confidence=0.9, notes="Ready.", status="ready", ready_frames=[])
    checked = svc._apply_deterministic_checks(READINESS, result, profile=profile, video=_video(), language="hi")
    assert checked.passed is False
    assert checked.status == "unclear"
    assert says in checked.notes
    assert "Ready light" not in checked.notes  # the Philips wording
    assert checked.notes_hi


# ── Brands ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("profile", POWERHEART, ids=lambda p: p.id)
@pytest.mark.parametrize("brand", ["Cardiac Science", "ZOLL", "Powerheart"])
def test_a_powerheart_branded_either_way_is_not_a_different_aed(profile, brand):
    result = svc.ChecklistAnalysisResult(passed=True, confidence=0.9, notes="ok", brand_seen=brand)
    assert svc._different_brand(get_item("pads_connected"), result, profile) is None


@pytest.mark.unit
@pytest.mark.parametrize("profile", (ZOLL_AED_3, *POWERHEART), ids=lambda p: p.id)
def test_a_philips_photo_still_fails_these_units(profile):
    result = svc.ChecklistAnalysisResult(passed=True, confidence=0.9, notes="ok", brand_seen="Philips HeartStart")
    assert svc._different_brand(get_item("pads_connected"), result, profile) == "Philips"


# ── Child pads ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_aed3_uni_padz_count_as_child_capable():
    guidance = ZOLL_AED_3.guidance["child_key_pad"]
    assert "CPR Uni-padz" in guidance and "Child button" in guidance


@pytest.mark.unit
def test_g5_paediatric_pads_plugged_in_are_a_fail():
    guidance = POWERHEART_G5.guidance["child_key_pad"]
    assert "NOT to be pre-connected" in guidance
    assert "passed=false" in guidance


# ── AED 3 battery: install-by in GS1 '(15)' ─────────────────────────────────


@pytest.mark.unit
def test_gs1_best_before_is_parsed():
    assert parse_gs1_dates("(01)00847946026745(15)280419(10)A123") == {"15": "2028-04-19"}


@pytest.mark.unit
def test_aed3_battery_install_by_is_corrected_from_its_gs1_field():
    result = svc.ChecklistAnalysisResult(
        passed=True,
        confidence=0.9,
        notes="ok",
        expiry_date="2028-01-19",  # a 4 misread as a 1
        expiry_raw_text="(15)280419 / (11)230419",
        date_legible=True,
        damage_seen=False,
    )
    checked = svc._apply_deterministic_checks(BATTERY, result, today=TODAY, profile=ZOLL_AED_3)
    assert checked.expiry_date == "2028-04-19"
    assert checked.passed is True


@pytest.mark.unit
def test_aed3_battery_manufacture_date_is_never_its_install_by():
    result = svc.ChecklistAnalysisResult(
        passed=True, confidence=0.9, notes="ok", expiry_date="2023-04-19", expiry_raw_text="(15)280419 (11)230419"
    )
    checked = svc._apply_deterministic_checks(BATTERY, result, today=TODAY, profile=ZOLL_AED_3)
    assert checked.passed is False
    assert checked.expiry_date is None


@pytest.mark.unit
def test_aed3_battery_is_not_dated_by_age():
    # It prints an install-by date; only the Powerheart batteries don't.
    assert ZOLL_AED_3.battery_life_months is None
    assert "INSTALL-BY" in ZOLL_AED_3.guidance["battery_expiry"]


# ── Powerheart batteries: dated by age ──────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("profile", POWERHEART, ids=lambda p: p.id)
def test_the_model_is_told_there_is_no_expiry_to_find(profile):
    guidance = profile.guidance["battery_expiry"]
    assert "NO expiry date" in guidance
    assert "manufacture_date" in guidance and "install_date" in guidance


@pytest.mark.unit
@pytest.mark.parametrize("profile", POWERHEART, ids=lambda p: p.id)
def test_a_recent_battery_passes_dated_four_years_from_manufacture(profile):
    checked = _battery(profile, manufacture_date="2025-03")
    assert checked.passed is True
    assert checked.expiry_date == "2029-03"
    assert "no expiry date printed" in checked.notes
    assert "31 Mar 2029" in checked.notes


@pytest.mark.unit
def test_a_battery_made_over_four_years_ago_is_due_and_says_why():
    checked = _battery(POWERHEART_G5, manufacture_date="2021-06-15", passed=True, language="hi")
    assert checked.passed is False
    assert checked.expiry_date == "2025-06-15"
    assert "was due for replacement by 15 Jun 2025" in checked.notes
    assert "installed later" in checked.notes
    assert checked.notes_hi and "बैटरी बदलें" in checked.notes_hi
    assert "expired_model_said_pass" in _overrides(POWERHEART_G5, manufacture_date="2021-06-15", passed=True)


@pytest.mark.unit
def test_a_written_install_date_is_dated_from():
    checked = _battery(POWERHEART_G3, manufacture_date="2021-01", install_date="2023-05-10")
    assert checked.passed is True
    assert checked.expiry_date == "2027-05-10"
    assert "installed" in checked.notes


@pytest.mark.unit
def test_an_install_date_before_manufacture_is_a_misread_and_ignored():
    checked = _battery(POWERHEART_G3, manufacture_date="2024-02", install_date="2019-02")
    assert checked.expiry_date == "2028-02"
    assert checked.install_date is None


@pytest.mark.unit
def test_a_manufacture_date_in_the_future_is_a_misread():
    checked = _battery(POWERHEART_G5, manufacture_date="2027-01", passed=True)
    assert checked.passed is False
    assert checked.expiry_date is None
    assert "date it was made" in checked.notes


@pytest.mark.unit
def test_a_lone_date_reported_as_expiry_is_taken_as_the_manufacture_date():
    # There is no expiry on this battery, so the date the model found is the
    # one printed: when it was made.
    checked = _battery(POWERHEART_G5, expiry_date="2024-08")
    assert checked.expiry_date == "2028-08"
    assert checked.manufacture_date == "2024-08"


@pytest.mark.unit
def test_no_readable_date_is_a_retake_that_says_where_to_look():
    checked = _battery(POWERHEART_G3, date_legible=False, language="hi")
    assert checked.passed is False
    assert "factory symbol" in checked.notes
    assert checked.notes_hi


@pytest.mark.unit
def test_a_damaged_or_uncertain_battery_is_not_passed():
    assert _battery(POWERHEART_G3, manufacture_date="2025-03", damage_seen=True).passed is False
    assert _battery(POWERHEART_G3, manufacture_date="2025-03", date_legible=False).passed is False
    assert _battery(POWERHEART_G3, manufacture_date="2025-03", confidence=0.3).passed is False


@pytest.mark.unit
def test_two_readings_of_a_powerheart_battery_compare_the_date_it_was_made():
    first = svc.ChecklistAnalysisResult(passed=False, confidence=0.9, notes="", manufacture_date="2025-03")
    same = svc.ChecklistAnalysisResult(passed=False, confidence=0.9, notes="", manufacture_date="2025-03-11")
    other = svc.ChecklistAnalysisResult(passed=False, confidence=0.9, notes="", manufacture_date="2025-08")
    assert svc._compare_readings(BATTERY, first, same, POWERHEART_G5) == "agrees"
    assert svc._compare_readings(BATTERY, first, other, POWERHEART_G5) == "disagrees"


def _overrides(profile, **fields):
    overrides = []
    defaults = dict(passed=False, confidence=0.9, notes="", date_legible=True)
    svc._apply_deterministic_checks(
        BATTERY,
        svc.ChecklistAnalysisResult(**{**defaults, **fields}),
        today=TODAY,
        profile=profile,
        overrides=overrides,
    )
    return overrides
