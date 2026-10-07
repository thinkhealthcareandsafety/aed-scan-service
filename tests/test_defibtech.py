"""The four Defibtech units are judged as themselves.

What each would get wrong if described like the units already known:

- All four prove readiness by FLASHING a small green Active Status
  Indicator, like a Philips — so a "ready" needs a flash found in the clip.
  On the Lifeline / AUTO it is at the top-right corner by the handle; on the
  VIEW / ECG just right of the On/Off button. The On/Off button is not it.
- The Lifeline / AUTO battery slides into the SIDE; the VIEW / ECG battery
  into the BACK, label facing in — so those two read their battery and pads
  dates off the AED Status Screen, which also shows the battery's serial
  beside the AED's.
- A Lifeline battery pack holds a 9V battery for the status light, whose
  own date is not the pack's expiry.
- Defibtech serials are 9 digits (DDU-120: first digit 3-9; DDU-2xxx:
  first digit 4), so a misread or the battery's serial is caught.
"""
from __future__ import annotations

from datetime import date

import pytest

from app.services import gemini_checklist_service as svc
from app.services import readiness_frames
from app.services.checklist_items import CHECKLIST_ITEMS, get_item
from app.services.device_profiles import (
    DEFIBTECH_LIFELINE,
    DEFIBTECH_LIFELINE_AUTO,
    DEFIBTECH_LIFELINE_ECG,
    DEFIBTECH_LIFELINE_VIEW,
    get_profile,
)

TODAY = date(2026, 10, 7)
LIFELINES = (DEFIBTECH_LIFELINE, DEFIBTECH_LIFELINE_AUTO)
SCREENED = (DEFIBTECH_LIFELINE_VIEW, DEFIBTECH_LIFELINE_ECG)
DEFIBTECH = (*LIFELINES, *SCREENED)
SERIAL = get_item("serial_number")
READINESS = get_item("readiness_indicator")
ids = lambda p: getattr(p, "id", p)  # noqa: E731


def _prompt(item_id: str, profile) -> str:
    return svc._build_prompt(get_item(item_id), profile)


def _video(flashes: int, frames: int = 24) -> readiness_frames.ReadinessFrames:
    return readiness_frames.ReadinessFrames(
        frames=[b"x"] * frames, times=[i * 0.4 for i in range(frames)], flash_count=flashes
    )


def _serial(profile, value: str, language=None):
    result = svc.ChecklistAnalysisResult(passed=True, confidence=0.9, notes="Read it.", serial_number=value)
    return svc._apply_deterministic_checks(SERIAL, result, profile=profile, language=language)


# ── Ids and prompts ─────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "model, profile",
    [
        ("Defibtech Lifeline", DEFIBTECH_LIFELINE),
        ("Defibtech Lifeline AUTO", DEFIBTECH_LIFELINE_AUTO),
        ("Defibtech Lifeline VIEW", DEFIBTECH_LIFELINE_VIEW),
        ("Defibtech Lifeline ECG", DEFIBTECH_LIFELINE_ECG),
    ],
)
def test_each_model_id_gets_its_own_profile(model, profile):
    assert get_profile(model) is profile


@pytest.mark.unit
@pytest.mark.parametrize("profile", DEFIBTECH, ids=ids)
@pytest.mark.parametrize("item", [item.id for item in CHECKLIST_ITEMS])
def test_no_prompt_describes_another_makers_unit(profile, item):
    prompt = _prompt(item, profile)
    for other in ("Philips", "SMART Pads", "M5070A", "Rescue Ready", "CPR Uni-padz", "Intellisense"):
        assert other not in prompt


@pytest.mark.unit
@pytest.mark.parametrize("profile", DEFIBTECH, ids=ids)
def test_every_prompt_names_this_unit(profile):
    assert profile.name in _prompt("pads_connected", profile)
    assert "Defibtech" in profile.name


@pytest.mark.unit
def test_the_auto_is_told_it_has_no_shock_button():
    assert "NO shock button" in DEFIBTECH_LIFELINE_AUTO.appearance
    assert "'auto'" in DEFIBTECH_LIFELINE_AUTO.guidance["readiness_indicator"]
    assert "red SHOCK button" in DEFIBTECH_LIFELINE.guidance["readiness_indicator"]


# ── Readiness: a flashing light, in this unit's own place ───────────────────


@pytest.mark.unit
@pytest.mark.parametrize("profile", DEFIBTECH, ids=ids)
def test_readiness_must_be_backed_by_a_flash_in_the_clip(profile):
    assert profile.blinking_ready
    result = svc.ChecklistAnalysisResult(passed=True, confidence=0.9, notes="Green.", status="ready", ready_frames=[4])
    checked = svc._apply_deterministic_checks(READINESS, result, profile=profile, video=_video(flashes=0))
    assert checked.passed is False and checked.status == "unclear"

    flashed = svc._apply_deterministic_checks(READINESS, result, profile=profile, video=_video(flashes=2))
    assert flashed.passed is True and flashed.status == "ready"


@pytest.mark.unit
@pytest.mark.parametrize(
    "profile, place",
    [
        (DEFIBTECH_LIFELINE, "top-right corner"),
        (DEFIBTECH_LIFELINE_AUTO, "top-right corner"),
        (DEFIBTECH_LIFELINE_VIEW, "right of the On/Off button"),
        (DEFIBTECH_LIFELINE_ECG, "right of the On/Off button"),
    ],
    ids=ids,
)
def test_a_retake_points_at_this_units_status_light(profile, place):
    guidance = profile.guidance["readiness_indicator"].lower()
    assert ("top-right corner" if profile in LIFELINES else "right of the green on/off button") in guidance
    result = svc.ChecklistAnalysisResult(passed=True, confidence=0.9, notes="Ready.", status="ready", ready_frames=[])
    checked = svc._apply_deterministic_checks(READINESS, result, profile=profile, video=_video(1), language="hi")
    assert place in checked.notes
    assert "Ready light" not in checked.notes  # the Philips wording
    assert checked.notes_hi


@pytest.mark.unit
@pytest.mark.parametrize("profile", DEFIBTECH, ids=ids)
def test_the_on_off_button_is_never_the_indicator(profile):
    guidance = profile.guidance["readiness_indicator"]
    assert "Never judge the green On/Off button" in guidance or "Never judge the On/Off button" in guidance
    assert "RED" in guidance and "status='fault'" in guidance


# ── Serial: 9 digits, in Defibtech's own form ───────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "profile, good",
    [
        (DEFIBTECH_LIFELINE, "123456789"),
        (DEFIBTECH_LIFELINE_AUTO, "312345678"),
        (DEFIBTECH_LIFELINE_VIEW, "412345678"),
        (DEFIBTECH_LIFELINE_ECG, "4 1234 5678"),
    ],
    ids=ids,
)
def test_a_serial_in_defibtechs_form_passes(profile, good):
    checked = _serial(profile, good)
    assert checked.passed is True
    assert checked.serial_number == good.replace(" ", "")


@pytest.mark.unit
@pytest.mark.parametrize(
    "profile, bad",
    [
        (DEFIBTECH_LIFELINE, "12345678"),  # a digit short
        (DEFIBTECH_LIFELINE, "DDU-100"),  # the model number
        (DEFIBTECH_LIFELINE_AUTO, "212345678"),  # DDU-120 serials start 3-9
        (DEFIBTECH_LIFELINE_VIEW, "312345678"),  # DDU-2xxx serials start with 4
        (DEFIBTECH_LIFELINE_ECG, "4123456789"),  # a digit too many
    ],
    ids=ids,
)
def test_anything_else_is_a_retake_that_says_where_the_serial_is(profile, bad):
    checked = _serial(profile, bad, language="hi")
    assert checked.passed is False
    assert checked.serial_number is None
    assert "9 digits" in checked.notes
    assert checked.notes_hi
    if profile in SCREENED:
        assert "AED S/N" in checked.notes
    else:
        assert "behind" in checked.notes


@pytest.mark.unit
@pytest.mark.parametrize("profile", SCREENED, ids=ids)
def test_the_status_screen_battery_serial_is_never_the_aeds(profile):
    guidance = profile.guidance["serial_number"]
    assert "AED S/N" in guidance and "Battery S/N" in guidance
    assert "never be reported as the AED's" in guidance


# ── Dates: the right date, from where it can be read ────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("profile", SCREENED, ids=ids)
def test_the_status_screen_gives_battery_and_pads_dates(profile):
    assert "'Battery status' row" in profile.guidance["battery_expiry"]
    assert "never the pads'" in profile.guidance["battery_expiry"]
    assert "'Pads status' row" in profile.guidance["pads_expiry"]
    assert "never the battery's" in profile.guidance["pads_expiry"]


@pytest.mark.unit
@pytest.mark.parametrize("profile", SCREENED, ids=ids)
def test_an_in_date_battery_read_off_the_status_screen_passes(profile):
    result = svc.ChecklistAnalysisResult(
        passed=True,
        confidence=0.9,
        notes="Battery expires January 2029.",
        expiry_date="2029-01",
        expiry_raw_text="Battery status Expires 01/2029",
        date_legible=True,
        damage_seen=False,
    )
    checked = svc._apply_deterministic_checks(
        get_item("battery_expiry"), result, profile=profile, today=TODAY
    )
    assert checked.passed is True and checked.expiry_date == "2029-01"


@pytest.mark.unit
@pytest.mark.parametrize("profile", SCREENED, ids=ids)
def test_expired_pads_read_off_the_status_screen_fail(profile):
    result = svc.ChecklistAnalysisResult(
        passed=True,
        confidence=0.9,
        notes="Pads fine.",
        expiry_date="2026-06",
        expiry_raw_text="Pads status Adult Expires 06/2026",
        date_legible=True,
        damage_seen=False,
    )
    checked = svc._apply_deterministic_checks(get_item("pads_expiry"), result, profile=profile, today=TODAY)
    assert checked.passed is False


@pytest.mark.unit
@pytest.mark.parametrize("profile", LIFELINES, ids=ids)
def test_the_9v_status_battery_date_is_not_the_packs_expiry(profile):
    guidance = profile.guidance["battery_expiry"]
    assert "9V" in guidance and "never report it" in guidance
    assert "SIDE" in guidance


@pytest.mark.unit
@pytest.mark.parametrize("profile", DEFIBTECH, ids=ids)
def test_no_defibtech_battery_is_dated_by_age(profile):
    # Their packs print an expiration date; only Powerheart batteries don't.
    assert profile.battery_life_months is None


# ── Brand ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("profile", DEFIBTECH, ids=ids)
@pytest.mark.parametrize("seen, other", [("Defibtech", None), ("Lifeline VIEW", None), ("ZOLL", "ZOLL")])
def test_a_photo_of_another_makers_aed_is_caught(profile, seen, other):
    result = svc.ChecklistAnalysisResult(passed=True, confidence=0.9, notes="ok", brand_seen=seen)
    assert svc._different_brand(get_item("pads_connected"), result, profile) == other
