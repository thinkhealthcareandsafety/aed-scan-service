"""The prompt must describe the AED actually being inspected.

Every unit used to be described as a Philips. These pin down what that got
wrong: a ZOLL judged against a Philips "blinking light", the Philips battery
described as black, a GS1 field code kept in a serial number, a production
date mistaken for an expiry, and Hindi feedback that contradicted a verdict
the server had overruled.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from app.services import gemini_checklist_service as svc
from app.services.checklist_items import CHECKLIST_ITEMS, get_item
from app.services.device_profiles import (
    DEFIBTECH_LIFELINE,
    DEFIBTECH_LIFELINE_AUTO,
    DEFIBTECH_LIFELINE_ECG,
    DEFIBTECH_LIFELINE_VIEW,
    GENERIC,
    PHILIPS_FRX,
    PHILIPS_HS1,
    POWERHEART_G3,
    POWERHEART_G5,
    PROFILES,
    ZOLL_AED_3,
    ZOLL_AED_PLUS,
    get_profile,
)
from app.utils.date_parser import parse_all_expiry_dates, parse_gs1_dates
from app.utils.validators import normalise_serial

SUPPORTED = (
    PHILIPS_FRX,
    PHILIPS_HS1,
    ZOLL_AED_PLUS,
    ZOLL_AED_3,
    POWERHEART_G3,
    POWERHEART_G5,
    DEFIBTECH_LIFELINE,
    DEFIBTECH_LIFELINE_AUTO,
    DEFIBTECH_LIFELINE_VIEW,
    DEFIBTECH_LIFELINE_ECG,
)
DEVICE_SPECIFIC_ITEMS = (
    "serial_number",
    "pads_expiry",
    "battery_expiry",
    "battery_attached",
    "pads_connected",
    "readiness_indicator",
    "child_key_pad",
)
TODAY = date(2026, 9, 29)


def _prompt(item_id: str, profile, language=None, frames=None) -> str:
    return svc._build_prompt(get_item(item_id), profile, frame_count=frames, language=language)


# ── Profiles ────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("profile", SUPPORTED, ids=lambda p: p.id)
def test_every_supported_unit_has_guidance_for_every_device_specific_check(profile):
    missing = [item for item in DEVICE_SPECIFIC_ITEMS if not profile.guidance.get(item)]
    assert missing == []


@pytest.mark.unit
def test_profiles_are_keyed_by_the_model_ids_the_app_stores():
    assert set(PROFILES) == {
        "Philips FRx",
        "Philips HS1",
        "Zoll AED Plus",
        "Zoll AED 3",
        "Zoll Powerheart G3",
        "Zoll Powerheart G5",
        "Defibtech Lifeline",
        "Defibtech Lifeline AUTO",
        "Defibtech Lifeline VIEW",
        "Defibtech Lifeline ECG",
    }
    assert get_profile("Zoll AED Plus") is ZOLL_AED_PLUS
    assert get_profile(" Philips HS1 ") is PHILIPS_HS1


@pytest.mark.unit
@pytest.mark.parametrize("model", [None, "", "Mindray BeneHeart C1A", "philips frx"])
def test_a_missing_or_unknown_model_gets_the_brand_neutral_profile(model):
    assert get_profile(model) is GENERIC


# ── Prompts ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_zoll_readiness_is_judged_on_its_status_window_not_a_blinking_light():
    prompt = _prompt("readiness_indicator", ZOLL_AED_PLUS, frames=20)
    assert "GREEN CHECK" in prompt
    assert "NO blinking ready light" in prompt
    assert "never fail it for not blinking" in prompt
    assert "Philips" not in prompt


@pytest.mark.unit
@pytest.mark.parametrize("item", [item.id for item in CHECKLIST_ITEMS])
def test_no_zoll_prompt_describes_a_philips(item):
    assert "Philips" not in _prompt(item, ZOLL_AED_PLUS)


@pytest.mark.unit
@pytest.mark.parametrize("profile", (PHILIPS_FRX, PHILIPS_HS1), ids=lambda p: p.id)
@pytest.mark.parametrize("item", [item.id for item in CHECKLIST_ITEMS])
def test_no_philips_prompt_describes_a_zoll(profile, item):
    assert "ZOLL" not in _prompt(item, profile).upper().replace("ZOLL AED PLUS", "")


@pytest.mark.unit
@pytest.mark.parametrize("profile", (PHILIPS_FRX, PHILIPS_HS1), ids=lambda p: p.id)
def test_philips_readiness_warns_off_the_green_on_off_button(profile):
    prompt = _prompt("readiness_indicator", profile, frames=20)
    assert "BLINKS its Ready light green" in prompt
    assert "Never judge the On/Off button's colour" in prompt


@pytest.mark.unit
def test_the_philips_battery_is_described_as_blue():
    for profile in (PHILIPS_FRX, PHILIPS_HS1):
        text = profile.guidance["battery_expiry"] + profile.guidance["battery_attached"]
        assert "blue" in text
        assert "black" not in text.lower()


@pytest.mark.unit
def test_hs1_pads_are_judged_as_a_cartridge_in_the_front_well():
    assert "cartridge well on the front" in PHILIPS_HS1.guidance["pads_connected"]
    assert "all the way down" in PHILIPS_HS1.guidance["pads_connected"]


@pytest.mark.unit
def test_an_frx_child_key_left_in_the_slot_is_a_fail():
    assert "NOT to store the FRx with the key installed" in PHILIPS_FRX.guidance["child_key_pad"]


@pytest.mark.unit
def test_zoll_cell_dates_are_manufacture_dates_not_expiry():
    guidance = ZOLL_AED_PLUS.guidance["battery_expiry"]
    assert "REPLACE BATTERIES ON OR BEFORE" in guidance
    assert "MANUFACTURE date" in guidance


@pytest.mark.unit
def test_the_prompt_names_the_device_and_its_item_notes():
    prompt = _prompt("serial_number", ZOLL_AED_PLUS)
    assert "DEVICE: ZOLL AED Plus." in prompt
    assert "Device notes for this item (ZOLL AED Plus)" in prompt
    assert "'(21)' is the barcode's field code" in prompt


@pytest.mark.unit
def test_hindi_feedback_is_requested_only_for_hindi():
    assert "notes_hi with the same message as notes" in _prompt("pads_expiry", PHILIPS_FRX, language="hi")
    assert "Leave notes_hi null." in _prompt("pads_expiry", PHILIPS_FRX)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_model_passed_by_the_backend_reaches_the_prompt():
    mock_response = AsyncMock()
    mock_response.parsed = svc.ChecklistAnalysisResult(passed=True, confidence=0.9, notes="Status window shows a green check.")

    with patch.object(svc, "_get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        await svc.analyze_checklist_item(
            "battery_attached", b"fake-jpeg-bytes", "image/jpeg", aed_model="Zoll AED Plus", language="hi"
        )

    contents = mock_client.aio.models.generate_content.call_args.kwargs["contents"]
    prompt = contents[-1]
    assert "DEVICE: ZOLL AED Plus." in prompt
    assert "ten 123 cells" in prompt
    assert "notes_hi with the same message" in prompt


# ── Serial numbers ──────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("(21) X14K718292", "X14K718292"),
        ("(21)X14K718292", "X14K718292"),
        ("SN: B17C-00516", "B17C-00516"),
        ("S/N A18A-06336", "A18A-06336"),
        ("Serial No: 12345678", "12345678"),
        ("(21) SN: X14K718292", "X14K718292"),
        # Real serials that merely start with the letters SN keep them.
        ("SN04F12345", "SN04F12345"),
        ("B17C-00516", "B17C-00516"),
    ],
)
def test_serial_numbers_are_stored_without_field_codes_or_captions(raw, expected):
    assert normalise_serial(raw) == expected


@pytest.mark.unit
def test_a_gs1_serial_read_is_cleaned_before_it_is_stored():
    result = svc.ChecklistAnalysisResult(
        passed=True, confidence=0.9, notes="Serial read.", serial_number="(21) X14K718292"
    )
    checked = svc._apply_deterministic_checks(get_item("serial_number"), result, today=TODAY)
    assert checked.serial_number == "X14K718292"
    assert checked.passed is True


# ── GS1 dates ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_gs1_fields_are_parsed_by_what_they_are():
    parsed = parse_gs1_dates("(01)00847946000189(17)271228(10)3917 (11)250301")
    assert parsed == {"17": "2027-12-28", "11": "2025-03-01"}


@pytest.mark.unit
def test_a_gs1_day_of_00_means_the_whole_month():
    assert parse_gs1_dates("(17)280900") == {"17": "2028-09"}


@pytest.mark.unit
def test_gs1_dates_count_as_dates_on_the_label():
    assert "2027-12-28" in parse_all_expiry_dates("(17)271228 lot 3917")


@pytest.mark.unit
def test_a_gs1_expiry_overrules_a_misread_printed_date():
    result = svc.ChecklistAnalysisResult(
        passed=True,
        confidence=0.8,
        notes="Pads in date.",
        expiry_date="2027-11-28",
        expiry_raw_text="2027-11-28 / (17)271228",
    )
    checked = svc._apply_deterministic_checks(get_item("pads_expiry"), result, today=TODAY)
    assert checked.expiry_date == "2027-12-28"
    assert checked.passed is True


@pytest.mark.unit
def test_the_gs1_production_date_is_never_accepted_as_the_expiry():
    result = svc.ChecklistAnalysisResult(
        passed=True,
        confidence=0.8,
        notes="Battery in date.",
        expiry_date="2025-03-01",
        expiry_raw_text="(11)250301",
    )
    checked = svc._apply_deterministic_checks(get_item("battery_expiry"), result, today=TODAY)
    assert checked.passed is False
    assert checked.expiry_date is None


# ── Hindi feedback ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_an_overruled_verdict_gets_hindi_notes_that_match_it():
    # The model said "in date"; the server's clock says expired. A Hindi
    # reader must see the server's verdict, not the model's.
    result = svc.ChecklistAnalysisResult(
        passed=True,
        confidence=0.9,
        notes="Pads are in date.",
        notes_hi="पैड्स सही तारीख में हैं।",
        expiry_date="2026-03",
    )
    checked = svc._apply_deterministic_checks(get_item("pads_expiry"), result, today=TODAY, language="hi")
    assert checked.passed is False
    assert "expired on 31 Mar 2026" in checked.notes
    assert checked.notes_hi == (
        "पैड्स 31 मार्च 2026 को एक्सपायर हो चुके हैं। किसी इमरजेंसी में इस AED पर भरोसा करने से पहले पैड्स बदलें।"
    )


@pytest.mark.unit
def test_an_expired_battery_reads_in_the_feminine_in_hindi():
    result = svc.ChecklistAnalysisResult(passed=True, confidence=0.9, notes="ok", expiry_date="2025-12-31")
    checked = svc._apply_deterministic_checks(get_item("battery_expiry"), result, today=TODAY, language="hi")
    assert checked.notes_hi.startswith("बैटरी 31 दिसंबर 2025 को एक्सपायर हो चुकी है।")


@pytest.mark.unit
def test_hindi_notes_are_dropped_when_english_was_asked_for():
    result = svc.ChecklistAnalysisResult(passed=True, confidence=0.9, notes="Kit present.", notes_hi="किट मौजूद है।")
    checked = svc._apply_deterministic_checks(get_item("first_response_kit"), result, today=TODAY)
    assert checked.notes_hi is None
