"""Unit tests for the checklist-mode Gemini analysis service."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services import gemini_checklist_service as svc
from app.services.checklist_items import CHECKLIST_ITEMS, get_item


@pytest.mark.unit
def test_all_items_have_unique_ids_and_valid_sections():
    ids = [item.id for item in CHECKLIST_ITEMS]
    assert len(ids) == len(set(ids)) == 10
    assert {item.section for item in CHECKLIST_ITEMS} == {1, 2, 3}
    required_ids = {item.id for item in CHECKLIST_ITEMS if item.required}
    assert required_ids == {
        "serial_number",
        "pads_expiry",
        "battery_expiry",
        "battery_attached",
        "pads_connected",
        "readiness_indicator",
    }


@pytest.mark.unit
def test_get_item_unknown_returns_none():
    assert get_item("not_a_real_item") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analyze_checklist_item_unknown_id_raises():
    with pytest.raises(ValueError):
        await svc.analyze_checklist_item("not_a_real_item", b"fake-bytes")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analyze_checklist_item_happy_path():
    fake_result = svc.ChecklistAnalysisResult(
        passed=True,
        confidence=0.92,
        notes="Serial number clearly legible.",
        serial_number="US00123456",
    )
    mock_response = AsyncMock()
    mock_response.parsed = fake_result

    with patch.object(svc, "_get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await svc.analyze_checklist_item("serial_number", b"fake-jpeg-bytes")

    assert result.passed is True
    assert result.serial_number == "US00123456"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_implausible_serial_is_vetoed_even_if_gemini_says_pass():
    fake_result = svc.ChecklistAnalysisResult(
        passed=True,
        confidence=0.95,
        notes="Looks fine.",
        serial_number="!!",  # too short / not alphanumeric-plausible
    )
    mock_response = AsyncMock()
    mock_response.parsed = fake_result

    with patch.object(svc, "_get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await svc.analyze_checklist_item("serial_number", b"fake-jpeg-bytes")

    assert result.passed is False
    assert result.serial_number is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_implausible_expiry_is_vetoed():
    fake_result = svc.ChecklistAnalysisResult(
        passed=True,
        confidence=0.9,
        notes="Looks fine.",
        expiry_date="2099-01",  # far outside plausible window
        expiry_raw_text="2099-01",
    )
    mock_response = AsyncMock()
    mock_response.parsed = fake_result

    with patch.object(svc, "_get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await svc.analyze_checklist_item("pads_expiry", b"fake-jpeg-bytes")

    assert result.passed is False
    assert result.expiry_date is None
