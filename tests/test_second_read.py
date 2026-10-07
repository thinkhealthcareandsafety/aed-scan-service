"""Labels are read twice, by two models, and the readings must agree.

Measured on 1 Oct 2026: on a dim photo of the HS1 serial label, the photo
model read A18A-06336 as A1BA-06336 — once in five tries, confidently. A
wrong serial or date on an inspection record is worse than a retake, so a
second, different model reads the same label alongside the first.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from google.genai import errors

from app.services import gemini_checklist_service as svc

PRIMARY = svc.GEMINI_IMAGE_MODEL
SECOND = svc.GEMINI_IMAGE_FALLBACKS[0]


def _answer(**fields):
    response = AsyncMock()
    response.parsed = svc.ChecklistVerdict(confidence=0.9, notes="Read it.", **fields)
    return response


def _client(by_model):
    calls = []

    async def generate(*, model, **_):
        calls.append(model)
        outcome = by_model[model]
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, AsyncMock):  # an answer, ready
            return outcome
        return await outcome()  # a deliberately slow answer

    client = AsyncMock()
    client.aio.models.generate_content = AsyncMock(side_effect=generate)
    return client, calls


async def _analyse(item, by_model):
    client, calls = _client(by_model)
    with patch.object(svc, "_get_client", return_value=client):
        result = await svc.analyze_checklist_item(item, b"fake-jpeg", "image/jpeg", aed_model="Philips HS1")
    return result, calls


@pytest.mark.unit
@pytest.mark.asyncio
async def test_two_readings_that_agree_stand():
    result, calls = await _analyse("serial_number", {
        PRIMARY: _answer(passed=True, serial_number="A18A-06336"),
        SECOND: _answer(passed=True, serial_number="SN: A18A06336"),  # same serial, written differently
    })
    assert result.passed is True
    assert result.serial_number == "A18A-06336"
    assert result.meta.second_read == "agrees"
    assert sorted(calls) == sorted([PRIMARY, SECOND])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_two_readings_that_disagree_ask_for_a_retake():
    result, _ = await _analyse("serial_number", {
        PRIMARY: _answer(passed=True, serial_number="A1BA-06336"),
        SECOND: _answer(passed=True, serial_number="A18A-06336"),
    })
    assert result.passed is False
    assert result.serial_number is None  # never guess between them
    assert "couldn't read the serial number with certainty" in result.notes
    assert result.meta.second_read == "disagrees"
    assert result.meta.overrides == ["second_read_disagrees"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expiry_months_that_disagree_ask_for_a_retake():
    result, _ = await _analyse("pads_expiry", {
        PRIMARY: _answer(passed=True, expiry_date="2028-06", date_legible=True, damage_seen=False),
        SECOND: _answer(passed=True, expiry_date="2025-06", date_legible=True, damage_seen=False),
    })
    assert result.passed is False
    assert result.expiry_date is None
    assert "expiry date" in result.notes


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_agreed_date_still_goes_through_the_date_rules():
    result, _ = await _analyse("battery_expiry", {
        PRIMARY: _answer(passed=False, expiry_date="2028-12-31", date_legible=True, damage_seen=False),
        SECOND: _answer(passed=False, expiry_date="2028-12", date_legible=True, damage_seen=False),
    })
    assert result.meta.second_read == "agrees"
    assert result.passed is True  # in date: decided by the service, not the models
    assert result.notes.startswith("Battery in date until 31 Dec 2028")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_second_model_that_cannot_read_does_not_overrule_one_that_can():
    result, _ = await _analyse("serial_number", {
        PRIMARY: _answer(passed=True, serial_number="A18A-06336"),
        SECOND: _answer(passed=False, serial_number=None),
    })
    assert result.passed is True
    assert result.meta.second_read == "unread"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_second_model_that_is_down_never_blocks_the_check():
    overloaded = errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})
    result, _ = await _analyse("serial_number", {
        PRIMARY: _answer(passed=True, serial_number="A18A-06336"),
        SECOND: overloaded,
    })
    assert result.passed is True
    assert result.meta.second_read == "unavailable"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_slow_second_model_is_only_waited_for_briefly(monkeypatch):
    monkeypatch.setattr(svc, "SECOND_READ_GRACE_SECONDS", 0.1)

    async def slow():
        await asyncio.sleep(5)
        return _answer(passed=True, serial_number="A18A-06336")

    started = asyncio.get_running_loop().time()
    result, _ = await _analyse("serial_number", {
        PRIMARY: _answer(passed=True, serial_number="A18A-06336"),
        SECOND: slow,
    })
    assert result.passed is True
    assert result.meta.second_read == "unavailable"
    assert asyncio.get_running_loop().time() - started < 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_checks_without_a_label_are_read_once():
    result, calls = await _analyse("battery_attached", {
        PRIMARY: _answer(passed=True),
        SECOND: _answer(passed=True),
    })
    assert calls == [PRIMARY]
    assert result.meta.second_read is None
