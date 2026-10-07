"""An overloaded model must not take the check down with it.

Measured against production on 29 Sep 2026: gemini-3.6-flash answered every
video with "503 high demand", each refusal taking 12-16 s to arrive. Walking
the model chain in order spent the whole 35 s budget on those refusals, so
the readiness check — required on every inspection — timed out even though
the fallback model was healthy.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from google.genai import errors

from app.services import gemini_checklist_service as svc


def _overloaded() -> errors.ServerError:
    return errors.ServerError(
        503, {"error": {"code": 503, "message": "This model is currently experiencing high demand.", "status": "UNAVAILABLE"}}
    )


def _answer(notes: str):
    response = AsyncMock()
    # A ready verdict names where it saw the signal (here, second 3 of the
    # raw video) — without that, the readiness check won't accept it.
    response.parsed = svc.ChecklistAnalysisResult(
        passed=True, confidence=0.9, notes=notes, status="ready", ready_frames=[3]
    )
    return response


@pytest.fixture(autouse=True)
def fast_clock(monkeypatch):
    monkeypatch.setattr(svc, "HEDGE_DELAY_SECONDS", 0.05)
    monkeypatch.setattr(svc, "IMAGE_HEDGE_DELAY_SECONDS", 0.05)
    monkeypatch.setattr(svc, "RETRY_BACKOFF_SECONDS", (0.01, 0.02))


def _client(side_effect):
    client = AsyncMock()
    client.aio.models.generate_content = AsyncMock(side_effect=side_effect)
    return client


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_overloaded_first_model_is_answered_by_the_fallback():
    async def generate(*, model, **_):
        if model == svc.GEMINI_VIDEO_MODELS[0]:
            await asyncio.sleep(0.2)  # a slow refusal, as in production
            raise _overloaded()
        return _answer(f"answered by {model}")

    with patch.object(svc, "_get_client", return_value=_client(generate)):
        result = await svc.analyze_checklist_item("readiness_indicator", b"not-a-real-video", "video/mp4")

    assert result.notes == f"answered by {svc.GEMINI_VIDEO_MODELS[1]}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_healthy_first_model_still_wins():
    async def generate(*, model, **_):
        if model == svc.GEMINI_VIDEO_MODELS[0]:
            return _answer("answered by the stronger model")
        await asyncio.sleep(1)  # the fallback would be slower
        return _answer("answered by the fallback")

    with patch.object(svc, "_get_client", return_value=_client(generate)):
        result = await svc.analyze_checklist_item("readiness_indicator", b"not-a-real-video", "video/mp4")

    assert result.notes == "answered by the stronger model"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_transient_overload_on_a_photo_is_retried(monkeypatch):
    monkeypatch.setattr(svc, "GEMINI_IMAGE_FALLBACKS", ())  # one model: it must retry itself
    calls = {"n": 0}

    async def generate(*, model, **_):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _overloaded()
        return _answer("third time lucky")

    with patch.object(svc, "_get_client", return_value=_client(generate)):
        result = await svc.analyze_checklist_item("aed_cabinet", b"fake-jpeg-bytes", "image/jpeg")

    assert result.notes == "third time lucky"
    assert calls["n"] == 3


def _out_of_quota(per: str, retry_after: str = "34s") -> errors.ClientError:
    return errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": f"GenerateRequestsPer{per}PerProjectPerModel-FreeTier"}],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_after},
                ],
            }
        },
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_photo_model_out_of_daily_quota_hands_over_at_once(monkeypatch):
    # Measured 1 Oct 2026: a free-tier key allows a model 20 requests a day.
    # The backup must answer straight away, not after the hedge delay.
    monkeypatch.setattr(svc, "IMAGE_HEDGE_DELAY_SECONDS", 30.0)
    seen = []

    async def generate(*, model, **_):
        seen.append(model)
        if model == svc.GEMINI_IMAGE_MODEL:
            raise _out_of_quota("Day")
        return _answer(f"answered by {model}")

    with patch.object(svc, "_get_client", return_value=_client(generate)):
        result = await asyncio.wait_for(
            svc.analyze_checklist_item("aed_cabinet", b"fake-jpeg-bytes", "image/jpeg"), timeout=2
        )

    assert result.notes == f"answered by {svc.GEMINI_IMAGE_FALLBACKS[0]}"
    assert seen.count(svc.GEMINI_IMAGE_MODEL) == 1  # a daily refusal isn't retried
    assert result.meta.model == svc.GEMINI_IMAGE_FALLBACKS[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_per_minute_quota_refusal_is_waited_out(monkeypatch):
    monkeypatch.setattr(svc, "GEMINI_IMAGE_FALLBACKS", ())
    calls = {"n": 0}

    async def generate(*, model, **_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _out_of_quota("Minute", retry_after="0.05s")
        return _answer("answered after the minute's limit cleared")

    with patch.object(svc, "_get_client", return_value=_client(generate)):
        result = await svc.analyze_checklist_item("aed_cabinet", b"fake-jpeg-bytes", "image/jpeg")

    assert result.notes == "answered after the minute's limit cleared"
    assert calls["n"] == 2


@pytest.mark.unit
def test_a_quota_refusal_is_read_for_its_kind_and_wait():
    assert svc._quota_refusal(_out_of_quota("Day", "34s")) == (True, 34.0)
    assert svc._quota_refusal(_out_of_quota("Minute", "7s")) == (False, 7.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_stalled_call_is_abandoned_and_retried(monkeypatch):
    monkeypatch.setattr(svc, "PER_ATTEMPT_TIMEOUT_SECONDS", 0.1)
    calls = {"n": 0}

    async def generate(*, model, **_):
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(5)  # hangs far past the per-attempt limit
        return _answer("answered after a stall")

    with patch.object(svc, "_get_client", return_value=_client(generate)):
        result = await svc.analyze_checklist_item("aed_cabinet", b"fake-jpeg-bytes", "image/jpeg")

    assert result.notes == "answered after a stall"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_when_every_model_is_down_the_check_times_out_cleanly(monkeypatch):
    monkeypatch.setattr(svc, "OVERALL_TIMEOUT_SECONDS", 0.3)

    async def generate(**_):
        raise _overloaded()

    with patch.object(svc, "_get_client", return_value=_client(generate)):
        with pytest.raises(TimeoutError):
            await svc.analyze_checklist_item("aed_cabinet", b"fake-jpeg-bytes", "image/jpeg")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_daily_call_ceiling_stops_spending(monkeypatch):
    monkeypatch.setattr(svc, "DAILY_AI_CALL_LIMIT", 1)
    monkeypatch.setattr(svc, "_calls_today", {"day": None, "count": 0})
    monkeypatch.setattr(svc, "GEMINI_IMAGE_FALLBACKS", ())

    async def generate(**_):
        return _answer("ok")

    with patch.object(svc, "_get_client", return_value=_client(generate)):
        await svc.analyze_checklist_item("aed_cabinet", b"fake-jpeg-bytes", "image/jpeg")
        with pytest.raises(svc.DailyLimitReached):
            await svc.analyze_checklist_item("aed_cabinet", b"fake-jpeg-bytes", "image/jpeg")
    assert svc.ai_calls_today() == {"used": 1, "limit": 1}
