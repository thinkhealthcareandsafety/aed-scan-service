"""
Gemini Vision Service — one multimodal call per checklist item: a single
photo for the 3 label-reading items, or a browser-extracted frame sequence
for the readiness-indicator item. Adapted from
github.com/thinkhealthcareandsafety/Aed-inspection-platform, with the
readiness-indicator prompt, model-context map, and timeout/retry tuning
ported from the sibling aed-readiness-campaign project's
lib/inspectionGemini.js (that project's numbers came from real measured
Gemini latency, not guesses — see the comment on REQUEST_TIMEOUT_SECONDS
below).
"""
from __future__ import annotations

import asyncio
import random
from typing import Optional

import structlog
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.config import settings
from app.checklist_items import ChecklistItem, get_item, model_context_for
from app import validators
from app.date_parser import parse_expiry_date

logger = structlog.get_logger(__name__)

# Flash-lite is enough for a single still frame/photo. The readiness-
# indicator item (a multi-frame sequence) benefits from the stronger
# reasoning of the full flash model when judging a sequence of frames.
GEMINI_IMAGE_MODEL = "gemini-3.1-flash-lite"
GEMINI_VIDEO_MODEL = "gemini-3.5-flash"

# The image items return in ~1-2s, so 25s is generous headroom. The video
# item is a genuinely different story — the sibling project measured
# 12-frame calls to gemini-3.5-flash taking 26s-50s across repeated real
# trials; a single 40s timeout for both was killing legitimate,
# still-in-progress video calls right before they would have succeeded.
# This isn't a quota/billing problem, it's this model's real latency for a
# multi-image request being longer than a naive single timeout assumes.
REQUEST_TIMEOUT_SECONDS_BY_MODEL = {
    GEMINI_IMAGE_MODEL: 25.0,
    GEMINI_VIDEO_MODEL: 75.0,
}
DEFAULT_REQUEST_TIMEOUT_SECONDS = 75.0


class ChecklistAnalysisResult(BaseModel):
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str
    serial_number: Optional[str] = None
    expiry_date: Optional[str] = None
    expiry_raw_text: Optional[str] = None
    lot_number: Optional[str] = None
    battery_serial_number: Optional[str] = None
    status: Optional[str] = None  # readiness_indicator only: "ready" | "fault" | "unclear"
    key_frame_index: Optional[int] = None  # readiness_indicator only, 1-based


_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _build_prompt(item: ChecklistItem, aed_model: Optional[str], frame_count: Optional[int]) -> str:
    sequence_note = ""
    if frame_count:
        sequence_note = (
            f"You are given {frame_count} still frames extracted evenly across a short video "
            "clip, in strict chronological order (the first image is the start of the clip, the "
            "last is the end). Treat them as one continuous observation of the same scene over "
            "time, not as separate unrelated photos — a change that appears in only one or two "
            "of the frames (e.g. a light turning on then off again) is exactly the kind of brief "
            "event you are looking for, not noise to discard.\n\n"
            f"Also set key_frame_index to the 1-based position (1..{frame_count}) of the single "
            "frame that most clearly shows your evidence — e.g. the frame the flash is visible "
            "in, or simply the sharpest, most in-focus frame if nothing decisive stands out. "
            "This is used to pick which exact frame gets saved as the audit record, so it should "
            "genuinely be the most representative one, not always frame 1.\n\n"
        )

    return (
        "You are the vision engine for an AED (defibrillator) inspection checklist app. You "
        "receive photo(s) for exactly one checklist item and must return a single structured "
        "verdict.\n\n"
        f"{model_context_for(aed_model)}\n\n"
        f"{sequence_note}"
        f"Checklist item: {item.title}\n"
        f"Task: {item.prompt}\n\n"
        "Always set confidence (0.0-1.0) to your own honest certainty in this specific media — "
        "a blurry, distant, dark, or ambiguous capture should score low even if you still "
        "produced a best-effort answer. notes is one short, friendly sentence: if passed=false, "
        "tell the inspector exactly what to fix or recapture; if passed=true, briefly confirm "
        "what you saw. Leave any data field you cannot determine as null — never guess."
    )


def _mime_type_for(declared_content_type: Optional[str]) -> str:
    if declared_content_type and "/" in declared_content_type:
        return declared_content_type
    return "image/jpeg"


# Retries only the failure modes that are actually transient. A burst of
# concurrent scans is exactly what trips Gemini's own per-minute rate
# limit; without a retry, that surfaces to the inspector as a flat
# "AI analysis failed" with no recovery. A malformed request or an
# implausible read won't succeed on attempt 2, so those still fail
# immediately (see _is_retryable / _is_hard_quota_exhaustion).
_RETRYABLE_STATUS = {429, 503}


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS:
        return True
    msg = str(exc)
    return any(
        token in msg
        for token in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded", "rate limit")
    ) or "rate_limit" in msg.lower()


def _is_hard_quota_exhaustion(exc: Exception) -> bool:
    # RESOURCE_EXHAUSTED covers two very different situations that
    # _is_retryable can't tell apart from the status code alone: a
    # short-lived per-minute rate limit (worth retrying) and a hard
    # daily/monthly quota or billing cap (won't clear no matter how long
    # this one request waits). Gemini's own error message spells out the
    # latter with the word "quota" specifically.
    return "quota" in str(exc).lower()


async def _with_retry(fn, attempts: int = 3):
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below if not retryable
            last_exc = exc
            if i == attempts - 1 or _is_hard_quota_exhaustion(exc) or not _is_retryable(exc):
                raise
            backoff = 0.6 * (2**i) + random.random() * 0.3
            await asyncio.sleep(backoff)
    raise last_exc  # pragma: no cover - loop always returns or raises above


async def analyze_checklist_item(
    item_id: str,
    media: list[tuple[bytes, Optional[str]]],
    aed_model: Optional[str] = None,
) -> ChecklistAnalysisResult:
    """
    media: one (bytes, content_type) entry for a photo item, several
    chronological (bytes, content_type) entries — one per extracted frame —
    for the readiness-indicator item.
    """
    item = get_item(item_id)
    if item is None:
        raise ValueError(f"Unknown checklist item: {item_id}")
    if not media:
        raise ValueError("No media provided")

    is_sequence = len(media) > 1
    model = GEMINI_VIDEO_MODEL if item.media_type == "video" else GEMINI_IMAGE_MODEL

    parts = [
        types.Part.from_bytes(data=data, mime_type=_mime_type_for(content_type))
        for data, content_type in media
    ]
    parts.append(_build_prompt(item, aed_model, len(media) if is_sequence else None))

    client = _get_client()
    timeout_seconds = REQUEST_TIMEOUT_SECONDS_BY_MODEL.get(model, DEFAULT_REQUEST_TIMEOUT_SECONDS)

    async def _call():
        return await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ChecklistAnalysisResult,
                    temperature=0.1,
                ),
            ),
            timeout=timeout_seconds,
        )

    try:
        response = await _with_retry(_call)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Gemini call for checklist item={item_id} exceeded {timeout_seconds}s"
        ) from exc

    parsed = response.parsed
    result = (
        parsed
        if isinstance(parsed, ChecklistAnalysisResult)
        else ChecklistAnalysisResult.model_validate_json(response.text)
    )

    return _apply_deterministic_checks(item, result)


def _apply_deterministic_checks(
    item: ChecklistItem, result: ChecklistAnalysisResult
) -> ChecklistAnalysisResult:
    """Downgrade an implausible read — cheap, deterministic sanity checks
    independent of Gemini's own confidence score. Never upgrades a result,
    only vetoes bad ones."""
    # Contract-compliance floor, applied to every item: the prompt requires
    # a short explanatory sentence for every verdict. A response that skips
    # it is a sign of a malformed or truncated read, not a genuinely
    # confident one, even if Gemini's own confidence score claims
    # otherwise.
    if not result.notes or not result.notes.strip():
        return result.model_copy(
            update={
                "confidence": min(result.confidence or 0.0, 0.4),
                "notes": "AI did not explain this result — please verify it yourself.",
            }
        )

    if item.id == "serial_number" and result.serial_number:
        if not validators.is_plausible_serial(result.serial_number):
            logger.warning("checklist.implausible_serial", value=result.serial_number)
            return result.model_copy(
                update={
                    "passed": False,
                    "serial_number": None,
                    "notes": "Serial number reading looked implausible — please recapture with the label centred and in focus.",
                }
            )

    if item.id in ("pads_expiry", "battery_expiry") and result.expiry_date:
        plausible = validators.is_plausible_expiry(result.expiry_date)
        agrees = validators.expiry_cross_check_agrees(result.expiry_date, result.expiry_raw_text)
        if not plausible or not agrees:
            logger.warning(
                "checklist.implausible_expiry",
                item=item.id,
                value=result.expiry_date,
                raw=result.expiry_raw_text,
            )
            return result.model_copy(
                update={
                    "passed": False,
                    "expiry_date": None,
                    "notes": "Expiry date reading looked implausible — please recapture with the label centred, well lit, and in focus.",
                }
            )
        if result.expiry_raw_text:
            deterministic = parse_expiry_date(result.expiry_raw_text)
            if deterministic and len(deterministic) > len(result.expiry_date):
                result = result.model_copy(update={"expiry_date": deterministic})

    return result
