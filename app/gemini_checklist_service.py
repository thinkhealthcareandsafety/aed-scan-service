"""
Gemini Vision Service — one multimodal call per uploaded photo, scoped to a
single checklist item. Adapted from
github.com/thinkhealthcareandsafety/Aed-inspection-platform (trimmed to the
3 items this phase needs; the video/frame-extraction path for the readiness
indicator is intentionally left out of Phase 1).
"""
from __future__ import annotations

import asyncio
from typing import Optional

import structlog
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.config import settings
from app.checklist_items import ChecklistItem, get_item
from app import validators
from app.date_parser import parse_expiry_date

logger = structlog.get_logger(__name__)

# Flash-lite is enough for a single still frame/photo — these 3 items are
# photo-only in Phase 1, so this is the only model this service needs.
GEMINI_IMAGE_MODEL = "gemini-3.1-flash-lite"

REQUEST_TIMEOUT_SECONDS = 40.0


class ChecklistAnalysisResult(BaseModel):
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str
    serial_number: Optional[str] = None
    expiry_date: Optional[str] = None
    expiry_raw_text: Optional[str] = None
    lot_number: Optional[str] = None
    battery_serial_number: Optional[str] = None


_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _build_prompt(item: ChecklistItem) -> str:
    return (
        "You are the vision engine for an AED (defibrillator) inspection "
        "checklist app. You receive one photo for exactly one checklist "
        "item and must return a single structured verdict.\n\n"
        f"Checklist item: {item.title}\n"
        f"Task: {item.prompt}\n\n"
        "Always set confidence (0.0-1.0) to your own honest certainty in "
        "this specific photo — a blurry, distant, dark, or ambiguous capture "
        "should score low even if you still produced a best-effort answer. "
        "notes is one short, friendly sentence: if passed=false, tell the "
        "inspector exactly what to fix or recapture; if passed=true, briefly "
        "confirm what you saw. Leave any data field you cannot determine as "
        "null — never guess."
    )


def _mime_type_for(declared_content_type: Optional[str]) -> str:
    if declared_content_type and "/" in declared_content_type:
        return declared_content_type
    return "image/jpeg"


async def analyze_checklist_item(
    item_id: str, image_bytes: bytes, content_type: Optional[str] = None
) -> ChecklistAnalysisResult:
    item = get_item(item_id)
    if item is None:
        raise ValueError(f"Unknown checklist item: {item_id}")

    client = _get_client()
    mime_type = _mime_type_for(content_type)
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        _build_prompt(item),
    ]

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ChecklistAnalysisResult,
                    temperature=0.1,
                ),
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Gemini call for checklist item={item_id} exceeded {REQUEST_TIMEOUT_SECONDS}s"
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
