"""Checklist item analysis endpoint — one Gemini call per photo, per readiness
video, or per browser-extracted frame sequence. Every route requires the
shared-secret bearer token (see app/core/auth.py)."""
from __future__ import annotations

from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.auth import require_service_token
from app.services import gemini_checklist_service
from app.services.checklist_items import CHECKLIST_ITEMS, get_item
from app.services.device_profiles import PROFILES, get_profile

logger = structlog.get_logger(__name__)
router = APIRouter(dependencies=[Depends(require_service_token)])

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # matches the Node backend's photo limit
# A 10-15 s phone clip of the readiness light, sent whole so every frame can
# be scanned for the Ready light's flash (see readiness_frames.py).
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_FRAMES = 30  # generous headroom over the ~24-frame max the client ever sends


@router.get("/items")
async def list_items():
    """The checklist catalogue, so callers can mirror it without hand-syncing."""
    return {
        "items": [
            {
                "id": item.id,
                "section": item.section,
                "order": item.order,
                "title": item.title,
                "description": item.description,
                "mediaType": item.media_type,
                "required": item.required,
            }
            for item in CHECKLIST_ITEMS
        ]
    }


@router.get("/profiles")
async def list_profiles():
    """The AED models this service has manufacturer-sourced guidance for."""
    return {
        "profiles": [
            {"id": p.id, "name": p.name, "brand": p.brand, "blinkingReady": p.blinking_ready}
            for p in PROFILES.values()
        ]
    }


def _is_video(content_type: Optional[str]) -> bool:
    return bool(content_type) and content_type.startswith("video/")


@router.post("/{item_id}/analyze")
async def analyze_item(
    item_id: str,
    file: List[UploadFile] = File(...),
    # Which AED this is, as a profile id ('Philips FRx', 'Zoll AED Plus',
    # 'Defibtech Lifeline AUTO'…), so the prompt describes the unit actually
    # in the photo. Optional: no model gets the brand-neutral profile.
    aedModel: Optional[str] = Form(default=None),
    aed_model: Optional[str] = Form(default=None),
    lang: Optional[str] = Form(default=None),
):
    item = get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Unknown checklist item '{item_id}'")

    if len(file) > MAX_FRAMES:
        raise HTTPException(status_code=400, detail=f"Too many frames (max {MAX_FRAMES})")
    if item.media_type == "image" and len(file) != 1:
        raise HTTPException(status_code=400, detail="This item takes exactly one photo")

    single_video = len(file) == 1 and _is_video(file[0].content_type)
    if single_video and item.media_type != "video":
        raise HTTPException(status_code=400, detail="This item takes a photo, not a video")
    limit = MAX_VIDEO_BYTES if single_video else MAX_IMAGE_BYTES

    media: list[tuple[bytes, Optional[str]]] = []
    total_bytes = 0
    for upload in file:
        contents = await upload.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty upload")
        total_bytes += len(contents)
        if total_bytes > limit:
            raise HTTPException(status_code=413, detail="Upload too large")
        media.append((contents, upload.content_type))

    model_id = aedModel or aed_model
    try:
        if len(media) > 1:
            result = await gemini_checklist_service.analyze_checklist_item(
                item_id,
                b"",
                None,
                aed_model=model_id,
                language=lang,
                frames=[data for data, _ in media],
            )
        else:
            data, content_type = media[0]
            result = await gemini_checklist_service.analyze_checklist_item(
                item_id, data, content_type, aed_model=model_id, language=lang
            )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except gemini_checklist_service.DailyLimitReached as exc:
        raise HTTPException(status_code=503, detail="The AI service has reached today's limit.") from exc
    except TimeoutError as exc:
        logger.error("checklist.analyze_timeout", item_id=item_id, error=str(exc))
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("checklist.analyze_error", item_id=item_id, error=str(exc))
        raise HTTPException(status_code=502, detail="AI analysis failed") from exc

    payload = result.model_dump()
    profile = get_profile(model_id)
    payload["profile"] = {"id": profile.id, "name": profile.name}
    return payload
