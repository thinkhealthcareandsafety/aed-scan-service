"""Checklist item analysis endpoint — one Gemini call per photo item, or per
browser-extracted frame sequence for the video (readiness-indicator) item.
Every route requires the shared-secret bearer token (see app/auth.py)."""
from __future__ import annotations

from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app import gemini_checklist_service
from app.auth import require_service_token
from app.checklist_items import get_item

logger = structlog.get_logger(__name__)
router = APIRouter(dependencies=[Depends(require_service_token)])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # matches the Node backend's multer limit
MAX_FRAMES = 30  # generous headroom over the ~24-frame max the client ever sends


@router.post("/{item_id}/analyze")
async def analyze_item(
    item_id: str,
    file: List[UploadFile] = File(...),
    aedModel: Optional[str] = Form(default=None),
):
    item = get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Unknown checklist item '{item_id}'")

    if len(file) > MAX_FRAMES:
        raise HTTPException(status_code=400, detail=f"Too many frames (max {MAX_FRAMES})")
    if item.media_type == "image" and len(file) != 1:
        raise HTTPException(status_code=400, detail="This item takes exactly one photo")

    media: list[tuple[bytes, str | None]] = []
    total_bytes = 0
    for upload in file:
        contents = await upload.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty upload")
        total_bytes += len(contents)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Upload too large")
        media.append((contents, upload.content_type))

    try:
        result = await gemini_checklist_service.analyze_checklist_item(item_id, media, aedModel)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TimeoutError as exc:
        logger.error("checklist.analyze_timeout", item_id=item_id, error=str(exc))
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("checklist.analyze_error", item_id=item_id, error=str(exc))
        raise HTTPException(status_code=502, detail="AI analysis failed") from exc

    return result.model_dump()
