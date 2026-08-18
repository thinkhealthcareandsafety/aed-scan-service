"""Checklist item analysis endpoint — one Gemini call per uploaded photo.
Every route requires the shared-secret bearer token (see app/auth.py)."""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app import gemini_checklist_service
from app.auth import require_service_token
from app.checklist_items import get_item

logger = structlog.get_logger(__name__)
router = APIRouter(dependencies=[Depends(require_service_token)])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # matches the Node backend's multer limit


@router.post("/{item_id}/analyze")
async def analyze_item(item_id: str, file: UploadFile = File(...)):
    item = get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Unknown checklist item '{item_id}'")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        result = await gemini_checklist_service.analyze_checklist_item(
            item_id, contents, file.content_type
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TimeoutError as exc:
        logger.error("checklist.analyze_timeout", item_id=item_id, error=str(exc))
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("checklist.analyze_error", item_id=item_id, error=str(exc))
        raise HTTPException(status_code=502, detail="AI analysis failed") from exc

    return result.model_dump()
