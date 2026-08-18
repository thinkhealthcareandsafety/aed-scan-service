"""
AED Scan Service (beta) — FastAPI entry point.

Internal-only microservice: one Gemini-driven analysis endpoint, called
exclusively by the AEDSmartX Node backend (never directly by a browser).
Do not expose this service's port publicly — see README.md.
"""
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.config import settings
from app.routes import router as checklist_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("aed_scan_service.startup", version=settings.APP_VERSION)
    if not settings.GEMINI_API_KEY:
        logger.warning("aed_scan_service.gemini_api_key_missing")
    if not settings.CV_SERVICE_TOKEN:
        logger.warning("aed_scan_service.cv_service_token_missing")
    yield
    logger.info("aed_scan_service.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # No CORS middleware: this service is never called from a browser, only
    # server-to-server from the Node backend. Deliberately no public route
    # exists for a browser to hit in the first place.
    app.include_router(checklist_router, prefix="/api/v1/checklist", tags=["checklist"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
