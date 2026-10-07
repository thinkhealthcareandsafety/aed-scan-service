"""
AED Scan Service — FastAPI entry point.

Internal-only microservice: one Gemini-driven analysis endpoint, called
exclusively by the AEDSmartX Node backend (never directly by a browser).
Do not expose this service's port publicly — see README.md.

The analysis engine (app/services, app/utils) is kept file-for-file in step
with github.com/thinkhealthcareandsafety/Aed-inspection-platform's python-cv
service, so improvements there can be copied across unchanged.
"""
import socket
from contextlib import asynccontextmanager

# Some container hosts (e.g. Render) advertise AAAA records for Google's API
# but have no IPv6 route at all — a plain socket connect() fails instantly
# with "Network unreachable", but the async HTTP client used by the Gemini
# SDK doesn't fall back to IPv4 as fast, and calls hang until the timeout
# kills them. Forcing IPv4-only resolution for the whole process sidesteps
# that entirely; every outbound call in this service only needs IPv4.
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_only_getaddrinfo

import structlog  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.api.routes.checklist import router as checklist_router  # noqa: E402
from app.services.gemini_checklist_service import ai_calls_today  # noqa: E402

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
        return {"status": "ok", "version": settings.APP_VERSION, "ai_calls": ai_calls_today()}

    return app


app = create_app()
