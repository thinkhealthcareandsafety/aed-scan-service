"""
Bearer-token auth for this internal service. This is deliberately the *second*
layer of defense, not the first — the primary control is that this service
should never be exposed on a public port (see README). This check exists so
that even a network misconfiguration doesn't turn into an open Gemini-quota
sink for anyone who finds the port.
"""
import hmac

from fastapi import Header, HTTPException

from app.config import settings


async def require_service_token(authorization: str = Header(default="")) -> None:
    if not settings.CV_SERVICE_TOKEN:
        # Refuse to run "open" — an unset token is a misconfiguration, not an
        # invitation to skip the check.
        raise HTTPException(status_code=503, detail="Service not configured")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token, settings.CV_SERVICE_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid token")
