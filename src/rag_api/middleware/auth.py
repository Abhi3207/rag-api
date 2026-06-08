"""
Optional API key authentication middleware.

Enabled by setting ``API_KEY_ENABLED=true`` and ``API_KEY=<secret>`` in the
environment.  When enabled every request (except ``/health`` and ``/docs``)
must include an ``X-API-Key`` header matching the configured key.
"""

from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from src.rag_api.config import settings

# Paths that never require authentication
_PUBLIC_PATHS: set[str] = {
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid ``X-API-Key`` header.

    Does nothing if ``settings.API_KEY_ENABLED`` is ``False``.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if not settings.API_KEY_ENABLED:
            return await call_next(request)

        # Allow public endpoints through
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != settings.API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key."},
            )

        return await call_next(request)

