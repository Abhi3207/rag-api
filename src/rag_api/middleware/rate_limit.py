"""
Rate limiting middleware using SlowAPI.

Provides a pre-configured limiter instance and a helper to install it on
the FastAPI application.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def _key_func(request: Request) -> str:
    """Extract the client identifier for rate-limiting (IP address)."""
    return get_remote_address(request)


# Module-level limiter instance — imported by route modules
limiter = Limiter(key_func=_key_func)


def install_rate_limiter(app: FastAPI) -> None:
    """Wire the limiter into *app*'s state and exception handlers."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
