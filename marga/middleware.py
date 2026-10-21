"""FastAPI middleware for authentication and CORS.

Provides optional bearer-token authentication and CORS headers for
cross-origin requests from web chat interfaces.
"""

from __future__ import annotations

import os
from typing import Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Require Bearer token authentication when MARGA_AUTH_TOKEN is set.

    Accepts tokens via:
    - Authorization: Bearer <token>
    - x-api-key: <token> (Anthropic-style clients)

    Public endpoints (/health, /docs, /redoc, /openapi.json) are always allowed.
    """

    def __init__(self, app: FastAPI, token: str | None = None):
        """Initialize auth middleware.

        Args:
            app: FastAPI application.
            token: Required bearer token. If None, auth is disabled.
        """
        super().__init__(app)
        self.token = token
        self.public_paths = {"/health", "/docs", "/redoc", "/openapi.json", "/"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check bearer token before routing request.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/route handler.

        Returns:
            Response from the handler or 401 Unauthorized.
        """
        # Auth disabled
        if self.token is None:
            return await call_next(request)

        # Public endpoints
        if request.url.path in self.public_paths:
            return await call_next(request)

        # Check Authorization: Bearer <token>
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            provided = auth_header[7:]  # Strip "Bearer "
            if self._constant_time_compare(provided, self.token):
                return await call_next(request)

        # Check x-api-key: <token> (Anthropic-style)
        api_key = request.headers.get("x-api-key", "")
        if api_key and self._constant_time_compare(api_key, self.token):
            return await call_next(request)

        # Unauthorized
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Unauthorized"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @staticmethod
    def _constant_time_compare(a: str, b: str) -> bool:
        """Compare two strings in constant time to prevent timing attacks.

        Args:
            a: First string.
            b: Second string.

        Returns:
            True if strings are equal.
        """
        if len(a) != len(b):
            return False
        result = 0
        for x, y in zip(a.encode(), b.encode(), strict=False):
            result |= x ^ y
        return result == 0


def add_cors(app: FastAPI, origins: list[str] | None = None) -> None:
    """Add CORS middleware to the FastAPI app.

    Args:
        app: FastAPI application.
        origins: Allowed origins. None = no CORS, ["*"] = allow all.

    Example:
        add_cors(app, origins=["https://chat.example.com", "http://localhost:3000"])
    """
    if not origins:
        return

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def get_auth_token() -> str | None:
    """Read bearer auth token from MARGA_AUTH_TOKEN environment variable.

    Returns:
        Token string or None if not set.
    """
    return os.getenv("MARGA_AUTH_TOKEN", "").strip() or None


def get_cors_origins() -> list[str] | None:
    """Read CORS allowed origins from ALLOWED_ORIGINS environment variable.

    Expects comma-separated list of origins, or "*" for allow-all.

    Returns:
        List of origin strings, or None if not set.

    Example:
        ALLOWED_ORIGINS="https://chat.example.com,http://localhost:3000"
        ALLOWED_ORIGINS="*"
    """
    origins_str = os.getenv("ALLOWED_ORIGINS", "").strip()
    if not origins_str:
        return None

    if origins_str == "*":
        return ["*"]

    return [origin.strip() for origin in origins_str.split(",") if origin.strip()]
