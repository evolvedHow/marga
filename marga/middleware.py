"""FastAPI middleware for authentication, request contexts and CORS.

Two authentication styles are supported because two client families exist:

* :class:`ContextMiddleware` - the default. Requires a bearer token, maps it to
  a request context (see :mod:`marga.contexts`), and lets the caller select
  another context only when that token is explicitly allowed to.
* :class:`BearerAuthMiddleware` - a single shared token with no context
  concept, kept for direct use and for deployments that predate contexts.
"""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable, Mapping

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from .contexts import CONTEXT_HEADER, CONTEXT_PATH_PREFIX, ContextResolver
from .errors import MargaError, error_payload_for_path, http_status_for

#: Endpoints reachable without a key (probes, docs, static metadata).
DEFAULT_PUBLIC_PATHS = frozenset({"/", "/health", "/docs", "/redoc", "/openapi.json"})


def extract_token(headers: Mapping[str, str]) -> str:
    """Read a bearer token from a request's headers.

    Args:
        headers: Request headers (case-insensitive mapping).

    Returns:
        The token, or an empty string when the client sent none.

    Example:
        >>> extract_token({"authorization": "Bearer sk-abc"})
        'sk-abc'
        >>> extract_token({"x-api-key": "sk-abc"})
        'sk-abc'
    """
    auth = headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return headers.get("x-api-key", "").strip()


def split_context_prefix(path: str) -> tuple[str, str]:
    """Split a ``/c/<context>`` path prefix off a request path.

    Args:
        path: Request path, e.g. ``"/c/nonprofit/v1/chat/completions"``.

    Returns:
        ``(context, remainder)``. ``("", path)`` when there is no prefix.

    Example:
        >>> split_context_prefix("/c/nonprofit/v1/models")
        ('nonprofit', '/v1/models')
        >>> split_context_prefix("/v1/models")
        ('', '/v1/models')
    """
    if not path.startswith(CONTEXT_PATH_PREFIX):
        return "", path
    remainder = path[len(CONTEXT_PATH_PREFIX) :]
    name, _, rest = remainder.partition("/")
    if not name:
        return "", path
    return name, "/" + rest if rest else "/"


class ContextMiddleware(BaseHTTPMiddleware):
    """Authenticate the caller and attach its request context.

    Runs before routing so every endpoint sees ``request.state.marga_context``.
    Errors are rendered in the envelope of the ingress API, because an
    Anthropic SDK cannot parse an OpenAI error body.
    """

    def __init__(
        self,
        app: FastAPI,
        resolver: ContextResolver,
        *,
        public_paths: frozenset[str] | None = None,
        loader: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Initialise the middleware.

        Args:
            app: FastAPI application.
            resolver: Context resolver deciding who may use which context.
            public_paths: Paths served without authentication.
            loader: Async callable that fetches context tokens from the vault.
                Called once, lazily, when tokens are still pending - which is
                what makes authentication work on servers that do not emit
                ASGI lifespan events.
        """
        super().__init__(app)
        self.resolver = resolver
        self.public_paths = public_paths or DEFAULT_PUBLIC_PATHS
        self.loader = loader
        self._load_lock = asyncio.Lock()

    async def _ensure_tokens(self) -> None:
        """Fetch pending context tokens once, tolerating concurrent requests."""
        if self.loader is None or not self.resolver.token_secrets:
            return
        async with self._load_lock:
            if self.resolver.token_secrets:
                await self.loader()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Resolve the caller's context, then continue.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            The handler's response, or an error response when the context
            cannot be resolved.
        """
        path_context, stripped = split_context_prefix(request.url.path)
        if stripped != request.url.path:
            request.scope["path"] = stripped

        if stripped in self.public_paths:
            request.state.marga_context = self.resolver.default
            return await call_next(request)

        await self._ensure_tokens()

        # A selector never widens a token's reach; the resolver enforces that.
        # With authentication disabled there is no trustworthy selector, so it
        # is ignored rather than honoured.
        requested = ""
        if self.resolver.required:
            requested = request.headers.get(CONTEXT_HEADER, "").strip() or path_context

        try:
            context = self.resolver.resolve(extract_token(request.headers), requested or None)
        except MargaError as exc:
            status_code = http_status_for(exc)
            headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else {}
            return JSONResponse(
                status_code=status_code,
                content=error_payload_for_path(
                    stripped, status_code, str(exc), kind=type(exc).__name__
                ),
                headers=headers,
            )

        request.state.marga_context = context
        return await call_next(request)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Require a single shared Bearer token when one is configured.

    Accepts tokens via:
    - Authorization: Bearer <token>
    - x-api-key: <token> (Anthropic-style clients)

    Public endpoints (/health, /docs, /redoc, /openapi.json) are always allowed.
    Use :class:`ContextMiddleware` instead when several callers must be billed
    to different upstream accounts.
    """

    def __init__(self, app: FastAPI, token: str | None = None):
        """Initialize auth middleware.

        Args:
            app: FastAPI application.
            token: Required bearer token. If None, auth is disabled.
        """
        super().__init__(app)
        self.token = token
        self.public_paths = set(DEFAULT_PUBLIC_PATHS)

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

        provided = extract_token(request.headers)
        if provided and self._constant_time_compare(provided, self.token):
            return await call_next(request)

        # Unauthorized
        status_code = status.HTTP_401_UNAUTHORIZED
        return JSONResponse(
            status_code=status_code,
            content=error_payload_for_path(
                request.url.path, status_code, "Unauthorized", kind="AuthenticationError"
            ),
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
