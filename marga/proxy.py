"""The :class:`Proxy` - middleware, routing and the OpenAI HTTP surface.

``Proxy`` ties the pieces together and is the only class most applications
touch. Declare backends, decorate the behaviour you want, serve:

    from marga import Proxy, round_robin

    proxy = Proxy(backends={
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key_secret": "KEY_GROQ",       # read from the worker-env vault
            "models": ["llama-3.3-70b-versatile"],
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key_secret": "KEY_OPENAI",
            "models": ["gpt-4o-mini"],
            "cost_per_1k_tokens": 0.15,
        },
    })
    proxy.route(round_robin())

    @proxy.before
    def add_context(ctx):
        ctx.messages.insert(0, {"role": "system", "content": "Be terse."})

    proxy.serve()

The server exposes ``POST /v1/chat/completions``, ``GET /v1/models``,
``GET /health``, ``GET /stats`` and the FastAPI documentation at ``/docs``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from ._version import __version__
from .errors import (
    AllBackendsFailedError,
    BackendFailedError,
    BudgetExhaustedError,
    MargaError,
    UnknownBackendError,
    VaultNotConfiguredError,
    error_detail,
    http_status_for,
)
from .hooks import AfterHook, BeforeHook, ErrorHook, HookContext, HookManager
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    ModelCard,
    ModelListResponse,
    ProxyInfo,
    StatsResponse,
)
from .router import Backend, Router, build_backends
from .store import UsageStore
from .transport import Transport, feed_sse, usage_tokens
from .vault import VaultClient

logger = logging.getLogger("marga")

_BACKEND_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive"}


class _HookRecovery(Exception):
    """Internal signal: an error hook produced a replacement response.

    Attributes:
        payload: The dict an error hook returned.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__("error hook produced a replacement response")


class Proxy:
    """Transparent OpenAI-compatible routing proxy with hooks.

    Attributes:
        name: Application name, used as the FastAPI title and vault context
            fallback is taken from ``MARGA_VAULT_CONTEXT`` or ``"marga"``.
        router: The :class:`~marga.router.Router` making routing decisions.
        hooks: The :class:`~marga.hooks.HookManager` holding before/after/error hooks.
        transport: HTTP transport used to reach backends.
        vault: Vault client; created from the environment on first need.
    """

    def __init__(
        self,
        backends: Mapping[str, Mapping[str, Any]],
        *,
        store: UsageStore | None = None,
        strategy: Any | None = None,
        failover: bool = True,
        vault: VaultClient | None = None,
        vault_context: str | None = None,
        transport: Transport | None = None,
        name: str = "marga",
    ) -> None:
        """Create a proxy.

        Args:
            backends: Backend configuration keyed by backend name. Each entry
                accepts ``base_url`` (required), ``api_key``,
                ``api_key_secret`` (vault key), ``api_key_env``, ``models``,
                ``cost_per_1k_tokens``, ``headers`` and ``enabled``.
            store: Usage accounting backend; in-memory when omitted.
            strategy: Default routing strategy or route function.
            failover: Retry the next backend when one fails.
            vault: Vault client to use for ``api_key_secret`` lookups. When
                omitted, one is created from ``VAULT_URL``/``VAULT_TOKEN`` on
                first need.
            vault_context: Vault context name; defaults to
                ``MARGA_VAULT_CONTEXT`` or ``"marga"``.
            transport: Custom transport (tests, or a runtime-specific client).
            name: Application name.

        Raises:
            ValueError: When the backend configuration is invalid.
        """
        self.name = name
        self.router = Router(
            build_backends(dict(backends)),
            store=store,
            failover=failover,
            strategy=strategy,
        )
        self.hooks = HookManager()
        self.transport = transport or Transport()
        self.vault = vault
        self.vault_context = (
            vault_context or os.environ.get("MARGA_VAULT_CONTEXT", "").strip() or "marga"
        )
        self._app: FastAPI | None = None
        self._secrets_lock = asyncio.Lock()
        self._secrets_resolved = False

    # ------------------------------------------------------------------
    # Decorators
    # ------------------------------------------------------------------
    def route(self, target: Any) -> Any:
        """Register a routing strategy or a custom routing function.

        Args:
            target: Either a strategy object exposing
                ``async def select(ctx, router)`` (see
                :mod:`marga.strategies`) or a function
                ``fn(ctx) -> "backend/model"``. Both may be synchronous or
                asynchronous.

        Returns:
            ``target`` unchanged, so ``@proxy.route`` works as a decorator.

        Example:
            proxy.route(round_robin())

            @proxy.route
            def by_length(ctx):
                return "local/llama3.2-1b" if ctx.estimated_tokens < 500 else "openai/gpt-4o-mini"
        """
        self.router.set_router(target)
        return target

    def before(self, fn: BeforeHook) -> BeforeHook:
        """Register a before hook that may mutate the request context.

        Args:
            fn: ``fn(ctx)``; may be ``async``.

        Returns:
            ``fn``, so this works as a decorator.

        Example:
            @proxy.before
            def enforce_system_prompt(ctx):
                ctx.messages.insert(0, {"role": "system", "content": "Be terse."})
        """
        return self.hooks.add_before(fn)

    def after(self, fn: AfterHook) -> AfterHook:
        """Register an after hook that may transform the response.

        Args:
            fn: ``fn(ctx, response) -> response``; may be ``async``.

        Returns:
            ``fn``, so this works as a decorator.

        Example:
            @proxy.after
            def add_model_header(ctx, response):
                response["marga_backend"] = ctx.metadata.get("backend")
                return response
        """
        return self.hooks.add_after(fn)

    def on_error(self, fn: ErrorHook) -> ErrorHook:
        """Register an error hook notified when a backend attempt fails.

        Args:
            fn: ``fn(ctx, exc)``. Return a dict to serve it as the response
                and stop failover; return ``None`` to continue to the next
                backend. May be ``async``.

        Returns:
            ``fn``, so this works as a decorator.

        Example:
            @proxy.on_error
            def fallback_message(ctx, exc):
                if isinstance(exc, BudgetExhaustedError):
                    return {"id": "local", "choices": [], "marga_error": str(exc)}
        """
        return self.hooks.add_error(fn)

    # ------------------------------------------------------------------
    # Secrets
    # ------------------------------------------------------------------
    def _vault_required(self) -> VaultClient:
        """Return the vault client, building one from the environment if needed.

        Returns:
            A usable :class:`~marga.vault.VaultClient`.

        Raises:
            VaultNotConfiguredError: When no client was supplied and
                ``VAULT_URL`` / ``VAULT_TOKEN`` are not set.
        """
        if self.vault is None:
            client = VaultClient.from_env(context=self.vault_context)
            if client is None:
                raise VaultNotConfiguredError(["VAULT_URL", "VAULT_TOKEN"])
            self.vault = client
        return self.vault

    async def resolve_secrets(self, *, refresh: bool = False) -> None:
        """Populate backend API keys from the vault or the environment.

        Backends declaring ``api_key_secret`` are looked up in the worker-env
        vault (context ``marga`` unless overridden); backends declaring
        ``api_key_env`` read the process environment instead. Called eagerly
        by the server lifespan and lazily before the first request, so it also
        works on runtimes that do not emit ASGI lifespan events.

        Args:
            refresh: Bypass the vault's in-memory cache.

        Raises:
            VaultNotConfiguredError: When a vault key is required but the
                vault credentials are missing.
            SecretNotFoundError: When the vault has no such key.
        """
        for backend in self.router.backends.values():
            if backend.api_key_secret:
                vault = self._vault_required()
                backend.api_key = await vault.get(backend.api_key_secret, refresh=refresh)
            elif backend.api_key_env and not backend.api_key:
                value = os.environ.get(backend.api_key_env, "").strip()
                if value:
                    backend.api_key = value
        self._secrets_resolved = True

    async def _ensure_secrets(self) -> None:
        """Resolve secrets once per process, tolerating concurrent requests."""
        if self._secrets_resolved:
            return
        async with self._secrets_lock:
            if not self._secrets_resolved:
                await self.resolve_secrets()

    async def aclose(self) -> None:
        """Release HTTP clients owned by this proxy."""
        await self.transport.close()
        if self.vault is not None:
            await self.vault.close()

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    def build_app(self) -> FastAPI:
        """Build (and cache) the FastAPI application.

        Returns:
            The FastAPI app exposing the OpenAI-compatible surface.
        """
        if self._app is not None:
            return self._app

        @asynccontextmanager
        async def lifespan(_: FastAPI) -> AsyncIterator[None]:
            await self.resolve_secrets()
            try:
                yield
            finally:
                await self.aclose()

        app = FastAPI(
            title=self.name,
            version=__version__,
            description=(
                "OpenAI-compatible LLM routing proxy. Requests are routed to the "
                "cheapest/least-used backend, optionally constrained by free-tier "
                "or spend budgets, with before/after hooks around every call."
            ),
            lifespan=lifespan,
        )

        @app.get("/", response_model=ProxyInfo, tags=["meta"], summary="Proxy metadata")
        async def index() -> ProxyInfo:
            """Describe this proxy and list its endpoints."""
            return ProxyInfo(
                name=self.name,
                version=__version__,
                docs="/docs",
                endpoints=[
                    "/v1/chat/completions",
                    "/v1/models",
                    "/health",
                    "/stats",
                ],
                backends=sorted(self.router.backends),
            )

        @app.get("/health", response_model=HealthResponse, tags=["meta"], summary="Liveness probe")
        async def health() -> HealthResponse:
            """Report process liveness, runtime, and backend availability."""
            from ._util import detect_runtime, is_loopback
            from .models import BackendHealth

            runtime = detect_runtime()
            backend_health = []

            for backend in self.router.backends.values():
                loopback = is_loopback(backend.base_url)

                # Get availability status if checking is enabled
                reachable = None
                reason = ""
                if self.router.check_availability and self.router._availability:
                    result = await self.router._availability.check(backend)
                    reachable = result.reachable
                    reason = result.reason

                backend_health.append(
                    BackendHealth(
                        name=backend.name,
                        url=backend.base_url,
                        enabled=backend.enabled,
                        loopback=loopback,
                        reachable=reachable,
                        reason=reason,
                    )
                )

            return HealthResponse(
                version=__version__,
                runtime=runtime,
                backends=backend_health,
            )

        @app.get(
            "/stats",
            response_model=StatsResponse,
            tags=["meta"],
            summary="Usage statistics",
        )
        async def stats(
            period: str = Query(
                "all",
                pattern="^(all|month|day)$",
                description="Reporting window for the counters.",
            )
        ) -> StatsResponse:
            """Return per-backend call, token and cost counters."""
            return StatsResponse(backends=await self.router.get_stats(period=period))

        @app.get(
            "/v1/models",
            response_model=ModelListResponse,
            tags=["openai"],
            summary="List available models",
        )
        async def list_models() -> ModelListResponse:
            """List every model exposed by the routing table."""
            data: list[ModelCard] = []
            for backend in self.router.enabled_backends():
                for model in backend.models:
                    data.append(
                        ModelCard(
                            id=f"{backend.name}/{model}",
                            created=0,
                            owned_by=backend.name,
                        )
                    )
            return ModelListResponse(data=data)

        @app.post(
            "/v1/chat/completions",
            response_model=ChatCompletionResponse,
            tags=["openai"],
            summary="Chat completions",
            response_description="OpenAI-compatible completion (or an SSE stream when stream=true).",
            responses={
                400: {"description": "Routing decision named an unknown backend."},
                429: {"description": "Every backend is out of budget."},
                502: {"description": "Every candidate backend failed."},
            },
        )
        async def chat_completions(
            request: ChatCompletionRequest,
        ) -> ChatCompletionResponse | StreamingResponse:
            """Route a chat completion to a backend, run hooks, account usage."""
            await self._ensure_secrets()
            return await self._handle_chat(request)

        # Anthropic /v1/messages endpoint
        @app.post(
            "/v1/messages",
            tags=["anthropic"],
            summary="Anthropic Messages API",
            response_description="Anthropic-compatible message response",
        )
        async def anthropic_messages(
            request: "AnthropicRequest",  # noqa: F821
        ) -> "AnthropicResponse | StreamingResponse":  # noqa: F821
            """Anthropic /v1/messages endpoint with translation to OpenAI format."""
            from .anthropic import (
                AnthropicRequest,
                anthropic_to_openai,
                openai_to_anthropic,
                translate_openai_stream,
            )

            await self._ensure_secrets()

            # Parse as Anthropic request
            anthropic_req = AnthropicRequest.model_validate(request)

            # Translate to OpenAI format
            openai_dict = anthropic_to_openai(anthropic_req)
            from .models import ChatCompletionRequest

            openai_req = ChatCompletionRequest.model_validate(openai_dict)

            # Route through internal OpenAI handler
            result = await self._handle_chat(openai_req)

            # Translate response back to Anthropic format
            if anthropic_req.stream:
                # Streaming: wrap and translate SSE events
                from .anthropic import translate_openai_stream

                async def anthropic_stream() -> AsyncIterator[bytes]:
                    async for chunk in translate_openai_stream(
                        result.body_iterator, anthropic_req.model
                    ):
                        yield chunk

                return StreamingResponse(
                    anthropic_stream(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            else:
                # Non-streaming: translate response dict
                response_dict = result.model_dump() if hasattr(result, "model_dump") else result
                anthropic_dict = openai_to_anthropic(response_dict, anthropic_req.model)
                from .anthropic import AnthropicResponse

                return AnthropicResponse.model_validate(anthropic_dict)

        # Add middleware
        from .middleware import (
            BearerAuthMiddleware,
            add_cors,
            get_auth_token,
            get_cors_origins,
        )

        # CORS (if configured)
        cors_origins = get_cors_origins()
        if cors_origins:
            add_cors(app, origins=cors_origins)

        # Bearer auth (if configured)
        auth_token = get_auth_token()
        if auth_token:
            app.add_middleware(BearerAuthMiddleware, token=auth_token)

        self._app = app
        return app

    async def _handle_chat(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse | StreamingResponse:
        """Execute one chat completion request end to end.

        Args:
            request: Validated request body.

        Returns:
            The backend response, an error-hook replacement, or a streaming
            response when ``request.stream`` is set.

        Raises:
            HTTPException: Mapped from marga errors (400/429/502).
        """
        body = request.model_dump(exclude_unset=True, exclude_none=True)
        ctx = HookContext(body=body, stream=bool(body.get("stream", False)))
        await self.hooks.run_before(ctx)

        try:
            candidates = await self.router.candidates(ctx)
        except MargaError as exc:
            recovered = await self.hooks.run_error(ctx, exc)
            if recovered is not None:
                return recovered  # type: ignore[return-value]
            raise _http_error(exc) from exc

        if ctx.stream:
            return await self._streaming_response(ctx, candidates)

        attempts: list[BackendFailedError] = []
        for backend, model in candidates:
            ctx.model = model
            try:
                response = await self.transport.complete(
                    backend.base_url, backend.api_key, ctx.body, headers=backend.headers
                )
            except (httpx.HTTPError, ValueError) as exc:
                failure = BackendFailedError(backend.name, exc)
                attempts.append(failure)
                logger.warning("Backend %s failed: %s", backend.name, exc)
                recovered = await self.hooks.run_error(ctx, failure)
                if recovered is not None:
                    return recovered  # type: ignore[return-value]
                continue

            response = await self.hooks.run_after(ctx, response)
            await self.router.record_usage(backend.name, _extract_tokens(response))
            return response  # type: ignore[return-value]

        error = AllBackendsFailedError(attempts)
        recovered = await self.hooks.run_error(ctx, error)
        if recovered is not None:
            return recovered  # type: ignore[return-value]
        raise _http_error(error) from error

    async def _streaming_response(
        self,
        ctx: HookContext,
        candidates: Sequence[tuple[Backend, str]],
    ) -> StreamingResponse:
        """Open the first healthy backend stream and wrap it for the client.

        The first backend is opened *before* the response starts so that
        failover still works: a backend that fails on connect is skipped while
        the client has not yet received headers.

        Args:
            ctx: Request context.
            candidates: Ordered ``(backend, model)`` failover list.

        Returns:
            A ``text/event-stream`` response.

        Raises:
            HTTPException: When every backend failed to open.
        """
        attempts: list[BackendFailedError] = []
        for backend, model in candidates:
            ctx.model = model
            stream = self.transport.stream(
                backend.base_url, backend.api_key, ctx.body, headers=backend.headers
            )
            try:
                first = await stream.__anext__()
            except StopAsyncIteration:
                await stream.aclose()
                return StreamingResponse(
                    self._pump(ctx, backend, None, None),
                    media_type="text/event-stream",
                    headers=_BACKEND_HEADERS,
                )
            except (httpx.HTTPError, ValueError) as exc:
                await stream.aclose()
                failure = BackendFailedError(backend.name, exc)
                attempts.append(failure)
                logger.warning("Stream open failed for %s: %s", backend.name, exc)
                recovered = await self.hooks.run_error(ctx, failure)
                if recovered is not None:
                    return self._recovery_response(recovered)
                continue
            return StreamingResponse(
                self._pump(ctx, backend, stream, first),
                media_type="text/event-stream",
                headers=_BACKEND_HEADERS,
            )

        error = AllBackendsFailedError(attempts)
        recovered = await self.hooks.run_error(ctx, error)
        if recovered is not None:
            return self._recovery_response(recovered)
        raise _http_error(error) from error

    @staticmethod
    def _recovery_response(payload: dict[str, Any]) -> StreamingResponse:
        """Wrap an error-hook replacement payload as a one-event SSE stream.

        Args:
            payload: Replacement response returned by an error hook.

        Returns:
            A single ``data:`` event carrying the payload.
        """
        return StreamingResponse(
            iter([f"data: {json.dumps(payload)}\n\n".encode()]),
            media_type="text/event-stream",
            headers=_BACKEND_HEADERS,
        )

    async def _pump(
        self,
        ctx: HookContext,
        backend: Backend,
        stream: AsyncIterator[bytes] | None,
        first: bytes | None,
    ) -> AsyncIterator[bytes]:
        """Relay stream chunks to the client while accounting token usage.

        Args:
            ctx: Request context (fallback for token estimation).
            backend: Backend currently streaming.
            stream: Remaining backend iterator, or ``None`` for an empty body.
            first: First chunk already pulled during failover, if any.

        Yields:
            Raw bytes from the backend.
        """
        buffer = b""
        tokens: int | None = None
        try:
            if first is not None:
                yield first
                buffer, events = feed_sse(buffer, first)
                tokens = usage_tokens(events) or tokens
            if stream is not None:
                async for chunk in stream:
                    yield chunk
                    buffer, events = feed_sse(buffer, chunk)
                    found = usage_tokens(events)
                    if found is not None:
                        tokens = found
        finally:
            if stream is not None:
                await stream.aclose()
            billed = tokens if tokens is not None else ctx.estimated_tokens
            await self.router.record_usage(backend.name, billed)

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------
    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        log_level: str = "info",
        reload: bool = False,
        **uvicorn_kwargs: Any,
    ) -> None:
        """Run the proxy with uvicorn.

        Args:
            host: Bind address. Defaults to loopback; this homestead forbids
                binding ``0.0.0.0`` - front the process with a container or
                tunnel instead.
            port: Bind port.
            log_level: Uvicorn log level.
            reload: Enable uvicorn's autoreload (development only).
            **uvicorn_kwargs: Forwarded to ``uvicorn.run``.

        Example:
            proxy.serve(port=8010)
        """
        import uvicorn  # imported lazily: not needed on Cloudflare Workers

        uvicorn.run(
            self.build_app(),
            host=host,
            port=port,
            log_level=log_level,
            reload=reload,
            **uvicorn_kwargs,
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_toml(cls, path: str | Path, **overrides: Any) -> Proxy:
        """Build a proxy from a TOML configuration file.

        Args:
            path: Path to the TOML file (see ``marga.example.toml``).
            **overrides: Passed to :class:`Proxy` (``store``, ``transport``,
                ``strategy`` ...), overriding file-level choices.

        Returns:
            A configured :class:`Proxy`.

        Raises:
            FileNotFoundError: When ``path`` does not exist.
            ValueError: When the TOML structure is invalid.
        """
        from .config import proxy_from_toml

        return proxy_from_toml(path, **overrides)

    @classmethod
    def from_env(cls, **overrides: Any) -> Proxy:
        """Build a proxy from ``MARGA_CONFIG`` (TOML path) or ``MARGA_BACKENDS`` (JSON).

        Args:
            **overrides: Passed to :class:`Proxy`.

        Returns:
            A configured :class:`Proxy`.

        Raises:
            ValueError: When neither environment variable is set.
        """
        from .config import proxy_from_env

        return proxy_from_env(**overrides)


def _http_error(error: MargaError) -> HTTPException:
    """Map a marga error onto an HTTPException.

    Args:
        error: Error raised while routing or calling a backend.

    Returns:
        An ``HTTPException`` with the mapped status and structured detail.
    """
    status = http_status_for(error)
    if isinstance(error, BudgetExhaustedError):
        status = 429
    elif isinstance(error, UnknownBackendError):
        status = 400
    return HTTPException(status_code=status, detail=error_detail(error))


def _extract_tokens(response: Mapping[str, Any]) -> int:
    """Read ``usage.total_tokens`` from a backend response.

    Args:
        response: Backend response payload.

    Returns:
        Total token count, or 0 when the backend did not report usage.
    """
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        total = usage.get("total_tokens")
        if isinstance(total, int):
            return total
    return 0


def _unused(fn: Callable[..., Any]) -> None:  # pragma: no cover - typing anchor
    """Placeholder used to keep the ``Callable`` import meaningful.

    Args:
        fn: Any callable.
    """
    return None
