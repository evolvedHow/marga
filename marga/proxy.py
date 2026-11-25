"""The :class:`Proxy` - middleware, routing and the HTTP surface.

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

Requests are served in a *context*: the credential scope the caller is billed
to. One API key maps to one context, and each context gets its own router,
its own resolved upstream keys and its own usage counters, so a shared proxy
never mixes two organisations' spend or fails one over onto the other.

The server exposes ``POST /v1/chat/completions``, ``POST /v1/messages``,
``POST /v1/messages/count_tokens``, ``POST /v1/embeddings``,
``GET /v1/models``, ``GET /health``, ``GET /stats`` and the FastAPI
documentation at ``/docs``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Sequence

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from ._version import __version__
from .anthropic import (
    AnthropicCountTokensRequest,
    AnthropicRequest,
    AnthropicResponse,
    anthropic_to_openai,
    openai_to_anthropic,
    translate_openai_stream,
)
from .availability import AvailabilityChecker
from .contexts import (
    BACKEND_CONTEXT_INHERIT,
    BACKEND_CONTEXT_SHARED,
    DEFAULT_CONTEXT_NAME,
    ContextResolver,
    ContextSpec,
    backend_visible,
)
from .errors import (
    AllBackendsFailedError,
    BackendFailedError,
    MargaError,
    NoBackendsForContextError,
    VaultNotConfiguredError,
    error_detail,
    error_payload_for_path,
    http_status_for,
    is_anthropic_path,
)
from .hooks import AfterHook, BeforeHook, ErrorHook, HookContext, HookManager
from .models import (
    AnthropicCountTokensResponse,
    BackendStats,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    ModelCard,
    ModelListResponse,
    ProxyInfo,
    StatsResponse,
)
from .router import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDINGS,
    Backend,
    Router,
    build_backends,
)
from .store import InMemoryStore, ScopedStore, UsageStore, usage_key
from .transport import Transport, feed_sse, usage_tokens
from .vault import VaultClient

logger = logging.getLogger("marga")

_BACKEND_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive"}

#: Pseudo-context accepted by ``GET /stats`` to aggregate every context.
ALL_CONTEXTS = "all"


class _HookRecoveryError(Exception):
    """Internal signal: an error hook produced a replacement response.

    Attributes:
        payload: The dict an error hook returned.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__("error hook produced a replacement response")


class Proxy:
    """Transparent multi-protocol routing proxy with hooks and contexts.

    Attributes:
        name: Application name, used as the FastAPI title.
        router: Router for the default context. Request handlers use the
            per-context views built by :meth:`router_for` instead.
        contexts: Resolver mapping API keys to contexts.
        default_context: Context served when no key is presented.
        hooks: The :class:`~marga.hooks.HookManager` holding before/after/error hooks.
        transport: HTTP transport used to reach backends.
        vault: Vault client; created from the environment on first need.
        vault_context: Vault context used by the default context.
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
        contexts: Mapping[str, Any] | ContextResolver | None = None,
        default_context: str | None = None,
    ) -> None:
        """Create a proxy.

        Args:
            backends: Backend configuration keyed by backend name. Each entry
                accepts ``base_url`` (required), ``api_key``,
                ``api_key_secret`` (vault key), ``api_key_env``, ``models``,
                ``cost_per_1k_tokens``, ``headers``, ``enabled``, ``context``,
                ``vault_context``, ``capabilities`` and ``advertise``.
            store: Usage accounting backend; in-memory when omitted.
            strategy: Default routing strategy or route function.
            failover: Retry the next backend when one fails, within the same
                context only.
            vault: Vault client to use for ``api_key_secret`` lookups. When
                omitted, one is created from ``VAULT_URL``/``VAULT_TOKEN`` on
                first need.
            vault_context: Vault context for the default context; defaults to
                ``MARGA_VAULT_CONTEXT`` or ``"marga"``.
            transport: Custom transport (tests, or a runtime-specific client).
            name: Application name.
            contexts: Either a :class:`~marga.contexts.ContextResolver` or a
                mapping parsed like the TOML ``[contexts]`` table.
            default_context: Context used when no key is presented.

        Raises:
            ValueError: When the backend configuration is invalid.
        """
        self.name = name
        self.vault = vault
        self.vault_context = (
            vault_context or os.environ.get("MARGA_VAULT_CONTEXT", "").strip() or "marga"
        )
        self.transport = transport or Transport()
        self.hooks = HookManager()

        self._backend_config = {key: dict(value) for key, value in backends.items()}
        self._template_backends = build_backends(self._backend_config)
        self._base_store: UsageStore = store if store is not None else InMemoryStore()

        if isinstance(contexts, ContextResolver):
            self.contexts = contexts
        else:
            self.contexts = ContextResolver.from_mapping(dict(contexts or {}))
        if not self.contexts.names:
            # Backwards compatibility: a lone MARGA_AUTH_TOKEN becomes the
            # default context's key, so single-tenant deployments keep working.
            legacy_token = os.getenv("MARGA_AUTH_TOKEN", "").strip()
            if legacy_token:
                self.contexts.register(
                    ContextSpec(
                        name=default_context or DEFAULT_CONTEXT_NAME,
                        tokens=(legacy_token,),
                        is_default=True,
                    )
                )
        if default_context and not self.contexts.is_known(default_context):
            self.contexts.register(
                ContextSpec(name=default_context, is_default=True, vault_context=self.vault_context)
            )

        self.default_context = default_context or self.contexts.default
        self._strategy = strategy
        self._failover = failover
        self._availability = AvailabilityChecker()
        self._views: dict[str, Router] = {}
        self._view_failures: dict[str, dict[str, str]] = {}
        self._resolved: set[str] = set()
        self._locks: dict[str, asyncio.Lock] = {}
        self._app: FastAPI | None = None

        # The default context's router is created up front so ``proxy.router``
        # is always usable; its keys resolve on first request.
        self.router = self._build_view(self.default_context)
        self._views[self.default_context] = self.router

    # ------------------------------------------------------------------
    # Decorators
    # ------------------------------------------------------------------
    def route(self, target: Any) -> Any:
        """Register a routing strategy or a custom routing function.

        The strategy is applied to every context view, including views built
        later, so one registration governs the whole process.

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
        self._strategy = target
        for view in self._views.values():
            view.set_router(target)
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
    # Contexts and secrets
    # ------------------------------------------------------------------
    def _build_view(self, context: str) -> Router:
        """Build an unresolved router view for one context.

        Args:
            context: Context name.

        Returns:
            Router holding clones of every backend visible from ``context``,
            sharing the usage store (scoped) and availability cache.
        """
        visible = {
            name: replace(template)
            for name, template in self._template_backends.items()
            if backend_visible(template.context, context)
        }
        view = Router(
            visible,
            store=ScopedStore(self._base_store, context),
            failover=self._failover,
            strategy=self._strategy,
            availability=self._availability,
        )
        return view

    def _lock(self, context: str) -> asyncio.Lock:
        """Return the materialisation lock for a context.

        Args:
            context: Context name.
        """
        lock = self._locks.get(context)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[context] = lock
        return lock

    def _vault_context_for_request(self, context: str) -> str:
        """Map a request context onto the vault context holding its keys.

        Args:
            context: Request context.

        Returns:
            The context's declared vault context, else the proxy's own.
        """
        return self.contexts.vault_context_for(context, self.vault_context)

    def _vault_context_for_backend(self, backend: Backend, context: str) -> str:
        """Decide which vault context a backend's key is read from.

        Args:
            backend: Backend being resolved.
            context: Context serving the request.

        Returns:
            Vault context name.

        Example:
            An ``inherit`` backend seen from the ``nonprofit`` context reads
            ``nonprofit/KEY_OPENCODE``; the same backend seen from the default
            context reads ``marga/KEY_OPENCODE``.
        """
        if backend.vault_context:
            return backend.vault_context
        if backend.context == BACKEND_CONTEXT_SHARED:
            return self.vault_context
        if backend.context in ("", BACKEND_CONTEXT_INHERIT):
            return self._vault_context_for_request(context)
        return self._vault_context_for_request(backend.context)

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

    async def _resolve_backend_secret(
        self,
        backend: Backend,
        context: str,
        *,
        refresh: bool = False,
    ) -> None:
        """Populate one backend's API key for one context.

        Args:
            backend: Backend to resolve.
            context: Context serving the request.
            refresh: Bypass the vault's in-memory cache.

        Raises:
            MargaError: When the vault lookup fails. The caller disables the
                backend rather than failing the whole context.
        """
        if backend.api_key and not backend.api_key_secret:
            return
        if backend.api_key_secret:
            vault = self._vault_required()
            backend.api_key = await vault.get(
                backend.api_key_secret,
                context=self._vault_context_for_backend(backend, context),
                refresh=refresh,
            )
        elif backend.api_key_env:
            value = os.environ.get(backend.api_key_env, "").strip()
            if value:
                backend.api_key = value

    async def _resolve_view_secrets(
        self,
        view: Router,
        context: str,
        *,
        refresh: bool = False,
    ) -> None:
        """Resolve every key in a view, disabling backends that fail.

        A missing key must not take the process down: it disables one backend,
        the failure is reported by ``/health``, and every other context keeps
        serving. Only a context with nothing left fails, and it fails alone.

        Args:
            view: Router view to populate.
            context: Context the view belongs to.
            refresh: Bypass the vault's in-memory cache.
        """
        failures: dict[str, str] = {}
        for backend in view.backends.values():
            if not backend.enabled:
                continue
            try:
                await self._resolve_backend_secret(backend, context, refresh=refresh)
            except Exception as exc:  # noqa: BLE001 - isolate per backend
                backend.enabled = False
                reason = str(exc) or type(exc).__name__
                failures[backend.name] = reason
                logger.warning(
                    "Context %r: backend %r disabled (%s)", context, backend.name, reason
                )
        self._view_failures[context] = failures
        if not any(backend.enabled for backend in view.backends.values()):
            logger.error(
                "Context %r has no usable backend: %s",
                context,
                "; ".join(f"{name}: {reason}" for name, reason in failures.items())
                or "no backends visible from this context",
            )

    async def router_for(self, context: str, *, refresh: bool = False) -> Router:
        """Return the router view for a context, resolving its keys once.

        Args:
            context: Context name; unknown names fall back to the default.
            refresh: Re-read every key for this context.

        Returns:
            The context's router.

        Raises:
            NoBackendsForContextError: When nothing is usable in that context.
        """
        target = context if self.contexts.is_known(context) else self.default_context
        target = target or self.default_context
        view = self._views.get(target)
        if view is None:
            view = self._build_view(target)
            self._views[target] = view

        if refresh:
            self._resolved.discard(target)
        if target not in self._resolved:
            async with self._lock(target):
                if target not in self._resolved:
                    await self._resolve_view_secrets(view, target, refresh=refresh)
                    self._resolved.add(target)

        if not any(backend.enabled for backend in view.backends.values()):
            raise NoBackendsForContextError(
                target, list(self._view_failures.get(target, {}).items())
            )
        return view

    async def resolve_context_tokens(self, *, refresh: bool = False) -> None:
        """Fetch each context's bearer token from the vault.

        Contexts declare ``token_secret`` so the token itself never lives in a
        config file. This runs before the first request because authentication
        happens in middleware, before any context is known.

        A token that cannot be fetched fails closed: authentication stays
        required, so the proxy refuses requests instead of opening up.

        Args:
            refresh: Bypass the vault's in-memory cache.
        """
        pending = self.contexts.token_secrets
        if not pending:
            return
        try:
            vault = self._vault_required()
        except MargaError as exc:
            logger.error(
                "Context tokens cannot be loaded (%s); every request will be rejected",
                exc,
            )
            return
        for context, secret_key in pending.items():
            try:
                token = await vault.get(
                    secret_key,
                    context=self._vault_context_for_request(context),
                    refresh=refresh,
                )
            except MargaError as exc:
                logger.error(
                    "Context %r: token secret %s unavailable (%s); "
                    "requests for this context will be rejected",
                    context,
                    secret_key,
                    exc,
                )
                continue
            self.contexts.add_token(context, token)
            logger.info("Context %r: bearer token loaded", context)

    async def resolve_secrets(self, *, refresh: bool = False, context: str | None = None) -> None:
        """Populate backend API keys from the vault or the environment.

        Resolves the default context eagerly (called by the server lifespan),
        then lazily for every other context on first use. Failures are logged
        and disable individual backends; they never abort startup.

        Args:
            refresh: Bypass the vault's in-memory cache.
            context: Context to resolve; defaults to the default context.
        """
        await self.resolve_context_tokens(refresh=refresh)
        try:
            await self.router_for(context or self.default_context, refresh=refresh)
        except NoBackendsForContextError as exc:
            logger.error("Startup: %s", exc)

    async def resolve_all_contexts(self, *, refresh: bool = False) -> None:
        """Resolve every configured context's keys.

        Args:
            refresh: Bypass the vault's in-memory cache.
        """
        names = self.contexts.names or [self.default_context]
        for name in names:
            await self.resolve_secrets(refresh=refresh, context=name)

    async def aclose(self) -> None:
        """Release HTTP clients owned by this proxy."""
        await self.transport.close()
        if self.vault is not None:
            await self.vault.close()

    def context_for_request(self, request: Request) -> str:
        """Read the context a request was assigned by the middleware.

        Args:
            request: Incoming HTTP request.

        Returns:
            The resolved context, or the default context when the middleware
            did not run (for example inside a bare unit test).
        """
        value = getattr(request.state, "marga_context", None)
        if isinstance(value, str) and value:
            return value
        return self.default_context

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    def build_app(self) -> FastAPI:
        """Build (and cache) the FastAPI application.

        Returns:
            The FastAPI app exposing the OpenAI, Anthropic and embeddings
            surfaces.
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
                "Multi-protocol LLM routing proxy. Speaks the OpenAI, Anthropic and "
                "embeddings APIs; routes each request to a backend chosen by the "
                "configured strategy, inside the caller's credential context."
            ),
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
            lifespan=lifespan,
        )

        @app.exception_handler(HTTPException)
        async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
            """Render an HTTP error in the caller's API envelope."""
            detail = exc.detail
            if isinstance(detail, Mapping):
                message = str(detail.get("message") or detail.get("detail") or detail)
                kind = str(detail.get("type") or "MargaError")
            else:
                message = str(detail)
                kind = "HTTPException"
            payload = error_payload_for_path(request.url.path, exc.status_code, message, kind=kind)
            if not is_anthropic_path(request.url.path) and isinstance(detail, Mapping):
                for key, value in detail.items():
                    payload.setdefault(key, value)
            return JSONResponse(status_code=exc.status_code, content=payload, headers=exc.headers)

        @app.exception_handler(RequestValidationError)
        async def _validation_handler(
            request: Request, exc: RequestValidationError
        ) -> JSONResponse:
            """Render a body-validation failure in the caller's API envelope."""
            message = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            return JSONResponse(
                status_code=422,
                content=error_payload_for_path(
                    request.url.path, 422, message, kind="ValidationError"
                ),
            )

        @app.exception_handler(MargaError)
        async def _marga_error_handler(request: Request, exc: MargaError) -> JSONResponse:
            """Render any uncaught marga error in the caller's API envelope."""
            status_code = http_status_for(exc)
            detail = error_detail(exc)
            payload = error_payload_for_path(
                request.url.path, status_code, str(exc), kind=type(exc).__name__
            )
            if not is_anthropic_path(request.url.path):
                for key, value in detail.items():
                    payload.setdefault(key, value)
            return JSONResponse(status_code=status_code, content=payload)

        @app.get("/", response_model=ProxyInfo, tags=["meta"], summary="Proxy metadata")
        async def index() -> ProxyInfo:
            """Describe this proxy, its contexts and its endpoints."""
            return ProxyInfo(
                name=self.name,
                version=__version__,
                docs="/docs",
                endpoints=[
                    "/v1/chat/completions",
                    "/v1/messages",
                    "/v1/messages/count_tokens",
                    "/v1/embeddings",
                    "/v1/models",
                    "/health",
                    "/stats",
                ],
                backends=sorted(self.router.backends),
                contexts=self.contexts.names,
                default_context=self.default_context,
            )

        @app.get("/health", response_model=HealthResponse, tags=["meta"], summary="Liveness probe")
        async def health() -> HealthResponse:
            """Report process liveness, runtime, and default-context backends."""
            from ._util import detect_runtime, is_loopback
            from .models import BackendHealth

            runtime = detect_runtime()
            failures = self._view_failures.get(self.default_context, {})
            backend_health = []

            for backend in self.router.backends.values():
                loopback = is_loopback(backend.base_url)

                reachable = None
                reason = failures.get(backend.name, "")
                if self.router.check_availability and self.router._availability:
                    result = await self.router._availability.check(backend)
                    reachable = result.reachable
                    reason = reason or result.reason

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
            request: Request,
            period: str = Query(
                "all",
                pattern="^(all|month|day)$",
                description="Reporting window for the counters.",
            ),
            context: str | None = Query(
                None,
                description=(
                    "Context to report. Defaults to the caller's context; "
                    f"'{ALL_CONTEXTS}' aggregates every context as 'context/backend'."
                ),
            ),
        ) -> StatsResponse:
            """Return per-backend call, token and cost counters for a context."""
            target = context or self.context_for_request(request)
            if target == ALL_CONTEXTS:
                merged: dict[str, BackendStats] = {}
                for name in self.contexts.names or [self.default_context]:
                    try:
                        view = await self.router_for(name)
                    except NoBackendsForContextError:
                        continue
                    counters = await view.get_stats(period=period)
                    for backend, values in counters.items():
                        merged[f"{name}/{backend}"] = BackendStats(**values)
                return StatsResponse(backends=merged)
            if not self.contexts.is_known(target) and target != self.default_context:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": f"Unknown context {target!r}",
                        "type": "UnknownContextError",
                    },
                )
            view = await self.router_for(target)
            return StatsResponse(backends=await view.get_stats(period=period))

        @app.get(
            "/v1/models",
            response_model=ModelListResponse,
            tags=["openai"],
            summary="List available models",
        )
        async def list_models(request: Request) -> ModelListResponse:
            """List every model exposed to the caller's context."""
            view = await self.router_for(self.context_for_request(request))
            data: list[ModelCard] = []
            for backend in view.backends_for(CAPABILITY_CHAT):
                for model in backend.advertised_models():
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
            response_description=(
                "OpenAI-compatible completion (or an SSE stream when stream=true)."
            ),
            responses={
                400: {"description": "Routing decision named an unknown backend."},
                401: {"description": "Missing or invalid API key."},
                403: {"description": "The key may not select the requested context."},
                429: {"description": "Every backend is out of budget."},
                502: {"description": "Every candidate backend failed."},
                503: {"description": "No usable backend for this context."},
            },
        )
        async def chat_completions(
            request: ChatCompletionRequest,
            http_request: Request,
        ) -> ChatCompletionResponse | StreamingResponse:
            """Route a chat completion to a backend, run hooks, account usage."""
            return await self._handle_chat(request, self.context_for_request(http_request))

        @app.post(
            "/v1/messages",
            response_model=AnthropicResponse,
            tags=["anthropic"],
            summary="Anthropic Messages API",
            response_description="Anthropic-compatible message response",
        )
        async def anthropic_messages(
            request: AnthropicRequest,
            http_request: Request,
        ) -> AnthropicResponse | StreamingResponse:
            """Serve ``/v1/messages`` by translating to and from OpenAI format."""
            context = self.context_for_request(http_request)
            openai_req = ChatCompletionRequest.model_validate(anthropic_to_openai(request))
            result = await self._handle_chat(openai_req, context)

            if request.stream:

                async def anthropic_stream() -> AsyncIterator[bytes]:
                    async for chunk in translate_openai_stream(result.body_iterator, request.model):
                        yield chunk

                return StreamingResponse(
                    anthropic_stream(),
                    media_type="text/event-stream",
                    headers=_BACKEND_HEADERS,
                )

            response_dict = result.model_dump() if hasattr(result, "model_dump") else result
            anthropic_dict = openai_to_anthropic(response_dict, request.model)
            return AnthropicResponse.model_validate(anthropic_dict)

        @app.post(
            "/v1/messages/count_tokens",
            response_model=AnthropicCountTokensResponse,
            tags=["anthropic"],
            summary="Anthropic token counting",
            response_description="Estimated prompt token count",
        )
        async def anthropic_count_tokens(
            request: AnthropicCountTokensRequest,
        ) -> AnthropicCountTokensResponse:
            """Estimate prompt tokens for a would-be Anthropic request.

            The estimate is a 4-characters-per-token approximation rather than
            the model's real tokenizer, which is what Claude Code needs it for:
            it is a size check, not billing.
            """
            body = anthropic_to_openai(
                AnthropicRequest(
                    model=request.model,
                    messages=request.messages,
                    max_tokens=1,
                    system=request.system,
                    tools=request.tools,
                )
            )
            ctx = HookContext(body=body)
            return AnthropicCountTokensResponse(input_tokens=ctx.estimated_tokens)

        @app.post(
            "/v1/embeddings",
            response_model=EmbeddingResponse,
            tags=["openai"],
            summary="Embeddings",
            response_description="OpenAI-compatible embeddings response",
            responses={
                502: {"description": "Every embedding backend failed."},
                503: {"description": "No embedding backend is configured for this context."},
            },
        )
        async def embeddings(
            request: EmbeddingRequest,
            http_request: Request,
        ) -> EmbeddingResponse:
            """Route an embeddings request to a backend that serves them."""
            return await self._handle_embeddings(request, self.context_for_request(http_request))

        from .middleware import (
            BearerAuthMiddleware,
            ContextMiddleware,
            add_cors,
            get_auth_token,
            get_cors_origins,
        )

        cors_origins = get_cors_origins()
        if cors_origins:
            add_cors(app, origins=cors_origins)

        # ContextMiddleware doubles as authentication: it requires a key when
        # any token is configured, and is the only thing that can attribute a
        # request to a context.
        if self.contexts.required:
            app.add_middleware(
                ContextMiddleware,
                resolver=self.contexts,
                loader=self.resolve_context_tokens,
            )
        else:
            auth_token = get_auth_token()
            if auth_token:
                app.add_middleware(BearerAuthMiddleware, token=auth_token)

        self._app = app
        return app

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------
    async def _handle_chat(
        self,
        request: ChatCompletionRequest,
        context: str,
    ) -> ChatCompletionResponse | StreamingResponse:
        """Execute one chat completion request end to end.

        Args:
            request: Validated request body.
            context: Context serving the request.

        Returns:
            The backend response, an error-hook replacement, or a streaming
            response when ``request.stream`` is set.

        Raises:
            HTTPException: Mapped from marga errors (400/403/429/502/503).
        """
        body = request.model_dump(exclude_unset=True, exclude_none=True)
        ctx = HookContext(body=body, stream=bool(body.get("stream", False)))
        ctx.metadata["context"] = context
        await self.hooks.run_before(ctx)

        try:
            router = await self.router_for(context)
            candidates = await router.candidates(ctx)
        except MargaError as exc:
            recovered = await self.hooks.run_error(ctx, exc)
            if recovered is not None:
                return recovered  # type: ignore[return-value]
            raise _http_error(exc) from exc

        if ctx.stream:
            return await self._streaming_response(ctx, candidates, router)

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
            await router.record_usage(backend.name, _extract_tokens(response))
            return response  # type: ignore[return-value]

        error = AllBackendsFailedError(attempts)
        recovered = await self.hooks.run_error(ctx, error)
        if recovered is not None:
            return recovered  # type: ignore[return-value]
        raise _http_error(error) from error

    async def _handle_embeddings(
        self,
        request: EmbeddingRequest,
        context: str,
    ) -> EmbeddingResponse:
        """Execute one embeddings request with failover inside one context.

        Args:
            request: Validated embeddings body.
            context: Context serving the request.

        Returns:
            The backend's embeddings response.

        Raises:
            HTTPException: 503 when the context has no embedding backend, 502
                when every one of them failed.
        """
        router = await self.router_for(context)
        candidates = router.backends_for(CAPABILITY_EMBEDDINGS)
        if not candidates:
            raise _http_error(
                NoBackendsForContextError(
                    context, [(self.default_context, "no embedding backend configured")]
                )
            )

        snapshots = await router.store.get_many([usage_key(b.name) for b in candidates])
        candidates.sort(key=lambda b: (snapshots[usage_key(b.name)].calls, b.name))

        body = request.model_dump(exclude_unset=True, exclude_none=True)
        attempts: list[BackendFailedError] = []
        for backend in candidates:
            payload = dict(body)
            payload["model"] = router.resolve_model(backend, request.model)
            try:
                response = await self.transport.embed(
                    backend.base_url, backend.api_key, payload, headers=backend.headers
                )
            except (httpx.HTTPError, ValueError) as exc:
                attempts.append(BackendFailedError(backend.name, exc))
                logger.warning("Embedding backend %s failed: %s", backend.name, exc)
                continue

            response.setdefault("model", payload["model"])
            usage = response.get("usage") or {}
            await router.record_usage(backend.name, int(usage.get("total_tokens", 0) or 0))
            return EmbeddingResponse.model_validate(response)

        raise _http_error(AllBackendsFailedError(attempts))

    async def _streaming_response(
        self,
        ctx: HookContext,
        candidates: Sequence[tuple[Backend, str]],
        router: Router,
    ) -> StreamingResponse:
        """Open the first healthy backend stream and wrap it for the client.

        The first backend is opened *before* the response starts so that
        failover still works: a backend that fails on connect is skipped while
        the client has not yet received headers. Failover never leaves the
        caller's context.

        Args:
            ctx: Request context.
            candidates: Ordered ``(backend, model)`` failover list.
            router: Router for the caller's context, used for accounting.

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
                    self._pump(ctx, router, backend, None, None),
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
                self._pump(ctx, router, backend, stream, first),
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
        router: Router,
        backend: Backend,
        stream: AsyncIterator[bytes] | None,
        first: bytes | None,
    ) -> AsyncIterator[bytes]:
        """Relay stream chunks to the client while accounting token usage.

        Args:
            ctx: Request context (fallback for token estimation).
            router: Router for the caller's context.
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
            await router.record_usage(backend.name, billed)

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------
    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8010,
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
            path: Path to the TOML file (see ``marga.toml.example``).
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
    detail = error_detail(error)
    detail["message"] = detail.get("detail", str(error))
    return HTTPException(status_code=status, detail=detail)


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
