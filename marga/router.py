"""Backend table, routing decisions, and usage accounting.

The :class:`Router` owns three things: the configured backends, the
:class:`~marga.store.UsageStore` they are billed against, and the strategy
that picks a target. It exposes :meth:`Router.candidates` - an ordered
failover list - so the proxy can move a request to the next backend when one
is down, over quota, or simply slow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._util import maybe_await
from .errors import UnknownBackendError
from .hooks import HookContext
from .store import InMemoryStore, UsageStore, usage_key
from .strategies import LeastUsed

DEFAULT_MODELS_FALLBACK = "default"


@dataclass
class Backend:
    """One upstream LLM endpoint.

    Attributes:
        name: Unique key used in routing targets (``"openai"``, ``"local"``).
            Must not contain ``/``, since targets are written ``backend/model``.
        base_url: OpenAI-compatible base URL, e.g. ``https://api.openai.com/v1``.
        api_key: Bearer token. Empty for unauthenticated endpoints such as
            a local llama.cpp server. Prefer :attr:`api_key_secret` so the key
            is read from the worker-env vault instead of being stored inline.
        api_key_secret: Vault key name (``"KEY_OPENAI"``) resolved lazily by
            :meth:`marga.proxy.Proxy.resolve_secrets`.
        api_key_env: Environment variable holding the key; the fallback for
            CI or local runs without vault access.
        models: Model ids this backend serves. Empty means "any model id is
            accepted and passed through", which suits model gateways.
        cost_per_1k_tokens: Blended USD price per 1000 tokens, used for cost
            accounting and the :func:`~marga.strategies.cheapest` strategy.
        headers: Extra HTTP headers sent upstream (provider-specific auth).
        enabled: Disabled backends are skipped by every strategy.
    """

    name: str
    base_url: str
    api_key: str = ""
    api_key_secret: str = ""
    api_key_env: str = ""
    models: list[str] = field(default_factory=list)
    cost_per_1k_tokens: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def serves(self, model: str) -> bool:
        """Report whether this backend accepts ``model``.

        Args:
            model: Requested model id; may be empty.

        Returns:
            ``True`` when the backend declares no models (pass-through) or
            lists the requested model.
        """
        if not self.models:
            return True
        return bool(model) and model in self.models


class Router:
    """Turns a request context into an ordered list of backend targets.

    The default strategy is :class:`~marga.strategies.LeastUsed`. Register
    another with :meth:`set_router`, passing either a strategy object or a
    plain function that returns ``"backend/model"``.

    Example:
        router = Router({"local": Backend(name="local", base_url="http://127.0.0.1:8080/v1")})
        backend, model = await router.route(ctx)
        for candidate in await router.candidates(ctx):  # failover order
            print(candidate[0].name, candidate[1])
    """

    def __init__(
        self,
        backends: dict[str, Backend],
        *,
        store: UsageStore | None = None,
        failover: bool = True,
        strategy: Any | None = None,
    ) -> None:
        """Create a router.

        Args:
            backends: Backend table keyed by backend name.
            store: Usage accounting backend; a new :class:`InMemoryStore` is
                created when omitted.
            failover: When ``True``, :meth:`candidates` appends every other
                enabled backend after the primary choice.
            strategy: Initial strategy or route function; ``None`` selects
                the default least-used strategy.
        """
        self.backends = backends
        self.store: UsageStore = store if store is not None else InMemoryStore()
        self.failover = failover
        self._target: Any | None = strategy

    def set_router(self, target: Any | None) -> None:
        """Register the routing decision.

        Args:
            target: A strategy exposing ``async def select(ctx, router)``, a
                function ``fn(ctx) -> str``, or ``None`` to restore the
                default least-used strategy.
        """
        self._target = target

    @property
    def strategy(self) -> Any | None:
        """The registered strategy or route function (``None`` = default)."""
        return self._target

    def enabled_backends(self) -> list[Backend]:
        """List backends that are configured and not disabled.

        Returns:
            Enabled backends in declaration order.
        """
        return [backend for backend in self.backends.values() if backend.enabled]

    def resolve_model(self, backend: Backend, requested: str) -> str:
        """Decide which model id to send to ``backend``.

        Args:
            backend: Chosen backend.
            requested: Model id asked for by the client (may be empty).

        Returns:
            ``requested`` when the backend serves it or declares no model
            list; otherwise the backend's first declared model, falling back
            to :data:`DEFAULT_MODELS_FALLBACK` for an empty declaration.
        """
        if not backend.models:
            return requested or DEFAULT_MODELS_FALLBACK
        if requested and requested in backend.models:
            return requested
        return backend.models[0]

    async def route(self, ctx: HookContext) -> tuple[Backend, str]:
        """Choose the primary backend and model for a request.

        Args:
            ctx: Request context. ``ctx.metadata["backend"]`` is set to the
                chosen backend name on success.

        Returns:
            ``(backend, model)`` tuple for the primary candidate.

        Raises:
            UnknownBackendError: When the strategy names an unknown backend.
        """
        if self._target is None:
            self._target = LeastUsed()
        select = getattr(self._target, "select", None)
        if select is not None:
            target = await maybe_await(select(ctx, self))
        else:
            target = await maybe_await(self._target(ctx))
        backend, model = self._parse_target(str(target))
        ctx.metadata["backend"] = backend.name
        return backend, model

    async def candidates(self, ctx: HookContext) -> list[tuple[Backend, str]]:
        """Build the ordered failover list for a request.

        The primary target comes first (from the strategy), followed by the
        remaining enabled backends - those that serve the requested model
        first, then the rest, each group ordered by fewest calls.

        Args:
            ctx: Request context.

        Returns:
            Non-empty list of ``(backend, model)`` candidates.
        """
        primary = await self.route(ctx)
        if not self.failover:
            return [primary]

        order = await self._failover_order(
            requested=ctx.model,
            exclude={primary[0].name},
        )
        candidates = [primary]
        candidates.extend(
            (backend, self.resolve_model(backend, ctx.model)) for backend in order
        )
        return candidates

    async def _failover_order(self, requested: str, exclude: set[str]) -> list[Backend]:
        """Rank enabled backends for failover.

        Backends serving the requested model rank above the rest; inside each
        group, fewer recorded calls rank first.

        Args:
            requested: Model id the client asked for.
            exclude: Backend names already used by the primary candidate.

        Returns:
            Backends in failover order.
        """
        backends = [b for b in self.enabled_backends() if b.name not in exclude]
        if len(backends) <= 1:
            return backends
        snapshots = await self.store.get_many([usage_key(b.name) for b in backends])
        ranked = sorted(
            backends,
            key=lambda b: (
                0 if b.serves(requested) else 1,
                snapshots[usage_key(b.name)].calls,
                b.name,
            ),
        )
        return ranked

    def _parse_target(self, target: str) -> tuple[Backend, str]:
        """Resolve a target string into a backend and model.

        Accepts ``"backend/model"``, ``"backend"`` (backend's own default
        model), or a bare ``"model"`` id that some backend declares.

        Args:
            target: Target string produced by a strategy or route function.

        Returns:
            ``(backend, model)`` tuple.

        Raises:
            UnknownBackendError: When no backend matches the target.
        """
        if "/" in target:
            head, rest = target.split("/", 1)
            if head in self.backends:
                return self.backends[head], rest or DEFAULT_MODELS_FALLBACK
        elif target in self.backends:
            backend = self.backends[target]
            return backend, self.resolve_model(backend, "")

        for backend in self.backends.values():
            if target in backend.models:
                return backend, target

        head = target.split("/", 1)[0] if "/" in target else target
        raise UnknownBackendError(head, list(self.backends))

    async def record_usage(self, backend_name: str, tokens: int = 0) -> float:
        """Account one completed request against a backend.

        Three keys are written so every consumer sees the right window: the
        all-time key (``"openai"``), the calendar month
        (``"openai:2026-10"``) and the calendar day (``"openai:2026-10-19"``).

        Args:
            backend_name: Backend that served the request.
            tokens: Total tokens consumed by the request.

        Returns:
            The USD cost attributed to the request (0.0 without a rate).
        """
        backend = self.backends.get(backend_name)
        rate = backend.cost_per_1k_tokens if backend else 0.0
        cost = (tokens / 1000.0) * rate if rate else 0.0
        keys = (
            usage_key(backend_name),
            usage_key(backend_name, "month"),
            usage_key(backend_name, "day"),
        )
        for key in keys:
            await self.store.record(key, tokens=tokens, cost=cost)
        return cost

    async def get_stats(self, period: str = "all") -> dict[str, dict[str, Any]]:
        """Read usage per backend for the specified period.

        Args:
            period: Reporting window - ``"all"`` for all-time, ``"month"`` for
                current month, ``"day"`` for current day.

        Returns:
            Mapping of backend name to ``calls``/``tokens``/``cost``/
            ``last_used`` counters for every configured backend.
        """
        # Map "all" to "total" for usage_key
        key_period = "total" if period == "all" else period  # type: ignore[arg-type]

        names = list(self.backends)
        snapshots = await self.store.get_many([usage_key(name, key_period) for name in names])
        stats: dict[str, dict[str, Any]] = {}
        for name in names:
            snapshot = snapshots[usage_key(name, key_period)]
            stats[name] = {
                "calls": snapshot.calls,
                "tokens": snapshot.tokens,
                "cost": round(snapshot.cost, 6),
                "last_used": snapshot.last_used,
            }
        return stats

    async def remaining_budget(
        self, backend_name: str, period: str = "month"
    ) -> dict[str, Any]:
        """Report recorded usage for one backend and reporting period.

        Args:
            backend_name: Backend to inspect.
            period: ``"month"``, ``"day"`` or ``"all"``.

        Returns:
            Snapshot counters for the requested window.
        """
        snapshot = await self.store.get(usage_key(backend_name, period))
        return {
            "period": period,
            "calls": snapshot.calls,
            "tokens": snapshot.tokens,
            "cost": round(snapshot.cost, 6),
            "last_used": snapshot.last_used,
        }


def build_backends(config: dict[str, dict[str, Any]]) -> dict[str, Backend]:
    """Convert plain config dicts into :class:`Backend` objects.

    Args:
        config: Mapping of backend name to its configuration dict. Recognised
            keys: ``base_url`` (required), ``api_key``, ``api_key_secret``,
            ``api_key_env``, ``models``, ``cost_per_1k_tokens``, ``headers``,
            ``enabled``.

    Returns:
        Backend table keyed by name.

    Raises:
        ValueError: When ``base_url`` is missing, the name contains ``/``,
            or the config maps no backends at all.
    """
    backends: dict[str, Backend] = {}
    for name, cfg in config.items():
        if "base_url" not in cfg:
            raise ValueError(f"Backend {name!r} is missing 'base_url'")
        if "/" in name:
            raise ValueError(f"Backend name {name!r} must not contain '/'")
        backends[name] = Backend(
            name=name,
            base_url=str(cfg["base_url"]),
            api_key=str(cfg.get("api_key", "")),
            api_key_secret=str(cfg.get("api_key_secret", "")),
            api_key_env=str(cfg.get("api_key_env", "")),
            models=[str(model) for model in cfg.get("models", [])],
            cost_per_1k_tokens=float(cfg.get("cost_per_1k_tokens", 0.0)),
            headers={str(k): str(v) for k, v in cfg.get("headers", {}).items()},
            enabled=bool(cfg.get("enabled", True)),
        )
    if not backends:
        raise ValueError("At least one backend must be configured")
    return backends
