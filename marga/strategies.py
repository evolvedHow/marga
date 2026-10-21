"""Built-in routing strategies.

A strategy turns a :class:`marga.hooks.HookContext` into a ``"backend/model"``
target string. Register one with ``proxy.route(strategy)`` or write your own
``@proxy.route`` function - both are called with the same context.

Strategies are ordinary objects, so they can hold state (round-robin cursors)
and read the usage store (budget limits) without global variables.

Available out of the box:

* :func:`least_used` - default; fewest total calls wins.
* :func:`round_robin` - cycles through backends in order.
* :func:`cheapest` - lowest configured ``cost_per_1k_tokens`` first.
* :func:`budget` - skips backends whose free-tier/token/$$ budget is spent,
  honouring a reporting period (``month``, ``day`` or ``all``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from .errors import BudgetExhaustedError, UnknownBackendError
from .hooks import HookContext
from .store import UsageSnapshot, usage_key

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .router import Router

    RouteTarget = "Strategy | Callable[[HookContext], Any]"


@runtime_checkable
class Strategy(Protocol):
    """Contract every routing strategy implements."""

    async def select(self, ctx: HookContext, router: Router) -> str:
        """Choose a ``"backend/model"`` target for this request.

        Args:
            ctx: The request context, including the estimated prompt size.
            router: The live router, exposing backends and the usage store.

        Returns:
            Target string in ``"backend/model"`` form. A bare ``"model"``
            string is also accepted; the router then picks a backend serving
            that model.

        Raises:
            BudgetExhaustedError: When no backend has budget left and the
                strategy has no fallback.
        """
        ...


def _is_enabled(router: Router, name: str) -> bool:
    """Check whether a backend exists and is enabled.

    Args:
        router: Router holding the backend table.
        name: Backend name to check.

    Returns:
        ``True`` when the backend is configured and not disabled.
    """
    backend = router.backends.get(name)
    return backend is not None and backend.enabled


def _target_for(router: Router, name: str, ctx: HookContext) -> str:
    """Build a target string for a backend, resolving the model.

    Args:
        router: Router used for model resolution.
        name: Backend name.
        ctx: Request context carrying the requested model.

    Returns:
        ``"backend/model"`` string.

    Raises:
        UnknownBackendError: If ``name`` is not configured.
    """
    backend = router.backends.get(name)
    if backend is None:
        raise UnknownBackendError(name, list(router.backends))
    return f"{name}/{router.resolve_model(backend, ctx.model)}"


@dataclass
class LeastUsed:
    """Select the backend with the fewest recorded calls.

    Ties are broken by backend declaration order, which makes the choice
    deterministic for a fresh store.
    """

    async def select(self, ctx: HookContext, router: Router) -> str:
        """Pick the enabled backend with the lowest all-time call count.

        Args:
            ctx: Request context (used to resolve the model).
            router: Router exposing backends and the usage store.

        Returns:
            ``"backend/model"`` string for the least-used backend.
        """
        names = [name for name in router.backends if _is_enabled(router, name)]
        if not names:
            raise UnknownBackendError("(none)", list(router.backends))
        snapshots = await router.store.get_many([usage_key(name) for name in names])
        chosen = names[0]
        fewest = snapshots[usage_key(chosen)].calls
        for name in names[1:]:
            calls = snapshots[usage_key(name)].calls
            if calls < fewest:
                chosen, fewest = name, calls
        return _target_for(router, chosen, ctx)


@dataclass
class RoundRobin:
    """Cycle through backends in declaration (or explicit) order.

    The cursor lives on the instance, so it advances per process. On
    Cloudflare each Worker isolate has its own cursor - fine for load
    spreading, but use :func:`budget` when the decision must be exact.

    Attributes:
        backends: Backend names to cycle through. ``None`` uses every enabled
            backend in declaration order.
    """

    backends: Sequence[str] | None = None

    def __post_init__(self) -> None:
        """Initialise the cursor."""
        self._index = 0

    async def select(self, ctx: HookContext, router: Router) -> str:
        """Return the next backend in the cycle.

        Args:
            ctx: Request context (used to resolve the model).
            router: Router exposing backends and the usage store.

        Returns:
            ``"backend/model"`` string for the next backend in rotation.
        """
        names = [name for name in (self.backends or router.backends) if _is_enabled(router, name)]
        if not names:
            raise UnknownBackendError("(none)", list(router.backends))
        name = names[self._index % len(names)]
        self._index += 1
        return _target_for(router, name, ctx)


@dataclass
class Cheapest:
    """Prefer the backend with the lowest ``cost_per_1k_tokens``.

    Backends without a configured rate sort as free. Ties fall back to the
    least-used backend so traffic still spreads.
    """

    async def select(self, ctx: HookContext, router: Router) -> str:
        """Pick the lowest-rate enabled backend.

        Args:
            ctx: Request context (used to resolve the model).
            router: Router exposing backends and the usage store.

        Returns:
            ``"backend/model"`` string for the cheapest backend.
        """
        names = [name for name in router.backends if _is_enabled(router, name)]
        if not names:
            raise UnknownBackendError("(none)", list(router.backends))
        rates = {name: router.backends[name].cost_per_1k_tokens for name in names}
        best_rate = min(rates.values())
        tied = [name for name in names if rates[name] == best_rate]
        if len(tied) == 1:
            return _target_for(router, tied[0], ctx)
        least_used = LeastUsed()
        selected = await least_used.select(ctx, router)
        if selected.split("/", 1)[0] in tied:
            return selected
        return _target_for(router, tied[0], ctx)


@dataclass
class Budget:
    """Spend ceiling for one backend over a reporting period.

    Attributes:
        backend: Backend the ceiling applies to.
        max_tokens: Tokens allowed per period. ``None`` means unlimited.
        max_cost: USD allowed per period. ``None`` means unlimited. Cost is
            derived from the backend's ``cost_per_1k_tokens``.
    """

    backend: str
    max_tokens: int | None = None
    max_cost: float | None = None

    def exhausted(self, snapshot: UsageSnapshot, headroom: int = 0) -> str | None:
        """Report whether this rule blocks further traffic.

        Args:
            snapshot: Usage already recorded for the period.
            headroom: Tokens the pending request is estimated to need.

        Returns:
            A human-readable reason when the rule blocks, else ``None``.
        """
        if self.max_tokens is not None:
            if snapshot.tokens >= self.max_tokens:
                return f"{self.backend}: {snapshot.tokens}/{self.max_tokens} tokens used"
            if headroom and snapshot.tokens + headroom > self.max_tokens:
                return (
                    f"{self.backend}: request of ~{headroom} tokens would exceed "
                    f"{snapshot.tokens}/{self.max_tokens}"
                )
        if self.max_cost is not None and snapshot.cost >= self.max_cost:
            return f"{self.backend}: ${snapshot.cost:.4f}/${self.max_cost:.4f} spent"
        return None

    def tokens_remaining(self, snapshot: UsageSnapshot) -> int:
        """Tokens left under this rule.

        Args:
            snapshot: Usage already recorded for the period.

        Returns:
            Remaining token allowance, or ``2**63 - 1`` when unlimited.
        """
        if self.max_tokens is None:
            return 2**63 - 1
        return max(0, self.max_tokens - snapshot.tokens)


@dataclass
class BudgetStrategy:
    """Route only to backends that still have budget.

    Rules are evaluated in list order, so the first entry with headroom wins -
    write cheapest/most-generous providers first.

    Attributes:
        rules: Per-backend ceilings.
        fallback: Strategy consulted when every rule blocks. When ``None``
            (the default) a :class:`~marga.errors.BudgetExhaustedError` is
            raised, which the proxy maps to HTTP 429.
        period: Reporting window - ``"month"``, ``"day"`` or ``"all"``.
    """

    rules: Sequence[Budget]
    fallback: "RouteTarget | None" = None
    period: str = "month"

    def __post_init__(self) -> None:
        """Validate the reporting period."""
        if self.period not in {"month", "day", "all"}:
            raise ValueError(f"period must be 'month', 'day' or 'all', got {self.period!r}")

    async def select(self, ctx: HookContext, router: Router) -> str:
        """Pick the first backend with sufficient remaining budget.

        Args:
            ctx: Request context; ``estimated_tokens`` is checked against the
                remaining token allowance before the request is sent.
            router: Router exposing backends and the usage store.

        Returns:
            ``"backend/model"`` string for the first backend with headroom,
            or the fallback strategy's answer.

        Raises:
            BudgetExhaustedError: When every rule blocks and no fallback is set.
        """
        keys = [usage_key(rule.backend, self.period) for rule in self.rules]
        snapshots = await router.store.get_many(keys)

        for rule in self.rules:
            backend = router.backends.get(rule.backend)
            if backend is None:
                continue
            if not backend.enabled:
                continue
            snapshot = snapshots[usage_key(rule.backend, self.period)]
            reason = rule.exhausted(snapshot, ctx.estimated_tokens)
            if reason is None:
                return _target_for(router, rule.backend, ctx)

        if self.fallback is not None:
            select = getattr(self.fallback, "select", None)
            if select is not None:
                return await select(ctx, router)
            return str(self.fallback(ctx))

        reasons = []
        for rule in self.rules:
            snapshot = snapshots[usage_key(rule.backend, self.period)]
            reason = rule.exhausted(snapshot, ctx.estimated_tokens)
            if reason is not None:
                reasons.append(reason)
        raise BudgetExhaustedError(
            [rule.backend for rule in self.rules],
            "; ".join(reasons),
        )


def least_used() -> LeastUsed:
    """Build the default least-used strategy.

    Returns:
        A :class:`LeastUsed` instance ready for ``proxy.route(...)``.
    """
    return LeastUsed()


def round_robin(backends: Sequence[str] | None = None) -> RoundRobin:
    """Build a round-robin strategy.

    Args:
        backends: Explicit rotation order, or ``None`` for declaration order.

    Returns:
        A :class:`RoundRobin` instance ready for ``proxy.route(...)``.
    """
    return RoundRobin(backends=backends)


def cheapest() -> Cheapest:
    """Build a lowest-cost-first strategy.

    Returns:
        A :class:`Cheapest` instance ready for ``proxy.route(...)``.
    """
    return Cheapest()


def budget(
    rules: Sequence[Budget],
    *,
    fallback: "RouteTarget | None" = None,
    period: str = "month",
) -> BudgetStrategy:
    """Build a budget-aware strategy.

    Args:
        rules: Per-backend ceilings, evaluated in order.
        fallback: Strategy used when every rule blocks (e.g. ``round_robin()``).
        period: ``"month"``, ``"day"`` or ``"all"``.

    Returns:
        A :class:`BudgetStrategy` instance ready for ``proxy.route(...)``.

    Example:
        proxy.route(budget([
            Budget(backend="groq", max_tokens=1_000_000),
            Budget(backend="openai", max_cost=5.00),
        ], period="month"))
    """
    return BudgetStrategy(rules=list(rules), fallback=fallback, period=period)
