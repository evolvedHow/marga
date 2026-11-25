"""Pluggable usage accounting.

The router records consumption through a :class:`UsageStore` instead of
keeping counters in process memory. That single seam is what lets the same
routing logic run in three very different places:

* :class:`InMemoryStore` - default, single process (local dev, Docker, Modal).
* A Cloudflare Durable Object - strongly consistent across Worker isolates.
* Redis / D1 / SQL - any store that can increment and read a counter.

Keys are opaque strings. Strategies embed the reporting period in the key
(``openai:2026-10``) so rollover is a key change, not a migration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol, Sequence, runtime_checkable

Period = Literal["total", "month", "day"]


def usage_key(backend: str, period: Period = "total") -> str:
    """Generate a usage store key for a backend and reporting period.

    Args:
        backend: Backend name.
        period: Reporting period - ``total`` for all-time, ``month`` for
            current month (``2026-10``), ``day`` for current day (``2026-10-20``).

    Returns:
        Opaque key string (e.g. ``"openai"``, ``"openai:2026-10"``).
    """
    if period == "total":
        return backend
    now = datetime.now(UTC)
    if period == "month":
        return f"{backend}:{now.year}-{now.month:02d}"
    # period == "day"
    return f"{backend}:{now.year}-{now.month:02d}-{now.day:02d}"


@dataclass
class UsageSnapshot:
    """Accumulated consumption for one store key.

    Attributes:
        calls: Number of recorded requests.
        tokens: Tokens accounted for the key.
        cost: Estimated spend in USD.
        last_used: Unix timestamp of the most recent record, 0.0 if unused.
    """

    calls: int = 0
    tokens: int = 0
    cost: float = 0.0
    last_used: float = 0.0

    def record(self, tokens: int = 0, cost: float = 0.0) -> None:
        """Fold one request into this snapshot.

        Args:
            tokens: Tokens consumed by the request.
            cost: Estimated spend in USD for the request.
        """
        self.calls += 1
        self.tokens += tokens
        self.cost += cost
        self.last_used = time.time()


@runtime_checkable
class UsageStore(Protocol):
    """Storage contract the router depends on.

    Implementations must be safe to call from an asyncio event loop; the
    Cloudflare Durable Object and Redis adapters do their I/O with ``await``.
    """

    async def record(self, key: str, tokens: int = 0, cost: float = 0.0) -> None:
        """Add one request's consumption to ``key``.

        Args:
            key: Opaque accounting key (e.g. ``"openai:2026-10"``).
            tokens: Tokens consumed.
            cost: Estimated spend in USD.
        """
        ...

    async def get(self, key: str) -> UsageSnapshot:
        """Read the snapshot for ``key``.

        Args:
            key: Opaque accounting key.

        Returns:
            The current snapshot; zero-valued when the key is unknown.
        """
        ...

    async def get_many(self, keys: Sequence[str]) -> dict[str, UsageSnapshot]:
        """Read several snapshots in one call.

        Args:
            keys: Keys to fetch. Unknown keys map to zero snapshots.

        Returns:
            Mapping of key to snapshot for every requested key.
        """
        ...

    async def reset(self, key: str) -> None:
        """Discard accumulated usage for ``key``.

        Args:
            key: Opaque accounting key to clear.
        """
        ...


class ScopedStore:
    """Prefix every accounting key with a scope, usually a context name.

    Without this, all contexts share one counter per backend: the non-profit's
    spend would land in your personal totals, and :class:`~marga.strategies.
    LeastUsed` would load-balance *across* accounts. Wrapping the store keeps
    the prefixing completely out of the strategies, which continue to see plain
    ``"openai:2026-10"`` keys.

    Example:
        scoped = ScopedStore(InMemoryStore(), "nonprofit")
        await scoped.record("opencode", tokens=1000)
        # inner store now holds "nonprofit::opencode"
    """

    def __init__(self, store: UsageStore, scope: str) -> None:
        """Wrap a store.

        Args:
            store: Underlying store holding the real counters.
            scope: Prefix applied to every key, e.g. a context name.
        """
        self.store = store
        self.scope = scope

    def _key(self, key: str) -> str:
        """Apply the scope prefix.

        Args:
            key: Unscoped accounting key.

        Returns:
            Scoped key.
        """
        return f"{self.scope}::{key}"

    async def record(self, key: str, tokens: int = 0, cost: float = 0.0) -> None:
        """Record usage under the scoped key.

        Args:
            key: Unscoped accounting key.
            tokens: Tokens consumed.
            cost: Estimated spend in USD.
        """
        await self.store.record(self._key(key), tokens=tokens, cost=cost)

    async def get(self, key: str) -> UsageSnapshot:
        """Read the snapshot for a scoped key.

        Args:
            key: Unscoped accounting key.

        Returns:
            The current snapshot, zero-valued when unknown.
        """
        return await self.store.get(self._key(key))

    async def get_many(self, keys: Sequence[str]) -> dict[str, UsageSnapshot]:
        """Read several snapshots, returning them under the unscoped keys.

        Args:
            keys: Unscoped accounting keys.

        Returns:
            Mapping of unscoped key to snapshot.
        """
        scoped = [self._key(key) for key in keys]
        resolved = await self.store.get_many(scoped)
        return {key: resolved.get(self._key(key), UsageSnapshot()) for key in keys}

    async def reset(self, key: str) -> None:
        """Discard accumulated usage for a scoped key.

        Args:
            key: Unscoped accounting key.
        """
        await self.store.reset(self._key(key))


class InMemoryStore:
    """Default :class:`UsageStore` backed by a plain dict.

    Appropriate for a single process. Counters do not survive a restart and
    are not shared between Worker isolates or replicas - use a persistent
    :class:`UsageStore` implementation when accounting must be global.

    Example:
        store = InMemoryStore()
        await store.record("openai:2026-10", tokens=1200, cost=0.002)
        snapshot = await store.get("openai:2026-10")
        print(snapshot.tokens)  # 1200
    """

    def __init__(self) -> None:
        """Initialise an empty counter table."""
        self._data: dict[str, UsageSnapshot] = {}

    async def record(self, key: str, tokens: int = 0, cost: float = 0.0) -> None:
        """Add one request's consumption to ``key``.

        Args:
            key: Opaque accounting key.
            tokens: Tokens consumed.
            cost: Estimated spend in USD.
        """
        self._data.setdefault(key, UsageSnapshot()).record(tokens=tokens, cost=cost)

    async def get(self, key: str) -> UsageSnapshot:
        """Read the snapshot for ``key``.

        Args:
            key: Opaque accounting key.

        Returns:
            The stored snapshot, or a zero-valued snapshot when unknown.
        """
        snapshot = self._data.get(key)
        if snapshot is None:
            return UsageSnapshot()
        return UsageSnapshot(
            calls=snapshot.calls,
            tokens=snapshot.tokens,
            cost=snapshot.cost,
            last_used=snapshot.last_used,
        )

    async def get_many(self, keys: Sequence[str]) -> dict[str, UsageSnapshot]:
        """Read several snapshots in one call.

        Args:
            keys: Keys to fetch.

        Returns:
            Mapping of key to snapshot for every requested key.
        """
        return {key: await self.get(key) for key in keys}

    async def reset(self, key: str) -> None:
        """Discard accumulated usage for ``key``.

        Args:
            key: Opaque accounting key to clear.
        """
        self._data.pop(key, None)

    def keys(self) -> list[str]:
        """List the keys currently tracked.

        Returns:
            Sorted list of accounting keys, useful for diagnostics.
        """
        return sorted(self._data)
