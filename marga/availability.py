"""Backend availability checking with TTL caching.

Probes backends via HTTP HEAD/GET with short timeouts to determine whether
they are reachable. Results are cached with TTL to avoid overhead on every
request. In cloud runtime, loopback backends are never probed (always marked
unreachable) since localhost is meaningless in a Cloudflare Worker isolate.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from ._util import detect_runtime, is_loopback

if TYPE_CHECKING:
    from .router import Backend

# Probe timeout and cache TTL
PROBE_TIMEOUT = 2.0  # seconds
CACHE_TTL_SUCCESS = 60.0  # seconds - cache successful probes longer
CACHE_TTL_FAILURE = 10.0  # seconds - retry failed probes more often


@dataclass
class AvailabilityResult:
    """Cached availability check result.

    Attributes:
        reachable: Whether the backend responded successfully.
        checked_at: Unix timestamp of the check.
        reason: Human-readable explanation (for /health endpoint).
    """

    reachable: bool
    checked_at: float
    reason: str


class AvailabilityChecker:
    """Maintains a TTL cache of backend availability checks.

    In cloud runtime, loopback backends are never probed and always return
    unreachable. In local runtime, backends are probed via HTTP HEAD/GET
    with a short timeout, and results are cached.
    """

    def __init__(self) -> None:
        """Initialize the checker with an empty cache."""
        self._cache: dict[str, AvailabilityResult] = {}
        self._runtime = detect_runtime()

    async def check(self, backend: Backend) -> AvailabilityResult:
        """Check if a backend is reachable, using cached result when fresh.

        Args:
            backend: Backend to probe.

        Returns:
            Availability result with reachable flag and reason.
        """
        # Check cache first
        cached = self._cache.get(backend.name)
        if cached:
            age = time.time() - cached.checked_at
            ttl = CACHE_TTL_SUCCESS if cached.reachable else CACHE_TTL_FAILURE
            if age < ttl:
                return cached

        # Perform fresh check
        result = await self._probe(backend)
        self._cache[backend.name] = result
        return result

    async def _probe(self, backend: Backend) -> AvailabilityResult:
        """Actually probe the backend.

        Args:
            backend: Backend to probe.

        Returns:
            Fresh availability result.
        """
        now = time.time()

        # Cloud runtime: loopback backends are unreachable
        if self._runtime == "cloud" and is_loopback(backend.base_url):
            return AvailabilityResult(
                reachable=False,
                checked_at=now,
                reason="loopback address unreachable in cloud runtime",
            )

        # Local runtime: probe via HTTP
        try:
            async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
                # Try HEAD first (lighter), fall back to GET
                try:
                    response = await client.head(backend.base_url)
                except httpx.HTTPStatusError:
                    # HEAD might not be supported; try GET
                    response = await client.get(backend.base_url)

                # Accept any 2xx/3xx/4xx as "reachable" (even 404 means the server is up)
                if response.status_code < 500:
                    return AvailabilityResult(
                        reachable=True, checked_at=now, reason="backend responding"
                    )
                return AvailabilityResult(
                    reachable=False,
                    checked_at=now,
                    reason=f"HTTP {response.status_code}",
                )
        except (httpx.TimeoutException, httpx.ConnectError, asyncio.TimeoutError) as e:
            return AvailabilityResult(
                reachable=False,
                checked_at=now,
                reason=f"connection failed: {type(e).__name__}",
            )
        except Exception as e:
            return AvailabilityResult(reachable=False, checked_at=now, reason=f"probe error: {e}")

    def invalidate(self, backend_name: str) -> None:
        """Remove cached availability for a backend.

        Args:
            backend_name: Backend to invalidate.
        """
        self._cache.pop(backend_name, None)

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
