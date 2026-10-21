"""Internal async helpers shared across the framework.

Kept private (leading underscore) so it does not appear in the public API
surface documented in :mod:`marga`.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any, Awaitable, Literal, TypeVar

T = TypeVar("T")

Runtime = Literal["local", "cloud"]


async def maybe_await(value: T | Awaitable[T]) -> T:
    """Await ``value`` if it is awaitable, otherwise return it unchanged.

    This lets user callbacks (routes, hooks) be either plain synchronous
    functions or ``async def`` coroutines, matching FastAPI's ergonomics.

    Args:
        value: Plain value or coroutine/awaitable.

    Returns:
        The resolved value after awaiting when necessary.
    """
    if inspect.isawaitable(value):
        return await value
    return value  # type: ignore[return-value]


async def call_maybe_async(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke ``fn`` with arguments and await the result when awaitable.

    Args:
        fn: Callable (sync or async) to invoke.
        *args: Positional arguments for ``fn``.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        The value returned by ``fn``, awaited if it was a coroutine.
    """
    return await maybe_await(fn(*args, **kwargs))


def detect_runtime() -> Runtime:
    """Detect whether marga is running locally or in the cloud.

    Detection order:
    1. ``MARGA_RUNTIME`` env var override (``"local"`` or ``"cloud"``)
    2. ``sys.platform == "emscripten"`` → cloud (Cloudflare Workers pyodide)
    3. Try importing ``js`` module → cloud (pyodide environment)
    4. Default → local

    Returns:
        ``"local"`` for native Python, ``"cloud"`` for Cloudflare Workers/pyodide.

    Example:
        >>> detect_runtime()
        'local'
        >>> os.environ["MARGA_RUNTIME"] = "cloud"
        >>> detect_runtime()
        'cloud'
    """
    # Explicit override
    override = os.getenv("MARGA_RUNTIME", "").lower()
    if override in ("local", "cloud"):
        return override  # type: ignore[return-value]

    # Pyodide platform
    if sys.platform == "emscripten":
        return "cloud"

    # Try importing the pyodide js module
    try:
        import js  # type: ignore[import-not-found] # noqa: F401

        return "cloud"
    except ImportError:
        pass

    return "local"


def is_loopback(url: str) -> bool:
    """Check if a URL points to a loopback address.

    Loopback addresses include:
    - ``127.0.0.1``, ``localhost``, ``::1`` (IPv6 loopback)
    - ``0.0.0.0`` (bind-all, unreachable from cloud)
    - ``*.local`` mDNS hostnames

    Args:
        url: Base URL to check (e.g., ``"http://127.0.0.1:8080/v1"``).

    Returns:
        True if the URL is a loopback/local-only address.

    Example:
        >>> is_loopback("http://127.0.0.1:8080/v1")
        True
        >>> is_loopback("https://api.openai.com/v1")
        False
        >>> is_loopback("http://mydevice.local/api")
        True
    """
    url_lower = url.lower()
    loopback_indicators = (
        "127.0.0.1",
        "localhost",
        "::1",
        "0.0.0.0",
        ".local",  # mDNS
    )
    return any(indicator in url_lower for indicator in loopback_indicators)
