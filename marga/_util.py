"""Internal async helpers shared across the framework.

Kept private (leading underscore) so it does not appear in the public API
surface documented in :mod:`marga`.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")


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
