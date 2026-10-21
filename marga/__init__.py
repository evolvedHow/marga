"""marga: Ultra-minimal LLM routing proxy.

Sanskrit: मार्ग (mārga) - "path" or "route"

A transparent passthrough layer with smart routing and prompt hooks.
"""

from ._version import __version__
from .hooks import HookContext
from .proxy import Proxy
from .router import Backend, Router
from .strategies import (
    budget,
    cheapest,
    least_used,
    round_robin,
)

__all__ = [
    "Proxy",
    "Router",
    "Backend",
    "HookContext",
    "least_used",
    "round_robin",
    "cheapest",
    "budget",
    "__version__",
]
