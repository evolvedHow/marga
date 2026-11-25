"""marga: Ultra-minimal LLM routing proxy.

Sanskrit: मार्ग (mārga) - "path" or "route"

A transparent passthrough layer with smart routing, prompt hooks and per-caller
credential contexts. Speaks the OpenAI, Anthropic and embeddings APIs.
"""

from ._version import __version__
from .contexts import (
    CONTEXT_HEADER,
    CONTEXT_PATH_PREFIX,
    ContextResolver,
    ContextSpec,
)
from .hooks import HookContext
from .proxy import Proxy
from .router import Backend, Router
from .store import InMemoryStore, ScopedStore, UsageStore
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
    "ContextResolver",
    "ContextSpec",
    "CONTEXT_HEADER",
    "CONTEXT_PATH_PREFIX",
    "UsageStore",
    "InMemoryStore",
    "ScopedStore",
    "least_used",
    "round_robin",
    "cheapest",
    "budget",
    "__version__",
]
