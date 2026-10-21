"""Single source of truth for the package version.

Kept in its own module so :mod:`marga.transport` can build a User-Agent
header without importing the package ``__init__`` (which would be circular),
and so the build backend can read the version without importing Python.
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = ["__version__"]
