"""Configuration loaders for ``Proxy.from_toml()`` and ``Proxy.from_env()``.

Parse TOML files or environment variables into a :class:`~marga.proxy.Proxy`
instance with backends, routing strategy, and hooks configured.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any


def proxy_from_toml(path: str | Path, **overrides: Any) -> "Proxy":  # noqa: F821
    """Build a Proxy from a TOML configuration file.

    Args:
        path: Path to the TOML configuration file.
        **overrides: Additional kwargs passed to the Proxy constructor.

    Returns:
        Configured Proxy instance.

    Raises:
        FileNotFoundError: When the path does not exist.
        ValueError: When the TOML structure is invalid.
    """
    from .proxy import Proxy

    toml_path = Path(path)
    if not toml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with toml_path.open("rb") as f:
        config = tomllib.load(f)

    backends = config.get("backends")
    if not backends:
        raise ValueError("TOML must contain a [backends] section")

    # Extract optional proxy-level settings
    name = config.get("name", overrides.pop("name", "marga"))

    # Strategy can be specified as a string reference or inline config
    # For now, just pass backends and let Proxy use defaults
    return Proxy(name=name, backends=backends, **overrides)


def proxy_from_env(**overrides: Any) -> "Proxy":  # noqa: F821
    """Build a Proxy from environment variables.

    Checks for:
    - ``MARGA_CONFIG``: path to a TOML file (delegates to proxy_from_toml)
    - ``MARGA_BACKENDS``: JSON object mapping backend names to config dicts

    Args:
        **overrides: Additional kwargs passed to the Proxy constructor.

    Returns:
        Configured Proxy instance.

    Raises:
        ValueError: When neither MARGA_CONFIG nor MARGA_BACKENDS is set.
    """
    from .proxy import Proxy

    # Option 1: TOML file path
    config_path = os.getenv("MARGA_CONFIG")
    if config_path:
        return proxy_from_toml(config_path, **overrides)

    # Option 2: JSON backends
    backends_json = os.getenv("MARGA_BACKENDS")
    if backends_json:
        try:
            backends = json.loads(backends_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"MARGA_BACKENDS is not valid JSON: {e}") from e

        if not isinstance(backends, dict):
            raise ValueError("MARGA_BACKENDS must be a JSON object")

        name = os.getenv("MARGA_NAME", "marga")
        return Proxy(name=name, backends=backends, **overrides)

    raise ValueError(
        "No configuration found. Set MARGA_CONFIG (TOML path) or "
        "MARGA_BACKENDS (JSON object)"
    )
