"""Configuration loaders for ``Proxy.from_toml()`` and ``Proxy.from_env()``.

Parse TOML files or environment variables into a :class:`~marga.proxy.Proxy`
instance with backends, routing strategy, and hooks configured.

Vault credentials are loaded from:
1. Environment variables (VAULT_URL, VAULT_TOKEN)
2. ~/docker/stack/.env file (local dev standard)
3. Cloudflare Worker environment (wrangler secrets for deployed workers)
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any


def _load_vault_env() -> None:
    """Load vault credentials from ~/docker/stack/.env if not already set.

    This is the standard location for vault credentials per homelab rules.
    Does nothing if VAULT_URL and VAULT_TOKEN are already in the environment.
    """
    if os.getenv("VAULT_URL") and os.getenv("VAULT_TOKEN"):
        return  # Already configured

    stack_env = Path.home() / "docker" / "stack" / ".env"
    if not stack_env.exists():
        return

    # Simple env file parser (key=value lines, ignoring comments and empty lines)
    with stack_env.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key in ("VAULT_URL", "VAULT_TOKEN") and not os.getenv(key):
                    os.environ[key] = value


def proxy_from_toml(path: str | Path, **overrides: Any) -> "Proxy":  # noqa: F821
    """Build a Proxy from a TOML configuration file.

    Loads vault credentials from environment or ~/docker/stack/.env if available.

    Recognised top-level keys: ``name``, ``vault_context``, ``default_context``,
    ``[backends.*]`` and ``[contexts.*]``.

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

    # Load vault credentials from standard location
    _load_vault_env()

    # Extract optional proxy-level settings
    name = config.get("name", overrides.pop("name", "marga"))
    vault_context = config.get("vault_context", name)
    contexts = config.get("contexts", {})
    default_context = config.get("default_context", "")

    overrides.setdefault("contexts", contexts)
    overrides.setdefault("default_context", default_context or None)

    return Proxy(name=name, backends=backends, vault_context=vault_context, **overrides)


def proxy_from_env(**overrides: Any) -> "Proxy":  # noqa: F821
    """Build a Proxy from environment variables.

    Checks for:
    - ``MARGA_CONFIG``: path to a TOML file (delegates to proxy_from_toml)
    - ``MARGA_BACKENDS``: JSON object mapping backend names to config dicts
    - ``MARGA_CONTEXTS``: JSON object parsed like the TOML ``[contexts]`` table
    - ``MARGA_DEFAULT_CONTEXT``: context served when no key is presented

    Vault credentials are loaded from VAULT_URL/VAULT_TOKEN environment variables
    or ~/docker/stack/.env if not already set.

    Args:
        **overrides: Additional kwargs passed to the Proxy constructor.

    Returns:
        Configured Proxy instance.

    Raises:
        ValueError: When neither MARGA_CONFIG nor MARGA_BACKENDS is set, or a
            JSON variable is malformed.
    """
    from .proxy import Proxy

    # Load vault credentials from standard location
    _load_vault_env()

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
        vault_context = os.getenv("MARGA_VAULT_CONTEXT", name)
        contexts = _json_object_env("MARGA_CONTEXTS")
        default_context = os.getenv("MARGA_DEFAULT_CONTEXT", "").strip() or None
        overrides.setdefault("contexts", contexts)
        overrides.setdefault("default_context", default_context)
        return Proxy(name=name, backends=backends, vault_context=vault_context, **overrides)

    raise ValueError(
        "No configuration found. Set MARGA_CONFIG (TOML path) or MARGA_BACKENDS (JSON object)"
    )


def _json_object_env(var: str) -> dict[str, Any]:
    """Read a JSON object from an environment variable.

    Args:
        var: Environment variable name.

    Returns:
        The parsed object, or an empty dict when the variable is unset.

    Raises:
        ValueError: When the variable is set but is not a JSON object.
    """
    raw = os.getenv(var, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{var} is not valid JSON: {e}") from e
    if not isinstance(value, dict):
        raise ValueError(f"{var} must be a JSON object")
    return value
