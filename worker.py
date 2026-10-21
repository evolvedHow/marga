"""Cloudflare Worker entrypoint for marga.

FastAPI runs natively on Cloudflare Python Workers via the workers.asgi connector.
The same application code runs both locally and on the edge with no rewrite needed.

Deploy: uv run pywrangler deploy
Dev: uv run pywrangler dev
"""

from __future__ import annotations

# Pyodide httpx patch - must be first import
# This routes httpx.AsyncClient over JS fetch instead of native sockets
try:
    import pyodide.http  # type: ignore[import-not-found]

    pyodide.http.patch_all()
except ImportError:
    # Not in pyodide environment (local dev) - no patch needed
    pass

from workers import asgi  # type: ignore[import-not-found]

from marga import Proxy

# Build the FastAPI app from environment
# MARGA_BACKENDS (JSON) or MARGA_CONFIG (TOML path) must be set
# Vault credentials: VAULT_URL, VAULT_TOKEN (wrangler secrets)
# Auth token: MARGA_AUTH_TOKEN (wrangler secret, optional)
proxy = Proxy.from_env()
app = proxy.build_app()

# Export as Cloudflare Worker default export
# The workers.asgi connector runs FastAPI natively
Default = asgi.entrypoint(app)
