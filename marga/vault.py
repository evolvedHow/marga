"""Read API keys and other secrets from the deployed ``worker-env`` vault.

The homestead runs one centralised secret vault as a Cloudflare Worker
(``~/codebox/worker-env``, deployed as ``worker-env-vault``). Every app reads
its credentials from there rather than keeping a ``.env`` file - the only
thing an app stores locally is the two vault credentials themselves
(``VAULT_URL`` / ``VAULT_TOKEN``).

This module is a small async client for that HTTP API. It deliberately does
not depend on the ``worker_env`` package (which is not on PyPI), so marga can
be published and bundled for Cloudflare Python Workers without pulling a
private dependency.

Wire format (see worker-env ``README.md``)::

    GET {VAULT_URL}/secrets/{context}/{key}
    Authorization: Bearer {VAULT_TOKEN}

    -> 200 {"context": "marga", "key": "KEY_OPENAI", "value": "sk-...", ...}
    -> 401 {"detail": "..."}   token rejected
    -> 404 {"detail": "..."}   key absent, even after default-context fallback

Contexts are named after the app (lowercased, joined): marga uses ``marga``.
Missing keys fall back to the shared default context server-side, so
cross-app secrets only need to exist once.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from .errors import SecretNotFoundError, VaultError, VaultNotConfiguredError

DEFAULT_CONTEXT = "marga"
CONTEXT_ENV = "MARGA_VAULT_CONTEXT"
URL_ENV = "VAULT_URL"
TOKEN_ENV = "VAULT_TOKEN"
HEALTH_PATH = "/health"
SECRETS_PATH = "/secrets"


class VaultClient:
    """Async client for the worker-env secret vault.

    Attributes:
        base_url: Deployed vault origin, e.g. ``https://worker-env-vault.example.workers.dev``.
        context: Default vault context (the app name, lowercased).

    Example:
        client = VaultClient.from_env()
        assert client is not None
        api_key = await client.get("KEY_OPENAI")
        await client.close()
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        context: str = DEFAULT_CONTEXT,
        timeout: float = 10.0,
        cache: bool = True,
        retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a vault client.

        Args:
            base_url: Vault origin; a trailing ``/`` is tolerated.
            token: Bearer token sent as ``Authorization``.
            context: Default context for lookups (``marga`` by convention).
            timeout: Per-request timeout in seconds.
            cache: Cache successful lookups in memory for the client's life.
            retries: Extra attempts for transient network/5xx failures.
            client: Optional pre-built ``httpx.AsyncClient`` (tests, or a
                caller that owns connection pooling). Not closed by
                :meth:`close` when supplied.
        """
        if not base_url:
            raise VaultNotConfiguredError([URL_ENV])
        if not token:
            raise VaultNotConfiguredError([TOKEN_ENV])
        self.base_url = base_url.rstrip("/")
        self.context = context
        self._token = token
        self._timeout = timeout
        self._retries = max(0, retries)
        self._cache_enabled = cache
        self._cache: dict[str, str] = {}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_env(
        cls,
        context: str | None = None,
        *,
        url_var: str = URL_ENV,
        token_var: str = TOKEN_ENV,
        context_var: str = CONTEXT_ENV,
        **kwargs: Any,
    ) -> VaultClient | None:
        """Build a client from environment variables.

        Args:
            context: Explicit context; falls back to ``MARGA_VAULT_CONTEXT``
                and then ``"marga"``.
            url_var: Environment variable holding the vault origin.
            token_var: Environment variable holding the bearer token.
            context_var: Environment variable holding the default context.
            **kwargs: Forwarded to :class:`VaultClient`.

        Returns:
            A configured client, or ``None`` when either credential variable
            is unset or empty.
        """
        base_url = os.environ.get(url_var, "").strip()
        token = os.environ.get(token_var, "").strip()
        if not base_url or not token:
            return None
        resolved = context or os.environ.get(context_var, "").strip() or DEFAULT_CONTEXT
        return cls(base_url, token, context=resolved, **kwargs)

    @classmethod
    def require_env(cls, context: str | None = None, **kwargs: Any) -> VaultClient:
        """Build a client from environment variables, or explain what is missing.

        Args:
            context: Explicit context override.
            **kwargs: Forwarded to :class:`VaultClient`.

        Returns:
            A configured client.

        Raises:
            VaultNotConfiguredError: When the credential variables are unset.
        """
        client = cls.from_env(context, **kwargs)
        if client is None:
            raise VaultNotConfiguredError([URL_ENV, TOKEN_ENV])
        return client

    def clear_cache(self) -> None:
        """Drop every cached secret value."""
        self._cache.clear()

    async def get(self, key: str, *, context: str | None = None, refresh: bool = False) -> str:
        """Fetch one secret value.

        Args:
            key: Secret key, e.g. ``"KEY_OPENAI"`` or ``"github-token"``.
            context: Override the client's default context.
            refresh: Bypass the in-memory cache for this call.

        Returns:
            The secret value.

        Raises:
            SecretNotFoundError: When the vault has no such key.
            VaultError: On authentication failures, malformed responses, or
                exhausted retries.
        """
        resolved = context or self.context
        cache_key = f"{resolved}/{key}"
        if self._cache_enabled and not refresh and cache_key in self._cache:
            return self._cache[cache_key]

        response = await self._request(
            "GET", f"{SECRETS_PATH}/{resolved}/{key}", not_found=(resolved, key)
        )
        payload = _json_object(response, source=cache_key)
        value = payload.get("value")
        if not isinstance(value, str) or value == "":
            raise VaultError(
                f"Vault returned no usable 'value' for {cache_key}",
                status=response.status_code,
            )
        if self._cache_enabled:
            self._cache[cache_key] = value
        return value

    async def get_optional(self, key: str, *, context: str | None = None) -> str | None:
        """Fetch a secret, treating a missing key as ``None``.

        Args:
            key: Secret key to look up.
            context: Override the client's default context.

        Returns:
            The secret value, or ``None`` when the key is absent.
        """
        try:
            return await self.get(key, context=context)
        except SecretNotFoundError:
            return None

    async def list_keys(self, context: str | None = None) -> list[str]:
        """List secret keys available in a context.

        Only key names are returned - values never leave the vault through
        this call, which makes it safe to log.

        Args:
            context: Context to list; defaults to the client's context.

        Returns:
            Sorted secret keys.
        """
        resolved = context or self.context
        response = await self._request(
            "GET", f"{SECRETS_PATH}?context={resolved}", not_found=(resolved, "*")
        )
        payload = _json_object(response, source=f"{resolved}/*")
        secrets = payload.get("secrets", [])
        keys = [
            str(item.get("key")) for item in secrets if isinstance(item, dict) and item.get("key")
        ]
        return sorted(keys)

    async def health(self) -> dict[str, Any]:
        """Probe the vault's health endpoint.

        Returns:
            The parsed health payload (typically ``{"status": "ok", ...}``).

        Raises:
            VaultError: When the vault is unreachable or unhealthy.
        """
        response = await self._request("GET", HEALTH_PATH)
        return _json_object(response, source=HEALTH_PATH)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        not_found: tuple[str, str] | None = None,
    ) -> httpx.Response:
        """Send an authenticated request with retries and error mapping.

        Args:
            method: HTTP method. Only ``GET`` is used in practice.
            path: Path beginning with ``/``.
            not_found: ``(context, key)`` used to raise a descriptive
                :class:`SecretNotFoundError` on HTTP 404.

        Returns:
            The successful HTTP response.

        Raises:
            SecretNotFoundError: On 404 when ``not_found`` is provided.
            VaultError: On 401/403, other 4xx, repeated 5xx, or transport errors.
        """
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, url, headers=headers)
            except httpx.HTTPError as exc:
                if attempt >= self._retries:
                    raise VaultError(f"Vault request failed: {exc}") from exc
                attempt += 1
                await asyncio.sleep(0.2 * attempt)
                continue

            if response.status_code in {401, 403}:
                raise VaultError(
                    "Vault rejected VAULT_TOKEN; refresh it from the worker-env deployment",
                    status=response.status_code,
                    detail=_detail(response),
                )
            if response.status_code == 404:
                if not_found is not None:
                    raise SecretNotFoundError(*not_found)
                raise VaultError(
                    f"Vault path not found: {path}",
                    status=404,
                    detail=_detail(response),
                )
            if response.status_code >= 500 and attempt < self._retries:
                attempt += 1
                await asyncio.sleep(0.2 * attempt)
                continue
            if response.status_code >= 400:
                raise VaultError(
                    f"Vault returned HTTP {response.status_code} for {path}",
                    status=response.status_code,
                    detail=_detail(response),
                )
            return response

    async def __aenter__(self) -> VaultClient:
        """Enter the client as an async context manager.

        Returns:
            This client.
        """
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the client when leaving the context manager.

        Args:
            *exc_info: Standard exception triple, ignored.
        """
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client when this instance owns it."""
        if self._owns_client:
            await self._client.aclose()


def _json_object(response: httpx.Response, *, source: str) -> dict[str, Any]:
    """Parse a JSON object response body.

    Args:
        response: Successful HTTP response.
        source: Human-readable label used in error messages.

    Returns:
        The decoded JSON object.

    Raises:
        VaultError: When the body is not a JSON object.
    """
    try:
        payload = response.json()
    except ValueError as exc:
        raise VaultError(f"Vault returned non-JSON for {source}") from exc
    if not isinstance(payload, dict):
        raise VaultError(f"Vault returned unexpected JSON for {source}")
    return payload


def _detail(response: httpx.Response) -> str:
    """Extract the vault's error detail string.

    Args:
        response: Error response from the vault.

    Returns:
        The ``detail`` field when present, else an empty string.
    """
    try:
        payload = response.json()
    except ValueError:
        return ""
    if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
        return str(payload["detail"])
    return ""
