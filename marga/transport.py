"""HTTP transport towards OpenAI-compatible backends.

One shared :class:`httpx.AsyncClient` keeps connection pooling and TLS
sessions warm across requests. Streaming responses are passed through as raw
bytes; :func:`feed_sse` parses the server-sent-event lines so callers can
recover ``usage`` from the final chunk without buffering the whole stream.

The client is deliberately provider-agnostic: any endpoint that speaks the
OpenAI chat-completions shape (llama.cpp, Groq, OpenRouter, Workers AI's
OpenAI-compatible route, ...) works unchanged.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Mapping

import httpx

from ._version import __version__

USER_AGENT = f"marga/{__version__}"


class Transport:
    """Async HTTP client for backend chat-completion calls.

    Example:
        transport = Transport(timeout=30.0)
        payload = await transport.complete(
            "https://api.openai.com/v1", api_key, {"model": "gpt-4o-mini", "messages": [...]}
        )
        await transport.close()
    """

    def __init__(
        self,
        timeout: float = 60.0,
        *,
        user_agent: str = USER_AGENT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a transport.

        Args:
            timeout: Per-request timeout in seconds.
            user_agent: ``User-Agent`` header sent upstream. Some providers
                (GitHub, a few gateways) reject requests without one.
            client: Optional pre-built client, useful in tests. Not closed by
                :meth:`close` when supplied.
        """
        self.timeout = timeout
        self.user_agent = user_agent
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    def _headers(
        self,
        api_key: str,
        extra: Mapping[str, str] | None,
    ) -> dict[str, str]:
        """Assemble request headers for a backend call.

        Args:
            api_key: Bearer token; omitted from headers when empty.
            extra: Backend-specific headers (organisation ids, gateways).

        Returns:
            Header mapping with ``Accept``, ``User-Agent`` and auth applied.
        """
        headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra:
            headers.update(dict(extra))
        return headers

    @staticmethod
    def _url(base_url: str, path: str = "/chat/completions") -> str:
        """Join a backend base URL with an endpoint path.

        Args:
            base_url: Backend base URL, with or without a trailing slash.
            path: Endpoint path beginning with ``/``.

        Returns:
            Fully qualified URL.
        """
        return f"{base_url.rstrip('/')}{path}"

    async def complete(
        self,
        base_url: str,
        api_key: str = "",
        body: dict[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send a blocking chat-completion request.

        Args:
            base_url: Backend base URL.
            api_key: Bearer token for the backend.
            body: OpenAI-format request body.
            headers: Extra backend-specific headers.

        Returns:
            The parsed JSON response.

        Raises:
            httpx.HTTPStatusError: On a 4xx/5xx backend response.
            httpx.HTTPError: On transport failures.
            ValueError: When the response body is not a JSON object.
        """
        response = await self._client.post(
            self._url(base_url),
            json=body or {},
            headers=self._headers(api_key, headers),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Backend {base_url} returned non-object JSON")
        return payload

    async def embed(
        self,
        base_url: str,
        api_key: str = "",
        body: dict[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send an OpenAI-compatible embeddings request.

        Args:
            base_url: Backend base URL.
            api_key: Bearer token for the backend.
            body: ``{"model": ..., "input": ...}`` request body.
            headers: Extra backend-specific headers.

        Returns:
            The parsed JSON response.

        Raises:
            httpx.HTTPStatusError: On a 4xx/5xx backend response.
            httpx.HTTPError: On transport failures.
            ValueError: When the response body is not a JSON object.
        """
        response = await self._client.post(
            self._url(base_url, "/embeddings"),
            json=body or {},
            headers=self._headers(api_key, headers),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Backend {base_url} returned non-object JSON")
        return payload

    async def stream(
        self,
        base_url: str,
        api_key: str = "",
        body: dict[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream a chat completion as server-sent events.

        Args:
            base_url: Backend base URL.
            api_key: Bearer token for the backend.
            body: OpenAI-format request body (``stream`` is forced to true).
            headers: Extra backend-specific headers.

        Yields:
            Raw response bytes, normally one or more ``data:`` blocks.

        Raises:
            httpx.HTTPStatusError: On a 4xx/5xx backend response.
            httpx.HTTPError: On transport failures.
        """
        request_body = dict(body or {})
        request_body["stream"] = True
        request_headers = self._headers(api_key, headers)
        request_headers["Accept"] = "text/event-stream"
        async with self._client.stream(
            "POST",
            self._url(base_url),
            json=request_body,
            headers=request_headers,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk

    async def close(self) -> None:
        """Close the underlying HTTP client when this instance owns it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Transport:
        """Enter the transport as an async context manager.

        Returns:
            This transport.
        """
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the transport when leaving the context manager.

        Args:
            *exc_info: Standard exception triple, ignored.
        """
        await self.close()


def feed_sse(buffer: bytes, chunk: bytes) -> tuple[bytes, list[dict[str, Any]]]:
    """Fold a stream chunk into the SSE line buffer and decode complete events.

    Only the trailing partial line is retained, so memory stays bounded even
    for very long completions.

    Args:
        buffer: Bytes held back from the previous call.
        chunk: Newly received bytes.

    Returns:
        ``(remaining_buffer, events)`` where ``events`` holds the decoded JSON
        objects from complete ``data:`` lines. ``[DONE]`` sentinels and
        non-JSON payloads are skipped.
    """
    data = buffer + chunk
    events: list[dict[str, Any]] = []
    while b"\n" in data:
        raw_line, data = data.split(b"\n", 1)
        line = raw_line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:") :].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            decoded = json.loads(payload)
        except ValueError:
            continue
        if isinstance(decoded, dict):
            events.append(decoded)
    return data, events


def usage_tokens(events: list[dict[str, Any]]) -> int | None:
    """Extract ``usage.total_tokens`` from decoded SSE events.

    Args:
        events: Decoded events from :func:`feed_sse`.

    Returns:
        The reported total token count, or ``None`` when the backend did not
        include usage (which is the case unless the client asked for it).
    """
    for event in reversed(events):
        usage = event.get("usage")
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if isinstance(total, int):
                return total
    return None
