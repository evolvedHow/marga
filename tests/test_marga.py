"""Tests for marga."""

from typing import Any, Mapping

import pytest
from fastapi.testclient import TestClient

from marga import HookContext, Proxy
from marga.anthropic import translate_openai_stream
from marga.transport import Transport


class StubTransport(Transport):
    """Transport returning a canned OpenAI-style completion, never touching the network."""

    def __init__(self) -> None:
        """Create the stub and remember the last request body."""
        super().__init__()
        self.last_body: dict[str, Any] = {}
        self.last_base_url = ""

    async def complete(
        self,
        base_url: str,
        api_key: str = "",
        body: dict[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return a fixed completion and record the request body.

        Args:
            base_url: Ignored.
            api_key: Ignored.
            body: Request body to record.
            headers: Ignored.

        Returns:
            An OpenAI-compatible completion payload.
        """
        self.last_body = dict(body or {})
        self.last_base_url = base_url
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": self.last_body.get("model", "test-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }


def _client() -> tuple[TestClient, StubTransport]:
    """Build a TestClient over a proxy with a stub backend.

    Returns:
        The client and the stub transport used by the proxy.
    """
    transport = StubTransport()
    proxy = Proxy(
        backends={"test": {"base_url": "http://test/v1", "models": ["test-model"]}},
        transport=transport,
    )
    return TestClient(proxy.build_app()), transport


@pytest.mark.asyncio
async def test_basic_routing():
    """Test basic router initialization and routing."""
    proxy = Proxy(
        backends={"test": {"base_url": "http://localhost:8080/v1", "models": ["test-model"]}}
    )

    ctx = HookContext(messages=[], model="test-model")
    backend, model = await proxy.router.route(ctx)

    assert backend.name == "test"
    assert model == "test-model"


@pytest.mark.asyncio
async def test_custom_router():
    """Test custom routing function."""
    proxy = Proxy(
        backends={
            "fast": {"base_url": "http://fast/v1", "models": ["fast-model"]},
            "slow": {"base_url": "http://slow/v1", "models": ["slow-model"]},
        }
    )

    @proxy.route
    async def custom(ctx):
        if ctx.estimated_tokens < 100:
            return "fast/fast-model"
        return "slow/slow-model"

    # Small request
    ctx = HookContext(messages=[{"role": "user", "content": "hi"}], model="")
    backend, model = await proxy.router.route(ctx)
    assert backend.name == "fast"

    # Large request
    long_msg = "x" * 1000
    ctx = HookContext(messages=[{"role": "user", "content": long_msg}], model="")
    backend, model = await proxy.router.route(ctx)
    assert backend.name == "slow"


@pytest.mark.asyncio
async def test_before_hooks():
    """Test before hooks modify context."""
    proxy = Proxy(backends={"test": {"base_url": "http://test/v1"}})

    @proxy.before
    async def add_system(ctx):
        ctx.messages.insert(0, {"role": "system", "content": "Be helpful"})

    ctx = HookContext(messages=[{"role": "user", "content": "hello"}], model="test")
    await proxy.hooks.run_before(ctx)

    assert len(ctx.messages) == 2
    assert ctx.messages[0]["role"] == "system"


@pytest.mark.asyncio
async def test_after_hooks():
    """Test after hooks modify response."""
    proxy = Proxy(backends={"test": {"base_url": "http://test/v1"}})

    @proxy.after
    async def add_metadata(ctx, response):
        response["custom_field"] = "added"
        return response

    ctx = HookContext(messages=[], model="test")
    response = {"choices": []}
    result = await proxy.hooks.run_after(ctx, response)

    assert "custom_field" in result
    assert result["custom_field"] == "added"


@pytest.mark.asyncio
async def test_usage_tracking():
    """Test usage statistics tracking."""
    proxy = Proxy(
        backends={
            "backend1": {"base_url": "http://b1/v1"},
            "backend2": {"base_url": "http://b2/v1"},
        }
    )

    await proxy.router.record_usage("backend1", tokens=100)
    await proxy.router.record_usage("backend1", tokens=200)
    await proxy.router.record_usage("backend2", tokens=50)

    stats = await proxy.router.get_stats()

    assert stats["backend1"]["calls"] == 2
    assert stats["backend1"]["tokens"] == 300
    assert stats["backend2"]["calls"] == 1
    assert stats["backend2"]["tokens"] == 50


@pytest.mark.asyncio
async def test_least_used_routing():
    """Test least-used backend selection."""
    from marga import least_used

    proxy = Proxy(
        backends={
            "backend1": {"base_url": "http://b1/v1", "models": ["m1"]},
            "backend2": {"base_url": "http://b2/v1", "models": ["m2"]},
        }
    )
    proxy.route(least_used())

    # Record usage on backend1
    await proxy.router.record_usage("backend1", tokens=100)

    # Should route to backend2 (least used)
    ctx = HookContext(messages=[], model="")
    backend, model = await proxy.router.route(ctx)

    assert backend.name == "backend2"


def test_token_estimation():
    """Test token estimation."""
    messages = [
        {"role": "user", "content": "a" * 400},  # ~100 tokens
        {"role": "assistant", "content": "b" * 800},  # ~200 tokens
    ]

    ctx = HookContext(messages=messages, model="test")
    assert ctx.estimated_tokens == 300  # (400 + 800) / 4


def test_openapi_schema_builds():
    """/openapi.json must render and document every public route."""
    client, _ = _client()
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for path in ("/health", "/v1/models", "/v1/chat/completions", "/v1/messages"):
        assert path in paths
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200


def test_chat_completions_endpoint():
    """OpenAI-style endpoint routes to the backend and returns its completion."""
    client, transport = _client()
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "pong"
    assert transport.last_body["messages"][0]["content"] == "ping"


def test_chat_completions_validation():
    """Invalid bodies are rejected with 422."""
    client, _ = _client()
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 422
    bad_temp = {"messages": [{"role": "user", "content": "x"}], "temperature": 9}
    assert client.post("/v1/chat/completions", json=bad_temp).status_code == 422


def test_anthropic_messages_endpoint():
    """/v1/messages parses the JSON body and translates both directions."""
    client, transport = _client()
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "system": "Be brief",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["type"] == "message"
    assert data["content"][0]["text"] == "pong"
    assert data["usage"] == {"input_tokens": 3, "output_tokens": 1}
    assert transport.last_body["messages"][0] == {"role": "system", "content": "Be brief"}


def test_anthropic_messages_validation():
    """/v1/messages requires max_tokens."""
    client, _ = _client()
    response = client.post(
        "/v1/messages",
        json={"model": "test-model", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 422


def _two_backend_client() -> tuple[TestClient, StubTransport]:
    """Build a TestClient over a proxy with two declared backends.

    Returns:
        The client and the stub transport used by the proxy.
    """
    transport = StubTransport()
    proxy = Proxy(
        backends={
            "local": {"base_url": "http://local/v1", "models": ["local-model"]},
            "cloud": {"base_url": "http://cloud/v1", "models": ["cloud-model"]},
        },
        transport=transport,
    )
    return TestClient(proxy.build_app()), transport


@pytest.mark.parametrize(
    ("requested", "base_url", "sent_model"),
    [
        ("cloud/cloud-model", "http://cloud/v1", "cloud-model"),
        ("cloud-model", "http://cloud/v1", "cloud-model"),
        ("local/local-model", "http://local/v1", "local-model"),
        ("local-model", "http://local/v1", "local-model"),
    ],
)
def test_explicit_model_is_honoured(requested: str, base_url: str, sent_model: str):
    """A requested model pins the backend that serves it, however usage is spread."""
    client, transport = _two_backend_client()
    for _ in range(3):
        response = client.post(
            "/v1/chat/completions",
            json={"model": requested, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        assert transport.last_base_url == base_url
        assert transport.last_body["model"] == sent_model


def test_unknown_model_falls_back_to_balancing():
    """An undeclared model id still gets served by some backend."""
    client, transport = _two_backend_client()
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert transport.last_base_url in {"http://local/v1", "http://cloud/v1"}


async def _chunks(*parts: bytes):
    """Yield byte chunks like a network stream would.

    Args:
        *parts: Chunks to yield in order.

    Yields:
        Each chunk unchanged.
    """
    for part in parts:
        yield part


@pytest.mark.asyncio
async def test_anthropic_stream_translation_handles_batched_and_split_chunks():
    """Events batched in one chunk or split across chunks must not be lost."""
    first = (
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"po'
    )
    second = b'ng"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    third = (
        b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        b"data: [DONE]\n\n"
    )
    out = b"".join(
        [event async for event in translate_openai_stream(_chunks(first, second, third), "m")]
    ).decode()

    names = [line[7:] for line in out.splitlines() if line.startswith("event: ")]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert '"text": "pong"' in out
    assert '"stop_reason": "end_turn"' in out
    assert '"output_tokens": 2' in out
    assert '"type": "message_delta"' in out


def test_anthropic_messages_streaming_endpoint():
    """/v1/messages with stream=true returns Anthropic SSE through the full app."""

    class StreamStub(StubTransport):
        """Stub whose stream yields one batched chunk."""

        async def stream(self, base_url, api_key="", body=None, *, headers=None):
            """Yield a single chunk holding every OpenAI event.

            Args:
                base_url: Ignored.
                api_key: Ignored.
                body: Ignored.
                headers: Ignored.

            Yields:
                One chunk with two events and the DONE sentinel.
            """
            yield (
                b'data: {"choices":[{"delta":{"content":"pong"}}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            )

    proxy = Proxy(
        backends={"test": {"base_url": "http://test/v1", "models": ["test-model"]}},
        transport=StreamStub(),
    )
    client = TestClient(proxy.build_app())
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 8,
            "stream": True,
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: content_block_delta" in response.text
    assert '"text": "pong"' in response.text
    assert response.text.rstrip().endswith('data: {"type": "message_stop"}')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
