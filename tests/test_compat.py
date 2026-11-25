"""Tests for OpenAI/Anthropic/HuggingFace-compatible API surfaces."""

from typing import Any, AsyncIterator, Mapping

import pytest
from fastapi.testclient import TestClient

from marga import Proxy
from marga.anthropic import (
    AnthropicMessage,
    AnthropicRequest,
    anthropic_to_openai,
    openai_to_anthropic,
    translate_messages,
    translate_openai_stream,
)
from marga.transport import Transport


class StubTransport(Transport):
    """Transport returning canned responses and remembering the last request."""

    def __init__(self, *, message: dict[str, Any] | None = None) -> None:
        """Create the stub.

        Args:
            message: Assistant message to return; defaults to plain text.
        """
        super().__init__()
        self.last_body: dict[str, Any] = {}
        self.last_url = ""
        self.message = message if message is not None else {"role": "assistant", "content": "pong"}
        self.embedding_calls = 0

    async def complete(
        self,
        base_url: str,
        api_key: str = "",
        body: dict[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return a fixed completion.

        Args:
            base_url: Backend URL.
            api_key: Ignored.
            body: Request body, recorded.
            headers: Ignored.

        Returns:
            An OpenAI-compatible completion.
        """
        self.last_body = dict(body or {})
        self.last_url = base_url
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": self.last_body.get("model", "m"),
            "choices": [{"index": 0, "message": self.message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    async def embed(
        self,
        base_url: str,
        api_key: str = "",
        body: dict[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return a fixed embeddings payload.

        Args:
            base_url: Backend URL.
            api_key: Ignored.
            body: Request body, recorded.
            headers: Ignored.

        Returns:
            An OpenAI-compatible embeddings response.
        """
        self.embedding_calls += 1
        self.last_body = dict(body or {})
        self.last_url = base_url
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "model": self.last_body.get("model", "embed"),
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }


def _client(
    transport: StubTransport | None = None, **backend_extra: Any
) -> tuple[TestClient, StubTransport]:
    """Build a single-backend client.

    Args:
        transport: Transport to use.
        **backend_extra: Extra keys for the backend config.

    Returns:
        ``(client, transport)``.
    """
    transport = transport or StubTransport()
    proxy = Proxy(
        backends={
            "test": {"base_url": "http://test/v1", "models": ["test-model"], **backend_extra}
        },
        transport=transport,
    )
    return TestClient(proxy.build_app()), transport


# ---------------------------------------------------------------------------
# Anthropic: request translation
# ---------------------------------------------------------------------------


def test_anthropic_tools_are_translated():
    """Anthropic tool declarations reach the backend as OpenAI functions."""
    client, transport = _client()
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "read the file"}],
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
            "tool_choice": {"type": "any"},
        },
    )
    assert response.status_code == 200, response.text
    tools = transport.last_body["tools"]
    assert tools[0] == {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
    assert transport.last_body["tool_choice"] == "required"


def test_anthropic_system_blocks_are_flattened():
    """A block-shaped system prompt is accepted, as Anthropic allows."""
    client, transport = _client()
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 8,
            "system": [
                {"type": "text", "text": "Be brief."},
                {"type": "text", "text": "Prefer JSON."},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200, response.text
    assert transport.last_body["messages"][0] == {
        "role": "system",
        "content": "Be brief.\nPrefer JSON.",
    }


def test_anthropic_tool_result_turns_become_tool_messages():
    """A tool round-trip survives the hop to an OpenAI-shaped backend."""
    request = AnthropicRequest(
        model="m",
        max_tokens=8,
        messages=[
            AnthropicMessage(
                role="assistant",
                content=[
                    {"type": "text", "text": "reading"},
                    {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"p": "a"}},
                ],
            ),
            AnthropicMessage(
                role="user",
                content=[{"type": "tool_result", "tool_use_id": "t1", "content": "file body"}],
            ),
        ],
    )
    body = anthropic_to_openai(request)
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "read_file"
    assert body["messages"][1] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "file body",
    }


def test_anthropic_stop_sequences_map_to_stop():
    """``stop_sequences`` becomes OpenAI ``stop``."""
    client, transport = _client()
    client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 8,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert transport.last_body["stop"] == ["END"]


# ---------------------------------------------------------------------------
# Anthropic: response translation
# ---------------------------------------------------------------------------


def test_anthropic_response_carries_tool_use_blocks():
    """OpenAI ``tool_calls`` come back as Anthropic ``tool_use`` blocks."""
    transport = StubTransport(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                }
            ],
        }
    )
    client, _ = _client(transport)
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "read it"}],
            "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    tool_use = [block for block in body["content"] if block["type"] == "tool_use"]
    assert tool_use[0]["name"] == "read_file"
    assert tool_use[0]["input"] == {"path": "a.txt"}


def test_openai_to_anthropic_maps_finish_reasons():
    """Finish reasons translate to the Anthropic vocabulary."""
    translated = openai_to_anthropic(
        {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}]}, "m"
    )
    assert translated["stop_reason"] == "max_tokens"


def test_translate_messages_handles_images():
    """Base64 image blocks become OpenAI image parts."""
    messages = translate_messages(
        [
            AnthropicMessage(
                role="user",
                content=[
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                    },
                ],
            )
        ]
    )
    parts = messages[0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this?"}
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,AAAA"


# ---------------------------------------------------------------------------
# Anthropic: streaming
# ---------------------------------------------------------------------------


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    """Yield byte chunks like a network stream.

    Args:
        *parts: Chunks to yield.

    Yields:
        Each chunk.
    """
    for part in parts:
        yield part


@pytest.mark.asyncio
async def test_stream_translates_tool_calls():
    """Streaming tool calls produce Anthropic tool_use blocks."""
    stream = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        b'"function":{"name":"read_file","arguments":"{\\"pa"}}]}}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        b'"function":{"arguments":"th\\": \\"a\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    out = b"".join(
        [event async for event in translate_openai_stream(_chunks(stream), "m")]
    ).decode()

    names = [line[7:] for line in out.splitlines() if line.startswith("event: ")]
    assert "content_block_start" in names
    assert names.count("content_block_delta") == 2
    assert names[-2:] == ["message_delta", "message_stop"]
    assert '"type": "tool_use"' in out
    assert '"type": "input_json_delta"' in out
    assert '"name": "read_file"' in out
    assert '"stop_reason": "tool_use"' in out


@pytest.mark.asyncio
async def test_stream_text_then_tool_uses_increasing_block_indexes():
    """Text and tool blocks get distinct, increasing content block indexes."""
    stream = (
        b'data: {"choices":[{"delta":{"content":"thinking"}}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
        b'"function":{"name":"f","arguments":"{}"}}]}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    out = b"".join(
        [event async for event in translate_openai_stream(_chunks(stream), "m")]
    ).decode()
    starts = [line for line in out.splitlines() if '"content_block_start"' in line]
    indexes = [line.split('"index": ')[1][0] for line in starts]
    assert indexes == ["0", "1"]


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def test_count_tokens_endpoint():
    """Claude Code's startup probe returns a positive estimate."""
    client, _ = _client()
    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "x" * 400}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["input_tokens"] >= 100


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_embeddings_route_only_to_embedding_backends():
    """A chat-only backend is never asked for an embedding."""
    transport = StubTransport()
    proxy = Proxy(
        backends={
            "chat": {"base_url": "http://chat/v1", "models": ["m"]},
            "embed": {
                "base_url": "http://embed/v1",
                "models": ["embed-model"],
                "capabilities": ["embeddings"],
            },
        },
        transport=transport,
    )
    client = TestClient(proxy.build_app())

    response = client.post("/v1/embeddings", json={"model": "embed-model", "input": ["a", "b"]})
    assert response.status_code == 200, response.text
    assert transport.last_url == "http://embed/v1"
    assert response.json()["data"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_embeddings_without_backend_is_503():
    """A context with no embedding backend says so instead of 500ing."""
    client, _ = _client()
    response = client.post("/v1/embeddings", json={"model": "m", "input": "a"})
    assert response.status_code == 503
    assert response.json()["type"] == "NoBackendsForContextError"


def test_embeddings_are_accounted():
    """Embedding usage lands in the same per-backend counters."""
    transport = StubTransport()
    proxy = Proxy(
        backends={
            "embed": {
                "base_url": "http://embed/v1",
                "models": ["embed-model"],
                "capabilities": ["embeddings"],
                "cost_per_1k_tokens": 1.0,
            }
        },
        transport=transport,
    )
    client = TestClient(proxy.build_app())
    client.post("/v1/embeddings", json={"model": "embed-model", "input": "a"})
    stats = client.get("/stats?period=all").json()
    assert stats["backends"]["embed"]["calls"] == 1
    assert stats["backends"]["embed"]["tokens"] == 4


# ---------------------------------------------------------------------------
# Error envelopes
# ---------------------------------------------------------------------------


def test_openai_validation_error_envelope():
    """OpenAI clients see ``error.message`` rather than a bare ``detail``."""
    client, _ = _client()
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["type"] == "ValidationError"
    assert body["detail"]


def test_anthropic_validation_error_envelope():
    """Anthropic clients see their own error type for a bad body."""
    client, _ = _client()
    response = client.post(
        "/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "x"}]}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


def test_unknown_backend_error_is_400(monkeypatch: pytest.MonkeyPatch):
    """A routing function naming a missing backend is a client error."""
    transport = StubTransport()
    proxy = Proxy(
        backends={"test": {"base_url": "http://test/v1", "models": ["m"]}},
        transport=transport,
    )

    @proxy.route
    def bad_route(ctx):
        return "does-not-exist/model"

    client = TestClient(proxy.build_app())
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    assert response.json()["type"] == "UnknownBackendError"


# ---------------------------------------------------------------------------
# OpenAPI surface
# ---------------------------------------------------------------------------


def test_openapi_documents_every_protocol():
    """All three API families are present in the generated schema."""
    client, _ = _client()
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/embeddings",
        "/v1/models",
        "/health",
        "/stats",
    ):
        assert path in paths
    assert client.get("/docs").status_code == 200
