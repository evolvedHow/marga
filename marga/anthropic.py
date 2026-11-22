"""Anthropic /v1/messages endpoint with translation to/from OpenAI format.

Accepts Anthropic-shaped requests and translates them to OpenAI format for
the internal routing layer, then translates responses back to Anthropic format.
Supports streaming with SSE event translation.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, Field

from .transport import feed_sse


class AnthropicMessage(BaseModel):
    """A message in Anthropic's format.

    Attributes:
        role: Message role (user or assistant).
        content: Message text or content blocks.
    """

    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


class AnthropicRequest(BaseModel):
    """Anthropic /v1/messages request.

    Attributes:
        model: Model ID.
        messages: Conversation history (no system role in messages).
        max_tokens: Maximum tokens to generate (required by Anthropic).
        system: Optional system prompt (separate from messages).
        temperature: Sampling temperature.
        stream: Enable streaming response.
    """

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = Field(..., ge=1)
    system: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool = False
    stop_sequences: list[str] | None = None


class AnthropicUsage(BaseModel):
    """Token usage in Anthropic format.

    Attributes:
        input_tokens: Tokens in the prompt.
        output_tokens: Tokens generated.
    """

    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicContent(BaseModel):
    """Content block in Anthropic response.

    Attributes:
        type: Content type (always "text" for now).
        text: Generated text.
    """

    type: Literal["text"] = "text"
    text: str = ""


class AnthropicResponse(BaseModel):
    """Anthropic /v1/messages response.

    Attributes:
        id: Completion ID.
        type: Always "message".
        role: Always "assistant".
        content: List of content blocks.
        model: Model that generated the response.
        stop_reason: Why generation stopped.
        usage: Token usage.
    """

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContent]
    model: str
    stop_reason: str | None = None
    usage: AnthropicUsage


def anthropic_to_openai(request: AnthropicRequest) -> dict[str, Any]:
    """Translate Anthropic request to OpenAI format.

    Args:
        request: Anthropic /v1/messages request.

    Returns:
        OpenAI-compatible /v1/chat/completions request dict.
    """
    messages = []

    # System prompt becomes first message
    if request.system:
        messages.append({"role": "system", "content": request.system})

    # Convert Anthropic messages
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    openai_request = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
    }

    if request.temperature is not None:
        openai_request["temperature"] = request.temperature

    if request.stop_sequences:
        openai_request["stop"] = request.stop_sequences

    return openai_request


def openai_to_anthropic(openai_response: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate OpenAI response to Anthropic format.

    Args:
        openai_response: OpenAI /v1/chat/completions response.
        model: Model ID to include in response.

    Returns:
        Anthropic /v1/messages response dict.
    """
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    content_text = message.get("content", "")

    usage = openai_response.get("usage", {})
    anthropic_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }

    return {
        "id": openai_response.get("id", "msg-unknown"),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content_text}],
        "model": model,
        "stop_reason": _map_finish_reason(choice.get("finish_reason")),
        "usage": anthropic_usage,
    }


async def translate_openai_stream(
    openai_stream: AsyncIterator[bytes], model: str
) -> AsyncIterator[bytes]:
    """Translate an OpenAI SSE stream to the Anthropic Messages SSE format.

    Network chunks are re-framed into SSE lines first, so events split across
    or batched within chunks are handled. The emitted sequence follows the
    Anthropic spec: ``message_start``, ``content_block_start``, one
    ``content_block_delta`` per text delta, ``content_block_stop``,
    ``message_delta`` (stop reason and usage) and ``message_stop``.

    Args:
        openai_stream: OpenAI SSE byte stream.
        model: Model ID for Anthropic events.

    Yields:
        Anthropic SSE event bytes.
    """
    message_id = f"msg-{id(openai_stream)}"
    input_tokens = 0
    output_tokens = 0
    content_started = False
    stop_reason = "end_turn"
    buffer = b""

    yield _anthropic_event(
        "message_start",
        {
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        },
    )

    async for chunk in openai_stream:
        buffer, events = feed_sse(buffer, chunk)
        for event in events:
            usage = event.get("usage") or {}
            input_tokens = usage.get("prompt_tokens", input_tokens)
            output_tokens = usage.get("completion_tokens", output_tokens)

            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            text = (choice.get("delta") or {}).get("content")
            if text:
                if not content_started:
                    content_started = True
                    yield _anthropic_event(
                        "content_block_start",
                        {"index": 0, "content_block": {"type": "text", "text": ""}},
                    )
                yield _anthropic_event(
                    "content_block_delta",
                    {"index": 0, "delta": {"type": "text_delta", "text": text}},
                )
            if choice.get("finish_reason"):
                stop_reason = _map_finish_reason(choice["finish_reason"])

    if content_started:
        yield _anthropic_event("content_block_stop", {"index": 0})
    yield _anthropic_event(
        "message_delta",
        {
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    )
    yield _anthropic_event("message_stop", {})


def _anthropic_event(event_type: str, data: dict[str, Any]) -> bytes:
    """Format an Anthropic SSE event.

    Args:
        event_type: Event type name; also written into the payload ``type``.
        data: Event data dict.

    Returns:
        SSE-formatted bytes.
    """
    payload = {"type": event_type, **data}
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def _map_finish_reason(openai_reason: str | None) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason.

    Args:
        openai_reason: OpenAI finish_reason value.

    Returns:
        Anthropic stop_reason value.
    """
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "stop_sequence",
        "tool_calls": "tool_use",
    }
    return mapping.get(openai_reason or "", "end_turn")
