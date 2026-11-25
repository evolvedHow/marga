"""Anthropic Messages API surface, translated to and from OpenAI format.

marga speaks OpenAI chat-completions to its backends and translates at the
edge, so an Anthropic client (Claude Code, the ``anthropic`` SDK, any tool that
only offers an "Anthropic" provider entry) can point at marga unchanged.

Translation covers the parts that actually change behaviour:

* ``system`` as a string *or* a list of content blocks.
* ``tools``/``tool_choice``, in both directions, including streaming tool-call
  deltas. Dropping these silently would make a tool-using client appear to
  forget every tool it declared.
* ``stop_sequences`` → ``stop``.
* ``stop_reason`` ← ``finish_reason``, with ``tool_use`` when the model called
  a tool.

Fields with no OpenAI equivalent (``top_k``, ``thinking``, ``metadata``) are
dropped with a warning instead of being forwarded, because OpenAI-compatible
backends reject unknown body fields with a 400.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from .transport import feed_sse

logger = logging.getLogger("marga.anthropic")

#: Anthropic request fields with no OpenAI equivalent.
UNSUPPORTED_FIELDS = ("top_k", "thinking", "metadata", "service_tier", "mcp_servers")


class AnthropicMessage(BaseModel):
    """A message in Anthropic's format.

    Attributes:
        role: Message role (user or assistant).
        content: Message text or content blocks (``text``, ``tool_use``,
            ``tool_result``, ``image``).
    """

    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


class AnthropicRequest(BaseModel):
    """Anthropic ``/v1/messages`` request.

    Attributes:
        model: Model ID.
        messages: Conversation history (no system role in messages).
        max_tokens: Maximum tokens to generate (required by Anthropic).
        system: System prompt, as text or as content blocks.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        stream: Enable streaming response.
        stop_sequences: Strings that stop generation.
        tools: Tool declarations in Anthropic shape (``input_schema``).
        tool_choice: How the model should choose a tool.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = Field(..., ge=1)
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic ``/v1/messages/count_tokens`` request.

    ``max_tokens`` is absent here because Anthropic does not require it when
    only counting prompt tokens.

    Attributes:
        model: Model ID the prompt would be sent to.
        messages: Conversation history.
        system: System prompt, as text or as content blocks.
        tools: Tool declarations, which count towards the prompt.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None


class AnthropicUsage(BaseModel):
    """Token usage in Anthropic format.

    Attributes:
        input_tokens: Tokens in the prompt.
        output_tokens: Tokens generated.
    """

    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicContent(BaseModel):
    """Content block in an Anthropic response.

    Attributes:
        type: Block type (``text`` or ``tool_use``).
        text: Generated text, for ``text`` blocks.
        id: Tool call id, for ``tool_use`` blocks.
        name: Tool name, for ``tool_use`` blocks.
        input: Tool arguments, for ``tool_use`` blocks.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["text", "tool_use"] = "text"
    text: str = ""
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


class AnthropicResponse(BaseModel):
    """Anthropic ``/v1/messages`` response.

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
    stop_sequence: str | None = None
    usage: AnthropicUsage


def system_text(system: str | list[dict[str, Any]] | None) -> str:
    """Flatten an Anthropic system prompt to text.

    Args:
        system: Text, list of content blocks, or ``None``.

    Returns:
        Concatenated text; empty when there is no system prompt.
    """
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts: list[str] = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert Anthropic tool declarations to OpenAI function tools.

    Args:
        tools: Anthropic tools (``{"name", "description", "input_schema"}``).

    Returns:
        OpenAI tools, or an empty list.
    """
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        function: dict[str, Any] = {
            "name": name,
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        }
        description = tool.get("description")
        if description:
            function["description"] = description
        converted.append({"type": "function", "function": function})
    return converted


def anthropic_tool_choice_to_openai(choice: dict[str, Any] | None) -> Any:
    """Convert an Anthropic ``tool_choice`` to its OpenAI equivalent.

    Args:
        choice: ``{"type": "auto"|"any"|"tool"|"none", "name": ...}`` or ``None``.

    Returns:
        ``"auto"``, ``"required"``, ``"none"``, an OpenAI function selector, or
        ``None`` when nothing was requested.
    """
    if not choice:
        return None
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def translate_messages(
    messages: list[AnthropicMessage],
) -> list[dict[str, Any]]:
    """Translate Anthropic messages into OpenAI chat messages.

    Assistant ``tool_use`` blocks are re-expressed as ``tool_calls`` and user
    ``tool_result`` blocks as ``tool`` messages, which is what an
    OpenAI-compatible backend expects for a tool round-trip.

    Args:
        messages: Anthropic messages.

    Returns:
        OpenAI chat messages.
    """
    converted: list[dict[str, Any]] = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            converted.append({"role": message.role, "content": content})
            continue

        if message.role == "assistant":
            texts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    texts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                    )
            payload: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
            if tool_calls:
                payload["tool_calls"] = tool_calls
            converted.append(payload)
            continue

        # user role: tool_result blocks become standalone tool messages.
        text_parts: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                result = block.get("content")
                if isinstance(result, list):
                    result = "\n".join(
                        str(part.get("text", ""))
                        for part in result
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": result if isinstance(result, str) else json.dumps(result),
                    }
                )
            elif block.get("type") == "text":
                text_parts.append({"type": "text", "text": block.get("text", "")})
            elif block.get("type") == "image":
                source = block.get("source") or {}
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    encoded = source.get("data", "")
                    text_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                        }
                    )
        if text_parts:
            if all(part["type"] == "text" for part in text_parts):
                converted.append(
                    {
                        "role": "user",
                        "content": "\n".join(part["text"] for part in text_parts),
                    }
                )
            else:
                converted.append({"role": "user", "content": text_parts})
    return converted


def anthropic_to_openai(request: AnthropicRequest) -> dict[str, Any]:
    """Translate an Anthropic request to OpenAI chat-completions format.

    Args:
        request: Anthropic ``/v1/messages`` request.

    Returns:
        OpenAI-compatible request dict, ready for routing.

    Raises:
        ValueError: Never raised today; kept in the signature for symmetry with
            :func:`openai_to_anthropic`.
    """
    messages: list[dict[str, Any]] = []
    system = system_text(request.system)
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(translate_messages(request.messages))

    openai_request: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
    }

    if request.temperature is not None:
        openai_request["temperature"] = request.temperature
    if request.top_p is not None:
        openai_request["top_p"] = request.top_p
    if request.stop_sequences:
        openai_request["stop"] = request.stop_sequences

    tools = anthropic_tools_to_openai(request.tools)
    if tools:
        openai_request["tools"] = tools
        choice = anthropic_tool_choice_to_openai(request.tool_choice)
        if choice is not None:
            openai_request["tool_choice"] = choice

    dropped = unsupported_fields(request)
    if dropped:
        logger.warning(
            "Dropping Anthropic fields with no OpenAI equivalent: %s", ", ".join(dropped)
        )

    return openai_request


def unsupported_fields(request: AnthropicRequest | AnthropicCountTokensRequest) -> list[str]:
    """List request fields that will not reach the backend.

    Args:
        request: Parsed Anthropic request.

    Returns:
        Names of fields present on the request but not translatable.
    """
    extra = getattr(request, "model_extra", None) or {}
    return [name for name in UNSUPPORTED_FIELDS if name in extra]


def openai_to_anthropic(openai_response: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate an OpenAI response to Anthropic format.

    Args:
        openai_response: OpenAI ``/v1/chat/completions`` response.
        model: Model ID to report in the response.

    Returns:
        Anthropic ``/v1/messages`` response dict.
    """
    choice = (openai_response.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    blocks: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id", ""),
                "name": function.get("name", ""),
                "input": _parse_arguments(function.get("arguments")),
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    usage = openai_response.get("usage") or {}
    stop_reason = _map_finish_reason(choice.get("finish_reason"))
    if any(block["type"] == "tool_use" for block in blocks):
        stop_reason = "tool_use"

    return {
        "id": openai_response.get("id", "msg-unknown"),
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Decode a tool-call argument string.

    Args:
        raw: JSON string supplied by the backend, or an already-decoded dict.

    Returns:
        The decoded object; empty when the string is missing or invalid.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        decoded = json.loads(raw)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


async def translate_openai_stream(
    openai_stream: AsyncIterator[bytes], model: str
) -> AsyncIterator[bytes]:
    """Translate an OpenAI SSE stream to the Anthropic Messages SSE format.

    Emits the sequence Anthropic clients expect - ``message_start``,
    ``content_block_start``, ``content_block_delta`` per fragment,
    ``content_block_stop`` per block, ``message_delta`` and ``message_stop`` -
    including tool-use blocks assembled from streaming ``tool_calls`` deltas.

    Args:
        openai_stream: OpenAI SSE byte stream.
        model: Model ID for Anthropic events.

    Yields:
        Anthropic SSE event bytes.
    """
    message_id = f"msg-{id(openai_stream)}"
    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"
    buffer = b""
    next_index = 0
    text_index: int | None = None
    open_blocks: list[int] = []
    tool_indexes: dict[int, int] = {}

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
            delta = choice.get("delta") or {}

            text = delta.get("content")
            if text:
                if text_index is None:
                    text_index = next_index
                    next_index += 1
                    open_blocks.append(text_index)
                    yield _anthropic_event(
                        "content_block_start",
                        {"index": text_index, "content_block": {"type": "text", "text": ""}},
                    )
                yield _anthropic_event(
                    "content_block_delta",
                    {
                        "index": text_index,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )

            for call in delta.get("tool_calls") or []:
                position = int(call.get("index", 0))
                function = call.get("function") or {}
                if position not in tool_indexes:
                    tool_indexes[position] = next_index
                    next_index += 1
                    open_blocks.append(tool_indexes[position])
                    yield _anthropic_event(
                        "content_block_start",
                        {
                            "index": tool_indexes[position],
                            "content_block": {
                                "type": "tool_use",
                                "id": call.get("id", f"toolu_{position}"),
                                "name": function.get("name", ""),
                                "input": {},
                            },
                        },
                    )
                arguments = function.get("arguments")
                if arguments:
                    yield _anthropic_event(
                        "content_block_delta",
                        {
                            "index": tool_indexes[position],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments,
                            },
                        },
                    )

            if choice.get("finish_reason"):
                stop_reason = _map_finish_reason(choice["finish_reason"])

    if tool_indexes:
        stop_reason = "tool_use"

    for index in sorted(open_blocks):
        yield _anthropic_event("content_block_stop", {"index": index})

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
    """Map an OpenAI ``finish_reason`` to an Anthropic ``stop_reason``.

    Args:
        openai_reason: OpenAI finish_reason value.

    Returns:
        Anthropic stop_reason value.
    """
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "refusal",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
    }
    return mapping.get(openai_reason or "", "end_turn")
