"""Pydantic schemas for the OpenAI-compatible HTTP surface.

Every request and response model is declared with explicit field types and
validations so that FastAPI renders a complete ``/docs`` and ``/redoc``
surface. ``extra="allow"`` is enabled on passthrough models so that provider
specific fields (``temperature``, ``tools``, ``logprobs``, ...) survive the
round trip untouched.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ChatRole = Literal["system", "user", "assistant", "tool", "developer", "function"]


class ChatMessage(BaseModel):
    """A single chat message in OpenAI wire format.

    Attributes:
        role: Author of the message. Standard OpenAI roles are enforced.
        content: Message text, or a list of content parts for multimodal input.
        name: Optional author name for ``function``/multi-agent flows.
        tool_call_id: Identifier linking a ``tool`` message to its call.
        tool_calls: Tool invocations requested by the assistant.
    """

    model_config = ConfigDict(extra="allow")

    role: ChatRole = Field(
        description="Author of the message (system, user, assistant, tool, developer, function)"
    )
    content: str | list[dict[str, Any]] | None = Field(
        default=None, description="Message text or multimodal content parts"
    )
    name: str | None = Field(
        default=None, description="Optional author name for function/multi-agent flows"
    )
    tool_call_id: str | None = Field(
        default=None, description="Identifier linking a tool message to its call"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool invocations requested by the assistant"
    )


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    Attributes:
        model: Requested model id. May be empty: the router is then free to
            pick any model exposed by the selected backend.
        messages: Conversation so far. Required by the OpenAI API.
        stream: When true the proxy replies with ``text/event-stream``.
        temperature: Sampling temperature forwarded to the backend.
        max_tokens: Maximum tokens to generate, forwarded to the backend.
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(
        default="",
        description=(
            "Model identifier (e.g., 'gpt-4o', 'llama3.2-1b'). "
            "Empty means router picks any available model"
        ),
    )
    messages: list[ChatMessage] = Field(
        min_length=1, description="Chat message history (at least one message required)"
    )
    stream: bool = Field(
        default=False, description="Whether to stream the response as Server-Sent Events"
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0). Controls randomness of responses",
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens to generate in the response"
    )


class ChoiceMessage(BaseModel):
    """Assistant message returned in a completion choice.

    Attributes:
        role: Always ``assistant`` for standard completions.
        content: Generated text, or ``None`` when only tool calls returned.
        tool_calls: Tool invocations requested by the model.
    """

    model_config = ConfigDict(extra="allow")

    role: Literal["assistant"] = Field(
        default="assistant", description="Message role (always 'assistant')"
    )
    content: str | None = Field(default=None, description="Generated response text")
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool/function calls requested by the model"
    )


class Choice(BaseModel):
    """A completion candidate.

    Attributes:
        index: Candidate index (0 for single-candidate requests).
        message: The generated assistant message.
        finish_reason: Why generation stopped (``stop``, ``length``, ...).
    """

    model_config = ConfigDict(extra="allow")

    index: int = Field(default=0, ge=0, description="Index of this choice in the response")
    message: ChoiceMessage = Field(description="The generated assistant message")
    finish_reason: str | None = Field(
        default=None, description="Reason generation stopped (stop, length, tool_calls, etc.)"
    )


class Usage(BaseModel):
    """Token accounting returned by the backend.

    Attributes:
        prompt_tokens: Tokens in the prompt.
        completion_tokens: Tokens generated.
        total_tokens: Sum of prompt and completion tokens.
    """

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = Field(default=0, ge=0, description="Number of tokens in the input prompt")
    completion_tokens: int = Field(
        default=0, ge=0, description="Number of tokens generated in the response"
    )
    total_tokens: int = Field(default=0, ge=0, description="Total tokens (prompt + completion)")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response.

    Attributes:
        id: Provider-assigned completion id.
        object: Always ``chat.completion``.
        created: Unix timestamp of creation.
        model: Model that actually produced the answer (post-routing).
        choices: Completion candidates, normally a single entry.
        usage: Token accounting, when the backend supplies it.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Unique identifier assigned by the backend")
    object: Literal["chat.completion"] = Field(
        default="chat.completion", description="Object type identifier"
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp when the completion was created",
    )
    model: str = Field(description="The model that generated the response (after routing)")
    choices: list[Choice] = Field(description="Array of completion choices (typically one)")
    usage: Usage | None = Field(
        default=None, description="Token usage accounting (if provided by backend)"
    )


class DeltaMessage(BaseModel):
    """Incremental assistant message carried by a streaming chunk.

    Attributes:
        role: Author, present only in the first chunk.
        content: Incremental text fragment.
        tool_calls: Incremental tool call fragments.
    """

    model_config = ConfigDict(extra="allow")

    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChunkChoice(BaseModel):
    """A streaming choice delta.

    Attributes:
        index: Candidate index.
        delta: Incremental message fragment.
        finish_reason: Present only on the final chunk.
    """

    model_config = ConfigDict(extra="allow")

    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk (one SSE ``data:`` payload).

    Attributes:
        id: Provider-assigned completion id.
        object: Always ``chat.completion.chunk``.
        created: Unix timestamp of creation.
        model: Model producing the stream (post-routing).
        choices: One delta per candidate.
        usage: Present on the final chunk when requested by the client.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice] = []
    usage: Usage | None = None


class ModelCard(BaseModel):
    """A model exposed by the proxy.

    Attributes:
        id: Model id in ``backend/model`` form.
        object: Always ``model``.
        created: Registration timestamp (proxy start time).
        owned_by: Backend that serves the model.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str


class ModelListResponse(BaseModel):
    """OpenAI-compatible model listing.

    Attributes:
        object: Always ``list``.
        data: Available model cards.
    """

    object: Literal["list"] = "list"
    data: list[ModelCard]


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embeddings request.

    Attributes:
        model: Embedding model id. Empty lets the router pick a backend.
        input: Text, list of texts, or token ids to embed.
        dimensions: Requested output width, for models that support truncation.
        encoding_format: ``float`` for a JSON array, ``base64`` for packed bytes.
        user: Opaque caller identifier forwarded to the backend.
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(
        default="", description="Embedding model identifier; empty lets the router choose"
    )
    input: str | list[str] | list[int] | list[list[int]] = Field(
        description="Text, batch of texts, or token ids to embed"
    )
    dimensions: int | None = Field(
        default=None, ge=1, description="Requested embedding width (model dependent)"
    )
    encoding_format: Literal["float", "base64"] | None = Field(
        default=None, description="Encoding for the embedding vectors"
    )
    user: str | None = Field(default=None, description="Opaque caller identifier")


class EmbeddingVector(BaseModel):
    """One embedding in an OpenAI-compatible response.

    Attributes:
        object: Always ``embedding``.
        index: Position of the input this vector belongs to.
        embedding: The vector, as floats or a base64 string.
    """

    model_config = ConfigDict(extra="allow")

    object: Literal["embedding"] = "embedding"
    index: int = Field(default=0, ge=0)
    embedding: list[float] | str


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embeddings response.

    Attributes:
        object: Always ``list``.
        data: One vector per input.
        model: Model that produced the embeddings (post-routing).
        usage: Token accounting, when the backend supplies it.
    """

    model_config = ConfigDict(extra="allow")

    object: Literal["list"] = "list"
    data: list[EmbeddingVector]
    model: str = ""
    usage: Usage | None = None


class AnthropicCountTokensResponse(BaseModel):
    """Response for ``POST /v1/messages/count_tokens``.

    Attributes:
        input_tokens: Estimated prompt token count.
    """

    input_tokens: int = Field(ge=0, description="Estimated number of prompt tokens")


class ProxyInfo(BaseModel):
    """Proxy metadata exposed at the root endpoint.

    Attributes:
        name: Application name.
        version: Installed marga version.
        docs: URL to the interactive API documentation.
        endpoints: List of available HTTP endpoints.
        backends: Names of configured backends for the default context.
        contexts: Names of configured request contexts.
        default_context: Context used when no key is presented.
    """

    name: str
    version: str
    docs: str
    endpoints: list[str]
    backends: list[str]
    contexts: list[str] = Field(default_factory=list)
    default_context: str = ""


class BackendHealth(BaseModel):
    """Health status for a single backend.

    Attributes:
        name: Backend name.
        url: Base URL.
        enabled: Whether the backend is enabled for routing.
        loopback: Whether the URL is a loopback address.
        reachable: Whether the backend is currently reachable (None if not checked).
        reason: Explanation for the reachable status.
    """

    name: str
    url: str
    enabled: bool
    loopback: bool
    reachable: bool | None = None
    reason: str = ""


class HealthResponse(BaseModel):
    """Liveness probe payload.

    Attributes:
        status: Always ``ok`` for a running process.
        version: Installed marga version.
        runtime: Detected runtime environment (``local`` or ``cloud``).
        backends: Health status for each configured backend.
    """

    status: Literal["ok"] = "ok"
    version: str
    runtime: Literal["local", "cloud"] | None = None
    backends: list[BackendHealth] | None = None


class BackendStats(BaseModel):
    """Usage counters for a single backend.

    Attributes:
        calls: Number of completed requests.
        tokens: Tokens accounted to this backend.
        cost: Estimated spend in USD (0.0 when no rate is configured).
        last_used: Unix timestamp of the most recent call, 0.0 if unused.
    """

    calls: int = 0
    tokens: int = 0
    cost: float = 0.0
    last_used: float = 0.0


class StatsResponse(BaseModel):
    """Routing statistics payload for ``GET /stats``.

    Attributes:
        backends: Per-backend counters keyed by backend name.
    """

    backends: dict[str, BackendStats]


class ErrorResponse(BaseModel):
    """Standard HTTP error payload.

    Attributes:
        detail: Human readable message.
        type: Exception class name, when the error came from marga.
    """

    model_config = ConfigDict(extra="allow")

    detail: str
    type: str | None = None
