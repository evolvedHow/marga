"""Before / after / error hooks around the LLM round trip.

Hooks are registered with decorators on :class:`marga.proxy.Proxy` and may be
plain functions or ``async def`` - the manager awaits whatever you return.

* **before** hooks mutate :class:`HookContext` (inject RAG context, enforce a
  system prompt, set sampling parameters) before routing happens.
* **after** hooks receive the backend response and return the payload that is
  sent to the client (redaction, logging, enrichment).
* **error** hooks are notified when a backend attempt fails. Returning a dict
  from an error hook short-circuits failover and serves that dict instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ._util import maybe_await

BeforeHook = Callable[["HookContext"], Any]
AfterHook = Callable[["HookContext", dict[str, Any]], Any]
ErrorHook = Callable[["HookContext", BaseException], Any]


@dataclass
class HookContext:
    """Mutable view of one proxy request shared by every hook.

    The full OpenAI request body lives in :attr:`body`; :attr:`messages` and
    :attr:`model` are live views onto it, so ``ctx.messages.insert(...)`` or
    ``ctx.model = "..."`` mutate the payload that is forwarded upstream.

    Attributes:
        body: Complete request body forwarded to the backend. Extra provider
            fields (``temperature``, ``tools``, ...) are preserved.
        stream: Whether the client requested server-sent events.
        metadata: Scratch space for passing data between hooks. The router
            writes ``metadata["backend"]`` once a target is chosen.

    Example:
        ctx = HookContext(messages=[{"role": "user", "content": "hi"}], model="")
        ctx.messages.insert(0, {"role": "system", "content": "Be terse."})
        ctx.temperature = 0.2  # extra field, forwarded untouched
    """

    body: dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        body: dict[str, Any] | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
        model: str | None = None,
        stream: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Build a context from either a raw body or explicit messages/model.

        Args:
            body: Full OpenAI request body to take ownership of.
            messages: Conversation messages (``body`` style construction).
            model: Requested model (``body`` style construction).
            stream: Whether the client asked for streaming.
            metadata: Initial hook scratch space.
        """
        self.body = dict(body) if body else {}
        if messages is not None:
            self.body["messages"] = messages
        if model is not None:
            self.body["model"] = model
        self.stream = stream or bool(self.body.get("stream", False))
        self.metadata = dict(metadata) if metadata else {}

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Live list of chat messages inside :attr:`body` (mutable)."""
        return self.body.setdefault("messages", [])

    @messages.setter
    def messages(self, value: list[dict[str, Any]]) -> None:
        """Replace the conversation stored in :attr:`body`."""
        self.body["messages"] = value

    @property
    def model(self) -> str:
        """Requested model from :attr:`body`; set it to force a model."""
        return str(self.body.get("model", ""))

    @model.setter
    def model(self, value: str) -> None:
        """Write the model into the forwarded body."""
        self.body["model"] = value

    @property
    def temperature(self) -> float | None:
        """Sampling temperature if the client supplied one."""
        value = self.body.get("temperature")
        return None if value is None else float(value)

    @temperature.setter
    def temperature(self, value: float | None) -> None:
        """Set (or clear with ``None``) the forwarded sampling temperature."""
        if value is None:
            self.body.pop("temperature", None)
        else:
            self.body["temperature"] = value

    @property
    def estimated_tokens(self) -> int:
        """Rough prompt size in tokens (4 characters per token).

        Multimodal content parts contribute their ``text`` field only; non-text
        parts are ignored, which is deliberately conservative.

        Returns:
            Estimated token count across all messages.
        """
        total = 0
        for message in self.messages:
            content = message.get("content") if isinstance(message, dict) else None
            total += _content_length(content)
        return total // 4


def _content_length(content: Any) -> int:
    """Measure a message content field in characters.

    Args:
        content: ``str``, list of content parts, or ``None``.

    Returns:
        Character count of the text payload.
    """
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    total += len(text)
        return total
    return len(str(content))


class HookManager:
    """Ordered registry of before / after / error hooks.

    Hooks run in registration order. Before and after hooks may mutate their
    argument; after hooks return the payload that gets sent onward, and error
    hooks may return a dict to take over the response.

    Example:
        hooks = HookManager()
        hooks.add_before(lambda ctx: ctx.messages.append({"role": "user", "content": ""}))
        await hooks.run_before(ctx)
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._before: list[BeforeHook] = []
        self._after: list[AfterHook] = []
        self._error: list[ErrorHook] = []

    @property
    def before_hooks(self) -> list[BeforeHook]:
        """Registered before hooks, in execution order."""
        return list(self._before)

    @property
    def after_hooks(self) -> list[AfterHook]:
        """Registered after hooks, in execution order."""
        return list(self._after)

    @property
    def error_hooks(self) -> list[ErrorHook]:
        """Registered error hooks, in execution order."""
        return list(self._error)

    def add_before(self, fn: BeforeHook) -> BeforeHook:
        """Register a before hook.

        Args:
            fn: Callable taking the :class:`HookContext`; may mutate it and
                may be a coroutine function.

        Returns:
            ``fn``, so this can be used as a decorator.
        """
        self._before.append(fn)
        return fn

    def add_after(self, fn: AfterHook) -> AfterHook:
        """Register an after hook.

        Args:
            fn: Callable taking ``(ctx, response)`` and returning the response
                payload; may be a coroutine function.

        Returns:
            ``fn``, so this can be used as a decorator.
        """
        self._after.append(fn)
        return fn

    def add_error(self, fn: ErrorHook) -> ErrorHook:
        """Register an error hook.

        Args:
            fn: Callable taking ``(ctx, exc)``. Returning a dict replaces the
                error response and stops failover; returning ``None`` lets the
                proxy continue with the next backend. May be a coroutine.

        Returns:
            ``fn``, so this can be used as a decorator.
        """
        self._error.append(fn)
        return fn

    async def run_before(self, ctx: HookContext) -> HookContext:
        """Run every before hook in registration order.

        Args:
            ctx: Context whose ``body`` hooks may mutate.

        Returns:
            The same context, after all hooks ran.
        """
        for hook in self._before:
            await maybe_await(hook(ctx))
        return ctx

    async def run_after(
        self, ctx: HookContext, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Run every after hook in registration order.

        Args:
            ctx: Context of the request that produced ``response``.
            response: Backend response payload.

        Returns:
            The response after all hooks transformed it.
        """
        for hook in self._after:
            result = await maybe_await(hook(ctx, response))
            if result is not None:
                response = result
        return response

    async def run_error(
        self, ctx: HookContext, error: BaseException
    ) -> dict[str, Any] | None:
        """Run every error hook until one recovers the request.

        Args:
            ctx: Context of the failing request.
            error: Exception raised by the backend attempt.

        Returns:
            A replacement response dict from the first hook that returns one,
            or ``None`` when no hook recovered and failover should continue.
        """
        for hook in self._error:
            result = await maybe_await(hook(ctx, error))
            if isinstance(result, dict):
                return result
        return None
