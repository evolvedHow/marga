"""Exception hierarchy for marga.

Every error raised by the framework derives from :class:`MargaError` so that
applications can catch a single base class. The proxy endpoint maps these
exceptions onto HTTP status codes:

    UnknownBackendError     -> 400 Bad Request
    BudgetExhaustedError    -> 429 Too Many Requests
    AllBackendsFailedError  -> 502 Bad Gateway
    UnknownContextError     -> 400 Bad Request
    ContextForbiddenError   -> 403 Forbidden
    NoBackendsForContextError -> 503 Service Unavailable

The module also renders errors in the envelope of the ingress API that received
the request, because an Anthropic SDK cannot parse an OpenAI error body (and
vice versa).
"""

from __future__ import annotations

from typing import Any


class MargaError(Exception):
    """Base class for all marga errors."""


class UnknownBackendError(MargaError):
    """Raised when a routing decision references an unconfigured backend.

    Attributes:
        backend: Name of the backend that was not found.
        available: Names of the backends that are actually configured.
    """

    def __init__(self, backend: str, available: list[str]) -> None:
        self.backend = backend
        self.available = available
        super().__init__(f"Unknown backend {backend!r}; configured: {', '.join(available)}")


class BudgetExhaustedError(MargaError):
    """Raised when every backend has exhausted its configured budget.

    Attributes:
        backend_names: The backends whose budget was checked.
        details: Optional per-backend explanation of why each was skipped.
    """

    def __init__(self, backend_names: list[str], details: str = "") -> None:
        self.backend_names = backend_names
        self.details = details
        message = f"All backends exhausted: {', '.join(backend_names)}"
        if details:
            message = f"{message} ({details})"
        super().__init__(message)


class VaultError(MargaError):
    """Base class for vault failures.

    Attributes:
        status: HTTP status returned by the vault, 0 for client-side errors.
        detail: Message returned by the vault, when it supplied one.
    """

    def __init__(self, message: str, *, status: int = 0, detail: str = "") -> None:
        self.status = status
        self.detail = detail
        super().__init__(message)


class VaultNotConfiguredError(VaultError):
    """Raised when a vault lookup is required but no credentials are present.

    Usually means ``VAULT_URL`` / ``VAULT_TOKEN`` were not exported. For local
    development they live in ``vault/.dev.vars``; see the homestead
    security rules for the deployed location.
    """

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(
            "Vault credentials missing: "
            + ", ".join(missing)
            + ". Set them from the vault (local dev: vault/.dev.vars)."
        )


class SecretNotFoundError(VaultError):
    """Raised when the vault has no secret under the requested key.

    Attributes:
        context: Vault context that was searched.
        key: Secret key that was not found.
    """

    def __init__(self, context: str, key: str) -> None:
        self.context = context
        self.key = key
        super().__init__(
            f"Secret {context}/{key} not found in the vault",
            status=404,
        )


class ContextError(MargaError):
    """Base class for request-context selection failures."""


class UnknownContextError(ContextError):
    """Raised when a caller asks for a context that is not configured.

    Attributes:
        context: The context name that was requested.
        available: Context names that are configured.
    """

    def __init__(self, context: str, available: list[str]) -> None:
        self.context = context
        self.available = available
        super().__init__(
            f"Unknown context {context!r}; configured: {', '.join(available) or '(none)'}"
        )


class ContextForbiddenError(ContextError):
    """Raised when a token is not permitted to select the requested context.

    Attributes:
        requested: The context the caller asked for.
        token_context: The context the presented token is bound to.
    """

    def __init__(self, requested: str, token_context: str) -> None:
        self.requested = requested
        self.token_context = token_context
        super().__init__(
            f"Token bound to context {token_context!r} may not select context {requested!r}"
        )


class AuthenticationError(ContextError):
    """Raised when no token was presented and authentication is required.

    Attributes:
        detail: Short explanation of what was missing.
    """

    def __init__(self, detail: str = "Missing API key") -> None:
        self.detail = detail
        super().__init__(detail)


class NoBackendsForContextError(ContextError):
    """Raised when a context has no usable backend left.

    This happens when every backend visible to a context failed to resolve its
    credentials. It is scoped to that context: other contexts keep serving.

    Attributes:
        context: The affected context.
        failures: ``(backend, reason)`` pairs collected while materialising.
    """

    def __init__(self, context: str, failures: list[tuple[str, str]] | None = None) -> None:
        self.context = context
        self.failures = failures or []
        message = f"No usable backend for context {context!r}"
        if self.failures:
            message = f"{message}: " + "; ".join(
                f"{name} ({reason})" for name, reason in self.failures
            )
        super().__init__(message)


class BackendFailedError(MargaError):
    """Raised when a single backend request fails.

    Attributes:
        backend: Name of the backend that failed.
        cause: The original exception raised by the transport.
    """

    def __init__(self, backend: str, cause: Exception) -> None:
        self.backend = backend
        self.cause = cause
        super().__init__(f"Backend {backend!r} failed: {cause}")


class AllBackendsFailedError(MargaError):
    """Raised when every candidate backend failed during failover.

    Attributes:
        attempts: List of :class:`BackendFailedError` in the order tried.
    """

    def __init__(self, attempts: list[BackendFailedError]) -> None:
        self.attempts = attempts
        summary = "; ".join(str(attempt) for attempt in attempts)
        super().__init__(f"All backends failed: {summary}")


def http_status_for(error: Exception) -> int:
    """Map a marga exception to the HTTP status code the proxy should return.

    Args:
        error: Exception raised while serving a request.

    Returns:
        HTTP status code (400, 401, 403, 429, 502, 503) or 500 for
        unclassified errors.
    """
    if isinstance(error, UnknownBackendError):
        return 400
    if isinstance(error, AuthenticationError):
        return 401
    if isinstance(error, UnknownContextError):
        return 400
    if isinstance(error, ContextForbiddenError):
        return 403
    if isinstance(error, BudgetExhaustedError):
        return 429
    if isinstance(error, AllBackendsFailedError):
        return 502
    if isinstance(error, NoBackendsForContextError):
        return 503
    return 500


def error_detail(error: Exception) -> Any:
    """Build a JSON-serialisable error payload for an HTTP error response.

    Args:
        error: Exception raised while serving a request.

    Returns:
        A dict with a ``detail`` key plus structured fields for marga errors.
    """
    payload: dict[str, Any] = {"detail": str(error), "type": type(error).__name__}
    if isinstance(error, BudgetExhaustedError):
        payload["backends"] = error.backend_names
    if isinstance(error, AllBackendsFailedError):
        payload["attempts"] = [
            {"backend": attempt.backend, "error": str(attempt.cause)} for attempt in error.attempts
        ]
    return payload


ANTHROPIC_PATH_PREFIX = "/v1/messages"

#: HTTP status -> Anthropic ``error.type`` value.
_ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    503: "overloaded_error",
    529: "overloaded_error",
}


def is_anthropic_path(path: str) -> bool:
    """Report whether a request path belongs to the Anthropic ingress.

    Args:
        path: Request path, e.g. ``"/v1/messages"`` or ``"/v1/chat/completions"``.

    Returns:
        ``True`` when the caller speaks the Anthropic Messages API.
    """
    return path.startswith(ANTHROPIC_PATH_PREFIX)


def anthropic_error_payload(status: int, message: str, *, kind: str) -> dict[str, Any]:
    """Build an Anthropic-shaped error body.

    Anthropic SDKs parse ``{"type": "error", "error": {"type", "message"}}`` and
    raise a parse error on any other shape, which hides the real cause.

    Args:
        status: HTTP status code being returned.
        message: Human-readable description.
        kind: Error class name, appended to the Anthropic error type.

    Returns:
        JSON-serialisable response body.
    """
    return {
        "type": "error",
        "error": {
            "type": _ANTHROPIC_ERROR_TYPES.get(status, "api_error"),
            "message": message if not kind else f"{message} [{kind}]",
        },
    }


def openai_error_payload(status: int, message: str, *, kind: str) -> dict[str, Any]:
    """Build an OpenAI-shaped error body.

    ``detail`` is retained alongside ``error`` because it is what marga has
    always returned and what its own tests assert on; the ``error`` object is
    what OpenAI SDKs actually read.

    Args:
        status: HTTP status code being returned.
        message: Human-readable description.
        kind: Error class name, forwarded as ``error.type``.

    Returns:
        JSON-serialisable response body.
    """
    return {
        "error": {
            "message": message,
            "type": kind,
            "code": status,
        },
        "detail": message,
        "type": kind,
    }


def error_payload_for_path(
    path: str,
    status: int,
    message: str,
    *,
    kind: str,
) -> dict[str, Any]:
    """Render an error body in the envelope the client expects.

    Args:
        path: Request path used to pick the envelope.
        status: HTTP status code being returned.
        message: Human-readable description.
        kind: Error class name.

    Returns:
        Anthropic-shaped body for ``/v1/messages*``, OpenAI-shaped otherwise.
    """
    if is_anthropic_path(path):
        return anthropic_error_payload(status, message, kind=kind)
    return openai_error_payload(status, message, kind=kind)
