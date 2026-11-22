"""Exception hierarchy for marga.

Every error raised by the framework derives from :class:`MargaError` so that
applications can catch a single base class. The proxy endpoint maps these
exceptions onto HTTP status codes:

    UnknownBackendError    -> 400 Bad Request
    BudgetExhaustedError   -> 429 Too Many Requests
    AllBackendsFailedError -> 502 Bad Gateway
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
    """Base class for worker-env vault failures.

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
    development they live in ``worker-env/.dev.vars``; see the homestead
    security rules for the deployed location.
    """

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(
            "Vault credentials missing: "
            + ", ".join(missing)
            + ". Set them from the worker-env vault (local dev: worker-env/.dev.vars)."
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
            f"Secret {context}/{key} not found in the worker-env vault",
            status=404,
        )


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
        HTTP status code (400, 429, 502) or 500 for unclassified errors.
    """
    if isinstance(error, UnknownBackendError):
        return 400
    if isinstance(error, BudgetExhaustedError):
        return 429
    if isinstance(error, AllBackendsFailedError):
        return 502
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
