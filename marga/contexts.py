"""Request contexts: which credential set a caller is billed to.

A *context* is the unit of credential isolation. One marga process can front
several upstream accounts - a personal OpenCode subscription and a non-profit
one, say - and each caller is served by exactly the account their context names.

Contexts deliberately mirror the ``worker-env`` vault's existing contexts rather
than inventing a parallel roster: a context with no vault context of its own
falls back to the vault's shared default, which is where the rule "if the
context is empty it is available everywhere" already lives. marga therefore
never stores a second copy of who-has-which-key.

Three ways to select a context, in precedence order:

1. ``X-Marga-Context: <name>`` - an explicit selector. Only honoured when the
   presented token is permitted for that context; a token can never widen its
   own reach by asking for another context.
2. ``/c/<name>/v1/...`` - a path prefix, for clients that let you change the
   base URL but not the headers.
3. The context the API key is bound to - the normal case, because every chat
   client already sends its key on every request.

The API key is *not* the context: it is a bearer token that maps to one. That
keeps contexts rotatable and stops a guessed context name from billing someone
else's account.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from .errors import AuthenticationError, ContextForbiddenError, UnknownContextError

#: Header used to override the token's context, when the token allows it.
CONTEXT_HEADER = "x-marga-context"

#: Path prefix form: ``/c/<context>/v1/chat/completions``.
CONTEXT_PATH_PREFIX = "/c/"

#: Environment variable holding the default context name.
DEFAULT_CONTEXT_ENV = "MARGA_DEFAULT_CONTEXT"

#: Name used when no default context is configured.
DEFAULT_CONTEXT_NAME = "default"

#: Backend ``context`` value: usable everywhere, key resolved per request context.
BACKEND_CONTEXT_INHERIT = "inherit"

#: Backend ``context`` value: usable everywhere, key resolved in a fixed context.
BACKEND_CONTEXT_SHARED = "shared"


def token_digest(token: str) -> str:
    """Hash a bearer token for storage.

    Only digests are kept in memory, so a heap dump or a stray ``repr`` cannot
    leak a working marga key. Comparison happens on the digest, which is a
    fixed-length value and therefore not vulnerable to prefix-length probing.

    Args:
        token: Raw bearer token presented by a client.

    Returns:
        Hex SHA-256 digest of the token.
    """
    return hashlib.sha256(token.encode()).hexdigest()


#: Context names appear in usage-store keys and in ``/c/<name>/`` paths, so
#: they are restricted to a slug.
_CONTEXT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def validate_context_name(name: str) -> str:
    """Validate a context name.

    Context names are embedded in usage-store keys (``<context>::<backend>``)
    and in path prefixes, so punctuation that could be mistaken for a separator
    is rejected rather than silently producing ambiguous counters.

    Args:
        name: Candidate context name.

    Returns:
        The name, unchanged.

    Raises:
        ValueError: When the name is empty or contains anything but lowercase
            letters, digits, ``-`` and ``_``.
    """
    if not _CONTEXT_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid context name {name!r}: use lowercase letters, digits, '-' and '_' "
            "(e.g. 'personal', 'non-profit')"
        )
    return name


@dataclass
class ContextSpec:
    """Configuration for one context.

    Attributes:
        name: Context name used on the wire (``"nonprofit"``).
        vault_context: ``worker-env`` context holding this context's API keys.
            Empty means "inherit the proxy's vault context".
        token_secret: Vault key holding this context's bearer token, resolved
            lazily so a missing token degrades one context instead of all.
        token_env: Environment variable holding the bearer token (dev/CI).
        tokens: Tokens declared inline in configuration.
        allow: Extra context names this context's tokens may select with
            :data:`CONTEXT_HEADER` or the path prefix.
        is_default: Whether this is the context used when a caller presents no
            token and authentication is disabled.
    """

    name: str
    vault_context: str = ""
    token_secret: str = ""
    token_env: str = ""
    tokens: tuple[str, ...] = ()
    allow: frozenset[str] = frozenset()
    is_default: bool = False

    def resolved_vault_context(self, fallback: str) -> str:
        """Return the vault context to read keys from.

        Args:
            fallback: The proxy's own vault context, used when this context
                does not pin one.

        Returns:
            The vault context name.
        """
        return self.vault_context or fallback


class ContextResolver:
    """Maps bearer tokens and selectors onto a context name.

    Example:
        resolver = ContextResolver([
            ContextSpec(name="personal", tokens=("sk-personal",), is_default=True),
            ContextSpec(name="nonprofit", tokens=("sk-np",), allow=frozenset({"personal"})),
        ])
        assert resolver.resolve("sk-np", None) == "nonprofit"
        assert resolver.resolve("sk-np", "personal") == "personal"
        with pytest.raises(ContextForbiddenError):
            resolver.resolve("sk-personal", "nonprofit")
    """

    def __init__(
        self,
        contexts: list[ContextSpec] | None = None,
        *,
        default: str | None = None,
        required: bool | None = None,
    ) -> None:
        """Create a resolver.

        Args:
            contexts: Configured contexts. An empty list means "no contexts
                configured": every request lands in
                :data:`DEFAULT_CONTEXT_NAME` and no authentication happens.
            default: Default context name. Falls back to the context flagged
                ``is_default``, then to :data:`DEFAULT_CONTEXT_NAME`.
            required: Force authentication on (``True``) or off (``False``).
                ``None`` derives it from whether any token is configured.
        """
        self._specs: dict[str, ContextSpec] = {}
        self._by_digest: dict[str, str] = {}
        self._pending_secrets: dict[str, str] = {}
        self._declared_auth = False
        for spec in contexts or []:
            self.register(spec)
        self._default_override = default
        self._required_override = required

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def register(self, spec: ContextSpec) -> None:
        """Add or replace a context, discarding any tokens bound to it.

        Args:
            spec: Context to register.

        Raises:
            ValueError: When the context name is not a valid slug.
        """
        validate_context_name(spec.name)
        self._specs[spec.name] = spec
        if spec.token_secret or spec.token_env or spec.tokens:
            self._declared_auth = True
        if spec.token_secret:
            self._pending_secrets[spec.name] = spec.token_secret
        for token in spec.tokens:
            self.add_token(spec.name, token)
        env_token = os.environ.get(spec.token_env, "").strip() if spec.token_env else ""
        if env_token:
            self.add_token(spec.name, env_token)

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        *,
        default: str | None = None,
        required: bool | None = None,
    ) -> ContextResolver:
        """Build a resolver from a ``[contexts]`` TOML table or JSON object.

        Two shapes are accepted per entry::

            [contexts]
            personal = "marga"                       # name -> vault context
            nonprofit = { vault_context = "nonprofit", token_secret = "MARGA_TOKEN_NP" }

        Args:
            config: Mapping of context name to a vault-context string or a
                table with ``vault_context``, ``token_secret``, ``token_env``,
                ``tokens``, ``allow`` and ``default`` keys.
            default: Explicit default context name.
            required: Force authentication on or off.

        Returns:
            A resolver holding every declared context.

        Raises:
            ValueError: When an entry has the wrong type.
        """
        specs: list[ContextSpec] = []
        for name, raw in config.items():
            if isinstance(raw, str):
                specs.append(ContextSpec(name=name, vault_context=raw))
                continue
            if not isinstance(raw, Mapping):
                raise ValueError(f"Context {name!r} must be a string or a table")
            specs.append(
                ContextSpec(
                    name=name,
                    vault_context=str(raw.get("vault_context", "")),
                    token_secret=str(raw.get("token_secret", "")),
                    token_env=str(raw.get("token_env", "")),
                    tokens=tuple(str(token) for token in raw.get("tokens", [])),
                    allow=frozenset(str(item) for item in raw.get("allow", [])),
                    is_default=bool(raw.get("default", False)),
                )
            )
        return cls(specs, default=default, required=required)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        """Configured context names, sorted."""
        return sorted(self._specs)

    @property
    def default(self) -> str:
        """Context used when a caller presents no token and auth is disabled."""
        if self._default_override:
            return self._default_override
        for spec in self._specs.values():
            if spec.is_default:
                return spec.name
        if self._specs:
            return sorted(self._specs)[0]
        return DEFAULT_CONTEXT_NAME

    @property
    def required(self) -> bool:
        """Whether a valid API key is mandatory to reach the proxy.

        True as soon as any context *declares* a token source, even if the
        token has not been fetched yet. That matters when the vault is
        unreachable at startup: the proxy must fail closed (401) rather than
        silently serve the default context to unauthenticated callers.
        """
        if self._required_override is not None:
            return self._required_override
        return self._declared_auth or bool(self._by_digest)

    @property
    def token_secrets(self) -> dict[str, str]:
        """Pending ``context -> vault key`` token lookups, for the proxy to resolve."""
        return dict(self._pending_secrets)

    def specs(self) -> Iterator[ContextSpec]:
        """Iterate over the configured contexts in declaration order."""
        return iter(list(self._specs.values()))

    def spec(self, name: str) -> ContextSpec | None:
        """Return the spec for a context, or ``None`` when unknown.

        Args:
            name: Context name.
        """
        return self._specs.get(name)

    def is_known(self, name: str) -> bool:
        """Report whether a context name is configured.

        Args:
            name: Context name to test.
        """
        return name in self._specs

    def vault_context_for(self, name: str, fallback: str) -> str:
        """Return the vault context a request context reads keys from.

        Args:
            name: Request context.
            fallback: The proxy's vault context.

        Returns:
            The configured vault context, or ``fallback``.
        """
        spec = self._specs.get(name)
        if spec is None:
            return fallback
        return spec.resolved_vault_context(fallback)

    def add_token(self, name: str, token: str) -> None:
        """Bind a bearer token to a context.

        Args:
            name: Context the token grants.
            token: Raw token value; only its digest is retained.
        """
        if not token:
            return
        digest = token_digest(token)
        self._by_digest[digest] = name
        self._pending_secrets.pop(name, None)

    def context_for_token(self, token: str) -> str | None:
        """Look up the context a token is bound to.

        Args:
            token: Raw bearer token; empty or unknown tokens return ``None``.

        Returns:
            The bound context name, or ``None``.
        """
        if not token:
            return None
        return self._by_digest.get(token_digest(token))

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def resolve(self, token: str, requested: str | None = None) -> str:
        """Decide which context serves this request.

        Args:
            token: Bearer token presented by the client (may be empty).
            requested: Context from :data:`CONTEXT_HEADER` or the path prefix.

        Returns:
            The context name to serve the request in.

        Raises:
            AuthenticationError: When authentication is required and the token
                is missing or unknown.
            UnknownContextError: When ``requested`` is not configured.
            ContextForbiddenError: When the token may not select ``requested``.
        """
        bound = self.context_for_token(token)
        if bound is None:
            if self.required:
                raise AuthenticationError(
                    "Missing or invalid API key" if not token else "Invalid API key"
                )
            bound = self.default

        if not requested or requested == bound:
            return bound

        if not self.is_known(requested):
            raise UnknownContextError(requested, self.names)

        spec = self._specs.get(bound)
        allowed = spec.allow if spec is not None else frozenset()
        if requested not in allowed:
            raise ContextForbiddenError(requested, bound)
        return requested


def backend_visible(backend_context: str, request_context: str) -> bool:
    """Report whether a backend is selectable from a request context.

    Args:
        backend_context: The backend's ``context`` setting: ``"inherit"``,
            ``"shared"``, or a pinned context name.
        request_context: The context serving the request.

    Returns:
        ``True`` when the backend may serve this request.

    Example:
        >>> backend_visible("inherit", "nonprofit")
        True
        >>> backend_visible("nonprofit", "personal")
        False
    """
    if backend_context in ("", BACKEND_CONTEXT_INHERIT, BACKEND_CONTEXT_SHARED):
        return True
    return backend_context == request_context
