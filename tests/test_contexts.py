"""Tests for request contexts: who may use which upstream account."""

from typing import Any, Iterator, Mapping
from urllib.parse import urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from marga import Proxy
from marga.contexts import (
    ContextResolver,
    ContextSpec,
    backend_visible,
    token_digest,
)
from marga.errors import (
    AuthenticationError,
    ContextForbiddenError,
    SecretNotFoundError,
    UnknownContextError,
)
from marga.middleware import extract_token, split_context_prefix
from marga.transport import Transport


class RecordingTransport(Transport):
    """Transport that records every call and fails on demand."""

    def __init__(self, *, fail_urls: set[str] | None = None) -> None:
        """Create the stub.

        Args:
            fail_urls: Base URLs that should raise a connection error.
        """
        super().__init__()
        self.calls: list[dict[str, Any]] = []
        self.fail_urls = fail_urls or set()

    async def complete(
        self,
        base_url: str,
        api_key: str = "",
        body: dict[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Record the call and return a canned completion.

        Args:
            base_url: Backend URL.
            api_key: Key marga resolved for this backend.
            body: Request body.
            headers: Extra headers.

        Returns:
            An OpenAI-compatible completion.

        Raises:
            httpx.ConnectError: When ``base_url`` is in ``fail_urls``.
        """
        if base_url in self.fail_urls:
            raise httpx.ConnectError("backend is down")
        self.calls.append({"url": base_url, "api_key": api_key, "body": dict(body or {})})
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": (body or {}).get("model", "test"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }

    async def stream(
        self,
        base_url: str,
        api_key: str = "",
        body: dict[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Iterator[bytes]:  # type: ignore[override]
        """Yield one canned SSE chunk.

        Args:
            base_url: Backend URL.
            api_key: Resolved key.
            body: Request body.
            headers: Extra headers.

        Yields:
            A single SSE chunk.
        """
        self.calls.append({"url": base_url, "api_key": api_key, "body": dict(body or {})})
        yield b'data: {"choices":[{"delta":{"content":"pong"}}]}\n\ndata: [DONE]\n\n'


class FakeVault:
    """In-memory stand-in for the worker-env vault client.

    Mirrors the vault's real behaviour: a context lookup falls back to the
    shared default context when the key is absent.
    """

    def __init__(
        self,
        contexts: dict[str, dict[str, str]],
        default: dict[str, str] | None = None,
    ) -> None:
        """Create the stub.

        Args:
            contexts: ``context -> {key: value}`` table.
            default: Shared fallback table.
        """
        self.contexts = contexts
        self.default = default or {}
        self.lookups: list[tuple[str, str]] = []

    async def get(self, key: str, *, context: str | None = None, refresh: bool = False) -> str:
        """Read a secret.

        Args:
            key: Secret key.
            context: Context to read from.
            refresh: Ignored.

        Returns:
            The secret value.

        Raises:
            SecretNotFoundError: When neither the context nor the default has it.
        """
        resolved = context or ""
        self.lookups.append((resolved, key))
        value = self.contexts.get(resolved, {}).get(key)
        if value is None:
            value = self.default.get(key)
        if value is None:
            raise SecretNotFoundError(resolved, key)
        return value

    async def close(self) -> None:
        """No-op close."""


PERSONAL = "sk-personal-token"
NONPROFIT = "sk-nonprofit-token"


def _contexts(*, personal_allows: tuple[str, ...] = ()) -> dict[str, Any]:
    """Build a personal/non-profit context mapping.

    Args:
        personal_allows: Extra contexts the personal token may select.

    Returns:
        Context configuration mapping.
    """
    return {
        "personal": {
            "vault_context": "marga",
            "tokens": [PERSONAL],
            "default": True,
            "allow": list(personal_allows),
        },
        "nonprofit": {
            "vault_context": "nonprofit",
            "tokens": [NONPROFIT],
        },
    }


def _proxy(
    contexts: dict[str, Any] | None = None,
    *,
    backends: dict[str, Any] | None = None,
    transport: RecordingTransport | None = None,
    vault: FakeVault | None = None,
) -> tuple[Proxy, RecordingTransport]:
    """Build a proxy with two contexts and a recording transport.

    Args:
        contexts: Context mapping; defaults to :func:`_contexts`.
        backends: Backend mapping; defaults to a single inheriting backend.
        transport: Transport to use; a new one when omitted.
        vault: Vault stub; a two-account one when omitted.

    Returns:
        ``(proxy, transport)``.
    """
    transport = transport or RecordingTransport()
    vault = vault or FakeVault(
        {
            "marga": {"KEY_OPENCODE": "personal-key"},
            "nonprofit": {"KEY_OPENCODE": "nonprofit-key"},
        }
    )
    proxy = Proxy(
        backends=backends
        or {
            "opencode": {
                "base_url": "http://opencode/v1",
                "models": ["big-model"],
                "api_key_secret": "KEY_OPENCODE",
            }
        },
        contexts=contexts if contexts is not None else _contexts(),
        transport=transport,
        vault=vault,  # type: ignore[arg-type]
    )
    return proxy, transport


def _chat(client: TestClient, token: str, **extra: str) -> httpx.Response:
    """Send a chat completion as a given token.

    Args:
        client: Test client.
        token: Bearer token to present.
        **extra: Extra headers.

    Returns:
        The HTTP response.
    """
    return client.post(
        "/v1/chat/completions",
        json={"model": "big-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {token}", **extra},
    )


# ---------------------------------------------------------------------------
# Resolver unit tests
# ---------------------------------------------------------------------------


def test_token_selects_its_context():
    """A token is bound to exactly one context."""
    resolver = ContextResolver.from_mapping(_contexts())
    assert resolver.resolve(PERSONAL) == "personal"
    assert resolver.resolve(NONPROFIT) == "nonprofit"


def test_unknown_token_is_rejected():
    """An unconfigured key cannot reach any context."""
    resolver = ContextResolver.from_mapping(_contexts())
    with pytest.raises(AuthenticationError):
        resolver.resolve("sk-not-a-key")


def test_missing_token_is_rejected_when_required():
    """Authentication is required as soon as one token exists."""
    resolver = ContextResolver.from_mapping(_contexts())
    with pytest.raises(AuthenticationError):
        resolver.resolve("")


def test_selector_is_forbidden_unless_allowed():
    """A token cannot widen its reach by asking for another context."""
    resolver = ContextResolver.from_mapping(_contexts())
    with pytest.raises(ContextForbiddenError):
        resolver.resolve(NONPROFIT, "personal")

    allowed = ContextResolver.from_mapping(_contexts(personal_allows=("nonprofit",)))
    assert allowed.resolve(PERSONAL, "nonprofit") == "nonprofit"


def test_unknown_selector_is_rejected():
    """Selecting a context that does not exist is a client error."""
    resolver = ContextResolver.from_mapping(_contexts(personal_allows=("nonprofit",)))
    with pytest.raises(UnknownContextError):
        resolver.resolve(PERSONAL, "some-other-org")


def test_digest_is_stable_and_hides_the_token():
    """Only the digest is retained, and it is deterministic."""
    assert token_digest("abc") == token_digest("abc")
    assert "abc" not in token_digest("abc")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/c/nonprofit/v1/models", ("nonprofit", "/v1/models")),
        ("/c/nonprofit", ("nonprofit", "/")),
        ("/v1/models", ("", "/v1/models")),
        ("/c//v1/models", ("", "/c//v1/models")),
    ],
)
def test_context_path_prefix_splitting(path: str, expected: tuple[str, str]):
    """The path-prefix selector is split without corrupting the path."""
    assert split_context_prefix(path) == expected


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ({"authorization": "Bearer sk-1"}, "sk-1"),
        ({"authorization": "bearer sk-1"}, "sk-1"),
        ({"x-api-key": "sk-2"}, "sk-2"),
        ({}, ""),
    ],
)
def test_token_extraction(header: dict[str, str], expected: str):
    """Both auth header styles are accepted."""
    assert extract_token(header) == expected


@pytest.mark.parametrize(
    ("backend_context", "request_context", "visible"),
    [
        ("inherit", "nonprofit", True),
        ("shared", "nonprofit", True),
        ("", "nonprofit", True),
        ("nonprofit", "nonprofit", True),
        ("nonprofit", "personal", False),
    ],
)
def test_backend_visibility(backend_context: str, request_context: str, visible: bool):
    """A pinned backend is only selectable from its own context."""
    assert backend_visible(backend_context, request_context) is visible


# ---------------------------------------------------------------------------
# Proxy behaviour
# ---------------------------------------------------------------------------


def test_each_context_uses_its_own_upstream_key():
    """One backend block, two accounts: the key follows the caller's context."""
    proxy, transport = _proxy()
    client = TestClient(proxy.build_app())

    assert _chat(client, PERSONAL).status_code == 200
    assert transport.calls[-1]["api_key"] == "personal-key"

    assert _chat(client, NONPROFIT).status_code == 200
    assert transport.calls[-1]["api_key"] == "nonprofit-key"


def test_usage_is_accounted_per_context():
    """Spend never merges across contexts."""
    proxy, _ = _proxy()
    client = TestClient(proxy.build_app())

    _chat(client, PERSONAL)
    _chat(client, PERSONAL)
    _chat(client, NONPROFIT)

    personal = client.get(
        "/stats?period=all", headers={"Authorization": f"Bearer {PERSONAL}"}
    ).json()
    assert personal["backends"]["opencode"]["calls"] == 2

    nonprofit = client.get(
        "/stats?period=all", headers={"Authorization": f"Bearer {NONPROFIT}"}
    ).json()
    assert nonprofit["backends"]["opencode"]["calls"] == 1

    everything = client.get(
        "/stats?period=all&context=all", headers={"Authorization": f"Bearer {PERSONAL}"}
    ).json()
    assert everything["backends"]["personal/opencode"]["calls"] == 2
    assert everything["backends"]["nonprofit/opencode"]["calls"] == 1


def test_failover_never_leaves_the_context():
    """A dead shared backend must not fail over onto another org's account."""
    transport = RecordingTransport(fail_urls={"http://dead/v1"})
    backends = {
        "dead": {"base_url": "http://dead/v1", "models": ["big-model"]},
        "mine": {"base_url": "http://mine/v1", "models": ["big-model"], "context": "personal"},
        "theirs": {
            "base_url": "http://theirs/v1",
            "models": ["big-model"],
            "context": "nonprofit",
        },
    }
    proxy, _ = _proxy(backends=backends, transport=transport)
    client = TestClient(proxy.build_app())

    assert _chat(client, PERSONAL).status_code == 200
    personal_backends = {urlparse(call["url"]).hostname for call in transport.calls}
    assert "theirs" not in personal_backends
    assert "mine" in personal_backends

    before = len(transport.calls)
    assert _chat(client, NONPROFIT).status_code == 200
    nonprofit_backends = {urlparse(call["url"]).hostname for call in transport.calls[before:]}
    assert nonprofit_backends <= {"theirs"}
    assert "mine" not in nonprofit_backends


def test_models_are_filtered_by_context():
    """A client does not see another context's backends or models."""
    backends = {
        "shared": {"base_url": "http://shared/v1", "models": ["shared-model"]},
        "mine": {"base_url": "http://mine/v1", "models": ["mine-model"], "context": "personal"},
        "theirs": {
            "base_url": "http://theirs/v1",
            "models": ["theirs-model"],
            "context": "nonprofit",
        },
    }
    proxy, _ = _proxy(backends=backends)
    client = TestClient(proxy.build_app())

    personal = client.get("/v1/models", headers={"Authorization": f"Bearer {PERSONAL}"}).json()
    personal_ids = {card["id"] for card in personal["data"]}
    assert "theirs/theirs-model" not in personal_ids
    assert "mine/mine-model" in personal_ids
    assert "shared/shared-model" in personal_ids

    nonprofit = client.get("/v1/models", headers={"Authorization": f"Bearer {NONPROFIT}"}).json()
    nonprofit_ids = {card["id"] for card in nonprofit["data"]}
    assert "theirs/theirs-model" in nonprofit_ids
    assert "mine/mine-model" not in nonprofit_ids


def test_advertise_lists_passthrough_models():
    """A gateway with no model allowlist can still publish its models."""
    proxy, _ = _proxy(
        backends={
            "gateway": {
                "base_url": "http://gateway/v1",
                "models": [],
                "advertise": ["frontier-1", "frontier-2"],
            }
        }
    )
    client = TestClient(proxy.build_app())
    cards = client.get("/v1/models", headers={"Authorization": f"Bearer {PERSONAL}"}).json()
    assert {card["id"] for card in cards["data"]} == {
        "gateway/frontier-1",
        "gateway/frontier-2",
    }


def test_allowed_selector_switches_context():
    """A token that is allowed to may select another context explicitly."""
    proxy, transport = _proxy(contexts=_contexts(personal_allows=("nonprofit",)))
    client = TestClient(proxy.build_app())

    response = _chat(client, PERSONAL, **{"X-Marga-Context": "nonprofit"})
    assert response.status_code == 200
    assert transport.calls[-1]["api_key"] == "nonprofit-key"


def test_disallowed_selector_is_403():
    """Selecting a context the token does not hold is forbidden."""
    proxy, _ = _proxy()
    client = TestClient(proxy.build_app())

    response = _chat(client, NONPROFIT, **{"X-Marga-Context": "personal"})
    assert response.status_code == 403
    assert response.json()["type"] == "ContextForbiddenError"


def test_unknown_selector_is_400():
    """An unknown context name is reported, not silently ignored."""
    proxy, _ = _proxy(contexts=_contexts(personal_allows=("nonprofit",)))
    client = TestClient(proxy.build_app())

    response = _chat(client, PERSONAL, **{"X-Marga-Context": "nope"})
    assert response.status_code == 400
    assert response.json()["type"] == "UnknownContextError"


def test_path_prefix_selects_context():
    """``/c/<context>/v1/...`` works for clients with only a base URL setting."""
    proxy, transport = _proxy()
    client = TestClient(proxy.build_app())

    response = client.post(
        "/c/nonprofit/v1/chat/completions",
        json={"model": "big-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {NONPROFIT}"},
    )
    assert response.status_code == 200, response.text
    assert transport.calls[-1]["api_key"] == "nonprofit-key"


def test_openai_401_envelope():
    """OpenAI clients get an OpenAI-shaped error with a WWW-Authenticate header."""
    proxy, _ = _proxy()
    client = TestClient(proxy.build_app())

    response = client.post(
        "/v1/chat/completions",
        json={"model": "big-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer sk-wrong"},
    )
    assert response.status_code == 401
    body = response.json()
    assert "error" in body
    assert body["error"]["type"] == "AuthenticationError"
    assert response.headers["www-authenticate"] == "Bearer"


def test_anthropic_401_envelope():
    """Anthropic SDKs get the envelope they can parse."""
    proxy, _ = _proxy()
    client = TestClient(proxy.build_app())

    response = client.post(
        "/v1/messages",
        json={
            "model": "big-model",
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert response.status_code == 401
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "authentication_error"


def test_missing_key_disables_only_that_context():
    """A context whose key is missing degrades alone."""
    vault = FakeVault({"marga": {"KEY_OPENCODE": "personal-key"}, "nonprofit": {}})
    proxy, _ = _proxy(vault=vault)
    client = TestClient(proxy.build_app())

    assert _chat(client, PERSONAL).status_code == 200
    response = _chat(client, NONPROFIT)
    assert response.status_code == 503
    assert response.json()["type"] == "NoBackendsForContextError"


def test_concurrent_first_requests_resolve_keys_once():
    """Two simultaneous first calls in one context do not race on the key."""
    proxy, transport = _proxy()
    client = TestClient(proxy.build_app())

    responses = [
        _chat(client, PERSONAL),
        _chat(client, PERSONAL),
    ]
    assert all(response.status_code == 200 for response in responses)
    assert {call["api_key"] for call in transport.calls} == {"personal-key"}


def test_no_contexts_configured_stays_open():
    """Without tokens, marga serves the default context and ignores selectors."""
    proxy = Proxy(
        backends={"test": {"base_url": "http://test/v1", "models": ["m"]}},
        transport=RecordingTransport(),
    )
    client = TestClient(proxy.build_app())

    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Marga-Context": "nonprofit"},
    )
    assert response.status_code == 200
    assert proxy.default_context == "default"


def test_legacy_auth_token_still_works(monkeypatch: pytest.MonkeyPatch):
    """MARGA_AUTH_TOKEN alone bootstraps a single default context."""
    monkeypatch.setenv("MARGA_AUTH_TOKEN", "sk-legacy")
    proxy = Proxy(
        backends={"test": {"base_url": "http://test/v1", "models": ["m"]}},
        transport=RecordingTransport(),
    )
    client = TestClient(proxy.build_app())

    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat/completions", json=body).status_code == 401
    ok = client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-legacy"}
    )
    assert ok.status_code == 200


def test_context_spec_defaults():
    """An unspecified vault context inherits the proxy's own."""
    spec = ContextSpec(name="personal")
    assert spec.resolved_vault_context("marga") == "marga"
    assert ContextSpec(name="org", vault_context="org").resolved_vault_context("marga") == "org"


# ---------------------------------------------------------------------------
# Tokens held in the vault
# ---------------------------------------------------------------------------


def _secret_token_vault(*, include_nonprofit_token: bool = True) -> FakeVault:
    """Build a vault holding context tokens plus one backend key.

    Args:
        include_nonprofit_token: Whether the org's token secret exists.

    Returns:
        The vault stub.
    """
    personal: dict[str, str] = {"MARGA_TOKEN_PERSONAL": PERSONAL, "KEY_OPENCODE": "personal-key"}
    nonprofit: dict[str, str] = {"KEY_OPENCODE": "nonprofit-key"}
    if include_nonprofit_token:
        nonprofit["MARGA_TOKEN_NONPROFIT"] = NONPROFIT
    return FakeVault({"marga": personal, "nonprofit": nonprofit})


def test_tokens_are_loaded_from_the_vault():
    """``token_secret`` keeps the bearer token out of the config file."""
    proxy = Proxy(
        backends={
            "opencode": {
                "base_url": "http://opencode/v1",
                "models": ["big-model"],
                "api_key_secret": "KEY_OPENCODE",
            }
        },
        contexts={
            "personal": {
                "vault_context": "marga",
                "token_secret": "MARGA_TOKEN_PERSONAL",
                "default": True,
            },
            "nonprofit": {
                "vault_context": "nonprofit",
                "token_secret": "MARGA_TOKEN_NONPROFIT",
            },
        },
        transport=RecordingTransport(),
        vault=_secret_token_vault(),  # type: ignore[arg-type]
    )
    client = TestClient(proxy.build_app())

    assert proxy.contexts.required is True
    assert _chat(client, PERSONAL).status_code == 200
    assert _chat(client, NONPROFIT).status_code == 200


def test_unresolvable_token_fails_closed():
    """A missing token secret must not silently open the proxy."""
    proxy = Proxy(
        backends={"opencode": {"base_url": "http://opencode/v1", "models": ["big-model"]}},
        contexts={
            "personal": {"vault_context": "marga", "tokens": [PERSONAL], "default": True},
            "nonprofit": {
                "vault_context": "nonprofit",
                "token_secret": "MARGA_TOKEN_MISSING",
            },
        },
        transport=RecordingTransport(),
        vault=_secret_token_vault(include_nonprofit_token=False),  # type: ignore[arg-type]
    )
    client = TestClient(proxy.build_app())

    assert proxy.contexts.required is True
    assert _chat(client, PERSONAL).status_code == 200
    # The org context never received its token, so nothing can reach it.
    assert _chat(client, NONPROFIT).status_code == 401
    assert _chat(client, "").status_code == 401


def test_streaming_survives_the_context_middleware():
    """SSE passes through ContextMiddleware unbuffered and intact."""
    proxy, _ = _proxy()
    client = TestClient(proxy.build_app())

    response = client.post(
        "/v1/messages",
        json={
            "model": "big-model",
            "max_tokens": 8,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"x-api-key": NONPROFIT},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: content_block_delta" in response.text
    assert response.text.rstrip().endswith('data: {"type": "message_stop"}')


def test_toml_configuration_declares_contexts(tmp_path):
    """``[contexts.*]`` in TOML reaches the resolver, including backend scoping."""
    config = tmp_path / "marga.toml"
    config.write_text(
        """
name = "marga"
vault_context = "marga"
default_context = "personal"

[contexts.personal]
vault_context = "marga"
tokens = ["sk-personal"]
default = true

[contexts.nonprofit]
vault_context = "nonprofit"
tokens = ["sk-nonprofit"]

[backends.shared]
base_url = "http://shared/v1"
api_key_secret = "KEY_SHARED"
models = ["shared-model"]

[backends.org_only]
base_url = "http://org/v1"
api_key_secret = "KEY_ORG"
context = "nonprofit"
models = ["org-model"]
""",
        encoding="utf-8",
    )
    proxy = Proxy.from_toml(
        config,
        transport=RecordingTransport(),
        vault=FakeVault(
            {"marga": {}, "nonprofit": {}}, default={"KEY_SHARED": "s", "KEY_ORG": "o"}
        ),
    )

    assert proxy.default_context == "personal"
    assert proxy.contexts.names == ["nonprofit", "personal"]
    assert proxy.contexts.required is True
    assert "org_only" not in proxy.router.backends
    assert "shared" in proxy.router.backends


def test_backend_capabilities_come_from_config():
    """Capabilities decide which endpoint family a backend serves."""
    proxy = Proxy(
        backends={
            "chat": {"base_url": "http://chat/v1"},
            "embed": {"base_url": "http://embed/v1", "capabilities": ["embeddings"]},
        },
        transport=RecordingTransport(),
    )
    assert [backend.name for backend in proxy.router.backends_for("chat")] == ["chat"]
    assert [backend.name for backend in proxy.router.backends_for("embeddings")] == ["embed"]


def test_default_context_vault_context_is_honoured():
    """A default context may name a vault context other than the proxy's."""
    transport = RecordingTransport()
    proxy = Proxy(
        backends={
            "opencode": {
                "base_url": "http://opencode/v1",
                "models": ["big-model"],
                "api_key_secret": "KEY_OPENCODE",
            }
        },
        contexts={
            "personal": {
                "vault_context": "own-accounts",
                "tokens": [PERSONAL],
                "default": True,
            }
        },
        vault=FakeVault({"own-accounts": {"KEY_OPENCODE": "own-key"}}),  # type: ignore[arg-type]
        vault_context="marga",
        transport=transport,
    )
    client = TestClient(proxy.build_app())

    assert _chat(client, PERSONAL).status_code == 200
    assert transport.calls[-1]["api_key"] == "own-key"


def test_backend_vault_context_override():
    """A backend can pin its key to one vault context, ignoring the caller."""
    transport = RecordingTransport()
    proxy = Proxy(
        backends={
            "shared_org_key": {
                "base_url": "http://shared/v1",
                "models": ["big-model"],
                "api_key_secret": "KEY_SHARED",
                "vault_context": "nonprofit",
            }
        },
        contexts={
            "personal": {"vault_context": "marga", "tokens": [PERSONAL], "default": True},
            "nonprofit": {"vault_context": "nonprofit", "tokens": [NONPROFIT]},
        },
        vault=FakeVault({"marga": {}, "nonprofit": {"KEY_SHARED": "shared-key"}}),  # type: ignore[arg-type]
        transport=transport,
    )
    client = TestClient(proxy.build_app())

    assert _chat(client, PERSONAL).status_code == 200
    assert transport.calls[-1]["api_key"] == "shared-key"


def test_invalid_context_name_is_rejected():
    """Context names are slugs, because they appear in store keys and paths."""
    with pytest.raises(ValueError, match="Invalid context name"):
        ContextResolver.from_mapping({"bad/name": "marga"})
    with pytest.raises(ValueError, match="Invalid context name"):
        ContextResolver.from_mapping({"Bad Name": "marga"})
    assert ContextResolver.from_mapping({"non-profit_2": "marga"}).names == ["non-profit_2"]
