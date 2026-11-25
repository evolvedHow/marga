# Contexts

A **context** is the credential scope a request is billed to. One marga process
can front several upstream accounts — your personal OpenCode subscription, a
non-profit's — and each caller is served by exactly the account their context
names.

This document explains the model, the isolation guarantees, and how to wire the
several kinds of client that will talk to marga.

## Why a context is not just an API key

Three things get conflated when you first think about this:

1. **Which upstream account pays** — the credential context.
2. **Who is calling, and proving it** — authentication.
3. **Which backends are eligible** — routing scope.

They must agree, but they are not the same mechanism. The obvious shortcut is
"the API key *is* the context name" (`Authorization: Bearer nonprofit`), and it
is wrong: context names are memorable and guessable, so anyone who guessed one
would bill that organisation's account. The key must carry real entropy.

So a context is selected *by* a key, not *named* by it:

```
API key ──lookup──▶ context ──lookup──▶ vault context ──lookup──▶ upstream key
```

marga keeps only SHA-256 digests of keys in memory, so a process dump cannot
leak a working key.

## Where the accounts live

marga deliberately does **not** hold a second roster of who-has-which-key. A
context maps onto an existing `worker-env` vault context, which already
implements overlay semantics and already falls back to its shared default:

| marga context | vault context | upstream key read |
|---|---|---|
| `personal` (default) | `marga` | `marga/KEY_OPENCODE` |
| `nonprofit` | `nonprofit` | `nonprofit/KEY_OPENCODE` |
| either | — | falls back to the vault's shared default when absent |

That is the "if the context is empty it is available everywhere" rule, and it
needed no new code in marga — it is the vault's existing behaviour.

## Configuration

```toml
[contexts.personal]
vault_context = "marga"
token_secret  = "MARGA_TOKEN_PERSONAL"
default       = true
allow         = ["nonprofit"]

[contexts.nonprofit]
vault_context = "nonprofit"
token_secret  = "MARGA_TOKEN_NONPROFIT"
```

| Key | Meaning |
|---|---|
| `vault_context` | `worker-env` context holding this context's keys. Empty inherits the proxy's. |
| `token_secret` | Vault key holding this context's bearer token. Resolved lazily. |
| `tokens` | Inline token list. Use for dev only. |
| `token_env` | Environment variable holding the token. |
| `allow` | Other contexts this context's tokens may select explicitly. |
| `default` | Serves callers that present no key. Exactly one. |

A bare string is shorthand for `vault_context`:

```toml
[contexts]
personal  = "marga"
nonprofit = "nonprofit"
```

Store the tokens in the vault, not in the file. The vault is CSV-backed, so
update the CSV and redeploy rather than calling a `set` command:

```bash
# worker-env/worker/secrets/marga.secrets.csv (default context)
echo "MARGA_TOKEN_PERSONAL,$(uv run vault gentoken),Personal marga key,1,$(date -Iseconds)" \
  >> worker/secrets/marga.secrets.csv

# worker-env/worker/secrets/nonprofit.secrets.csv
echo "MARGA_TOKEN_NONPROFIT,$(uv run vault gentoken),Nonprofit marga key,1,$(date -Iseconds)" \
  >> worker/secrets/nonprofit.secrets.csv

npm run deploy     # from ~/codebox/worker-env
```

The token is hashed in marga's memory only; the plaintext never leaves the
vault client.

### Backend scope

Each backend declares how it relates to contexts:

| `context` | Selectable from | Key read from |
|---|---|---|
| `"inherit"` (default) | every context | the **caller's** vault context |
| `"shared"` | every context | the proxy's own vault context |
| any other name | only that context | that context's vault context |

`inherit` is what makes one backend block serve many accounts:

```toml
[backends.opencode]
base_url       = "https://opencode.ai/v1"
api_key_secret = "KEY_OPENCODE"
context        = "inherit"          # default, shown for clarity
models         = ["gpt-5-codex"]
```

Use a pinned backend when the *provider endpoint* differs, not just the key:

```toml
[backends.nonprofit_anthropic]
base_url       = "https://gateway.nonprofit.example/v1"
api_key_secret = "KEY_ANTHROPIC"
context        = "nonprofit"
```

Pinned backends are distinct names, not shadows: the pinned entry appears as
`nonprofit_anthropic/<model>` in `/v1/models` and is invisible to other
contexts. There is no name shadowing, by design — a shadowing rule makes it hard
to tell which account a request will use.

Add `vault_context = "..."` to a backend when its key lives somewhere unrelated
to the requesting context (a shared org key that everyone may use, for example).

## Selecting a context

Three selectors, in precedence order.

### 1. The API key (normal case)

Every chat client sends its key on every request — there is no session and no
login step. OpenAI-style clients send `Authorization: Bearer <key>`, Anthropic
clients send `x-api-key: <key>`. Both are accepted, and both select the context
the key is bound to.

Nothing needs to change client-side: paste the non-profit token into the
non-profit profile.

### 2. `X-Marga-Context: <name>`

For clients that can send custom headers. This is a **selector, never an
authority**: it only narrows/switches between contexts the presented key is
explicitly allowed to use (`allow = [...]`). A key bound to `personal` asking
for `nonprofit` without that permission gets `403`.

### 3. `/c/<name>/v1/...` base URL

For clients that let you change the base URL but not the headers. marga strips
the prefix before routing, so `http://marga:8010/c/nonprofit/v1` behaves exactly
like `http://marga:8010/v1` in the `nonprofit` context. Useful because many
tools offer exactly two fields — base URL and key — and this gives you a second
independent "provider slot".

### When authentication is off

With no tokens configured at all, marga serves the default context and
**ignores** selectors: an unauthenticated caller must not be able to choose a
billing account. To exercise contexts locally, configure a token and export it.

## Isolation guarantees

Anything that could leak one account into another is scoped:

| Concern | Scoped how |
|---|---|
| Upstream keys | resolved per context, cached per context |
| Failover | candidates are filtered to the caller's context before ranking |
| Usage counters | store keys are prefixed `context::backend` |
| `LeastUsed` / `Cheapest` | read the scoped counters, so balancing never crosses accounts |
| `GET /v1/models` | lists only the caller's context |
| `GET /stats` | caller's context by default; `?context=all` aggregates as `context/backend` |
| `GET /health` | default context only, since it is unauthenticated |
| Startup | a missing key disables one backend and is reported, rather than aborting the process |

The failure mode this is designed around: a shared backend dies, failover picks
the next candidate, and that candidate belongs to a different organisation. That
cannot happen — the candidate list is filtered by context before ranking.

## Client recipes

### OpenCode

OpenCode takes per-provider options with a base URL and key. Either replace the
key in the org profile, or register a second provider entry pointing at the same
marga URL with the other token - the provider entry is where the context lives:

```
provider "marga-personal":   baseURL http://127.0.0.1:8010/v1   apiKey sk-personal...
provider "marga-nonprofit":  baseURL http://127.0.0.1:8010/v1   apiKey sk-nonprofit...
```

Pick the profile that matches who is paying for the work. The model id is the
same in both, because routing - not the client - decides which account serves
it.

### LibreChat

`librechat.yaml`, one custom endpoint per context:

```yaml
endpoints:
  custom:
    - name: "Marga Personal"
      apiKey: "${MARGA_TOKEN_PERSONAL}"
      baseURL: "http://127.0.0.1:8010/v1"
      models:
        default: ["claude-sonnet-5-5", "gpt-5-codex"]
        fetch: true
    - name: "Marga Nonprofit"
      apiKey: "${MARGA_TOKEN_NONPROFIT}"
      baseURL: "http://127.0.0.1:8010/v1"
      models:
        default: ["claude-sonnet-5-5"]
        fetch: true
```

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8010
export ANTHROPIC_AUTH_TOKEN=sk-nonprofit...
```

`/v1/messages`, streaming, tools, and `/v1/messages/count_tokens` are all
supported, which is what Claude Code exercises at startup.

### Zed / Cline / Continue

Point the OpenAI-compatible provider at `http://127.0.0.1:8010/v1`. If the tool
only has one key field, either add a second provider entry pointing at the same
URL, or use the path form `http://127.0.0.1:8010/c/nonprofit/v1`.

### HuggingFace

HF's inference router is OpenAI-compatible, so it needs no client-side support —
it is just a backend:

```toml
[backends.hf]
base_url = "https://router.huggingface.co/v1"
api_key_secret = "KEY_HF"
models = []
advertise = ["Qwen/Qwen3-235B-A22B-Instruct-2507"]
```

## API compatibility notes

marga speaks three surfaces and renders errors in the envelope of whichever one
the request arrived on:

| Surface | Endpoints | Error envelope |
|---|---|---|
| OpenAI | `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` | `{"error": {"message", "type", "code"}}` (+ legacy `detail`) |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens` | `{"type": "error", "error": {"type", "message"}}` |

Known limits:

- `count_tokens` returns a 4-characters-per-token **estimate**, not the model's
  real tokenizer. It exists because Claude Code probes it at startup.
- Anthropic fields with no OpenAI equivalent (`top_k`, `thinking`, `metadata`)
  are dropped and logged, never forwarded — OpenAI-compatible backends reject
  unknown body fields with a 400.
- `/v1/completions` (legacy) and `/v1/responses` are not implemented; no client
  in the homestead uses them. Add them the same way as `/v1/embeddings` if that
  changes.
- A pass-through backend (`models = []`) publishes nothing in `/v1/models`
  unless you set `advertise`. Clients accept a model id typed by hand, which is
  how gateways are normally used.
