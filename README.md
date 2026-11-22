# marga

**मार्ग (mārga)** - Sanskrit for "path" or "route"

**Intelligent LLM routing proxy** - route requests across multiple providers with automatic failover, cost optimization, and dual OpenAI/Anthropic APIs.

## What it does

- 🔄 **Smart routing** - least-used, cheapest, round-robin, or custom strategies
- 🌐 **Dual surface** - OpenAI `/v1/chat/completions` + Anthropic `/v1/messages`
- 🔐 **Vault integration** - API keys from worker-env vault, never in config files
- 🚀 **Dual deploy** - same code runs locally and on Cloudflare Workers
- 📡 **Backend availability** - auto-probes backends, filters unreachable ones
- ⚡ **Automatic failover** - tries next backend when primary fails
- 🪝 **Prompt hooks** - inject RAG context, modify responses
- 📊 **Usage tracking** - per-backend calls, tokens, cost accounting

## Quick Start

### 1. Install

```bash
uv pip install -e .
# or
pip install -e .
```

### 2. Configure

Copy the example config:

```bash
cp marga.toml.example marga.toml
```

Edit `marga.toml` with your backends:

```toml
[backends.llama]
base_url = "http://127.0.0.1:8080/v1"  # Local llama.cpp
models = ["qwen2.5-coder:7b"]

[backends.anthropic]
base_url = "https://api.anthropic.com/v1"
api_key_secret = "KEY_ANTHROPIC"  # Read from vault
models = ["claude-sonnet-5-5"]

[backends.openai]
base_url = "https://api.openai.com/v1"
api_key_secret = "KEY_OPENAI"
models = ["gpt-4o-mini"]
```

### 3. Set vault credentials

```bash
# Create credentials file
mkdir -p ~/docker/stack
cat > ~/docker/stack/.env << 'EOF'
VAULT_URL=https://worker-env-vault.your-domain.workers.dev
VAULT_TOKEN=your-vault-token
EOF
chmod 600 ~/docker/stack/.env
```

### 4. Run

```bash
# CLI
marga serve --config marga.toml --port 8010

# Or programmatically
python -c "
from marga import Proxy
proxy = Proxy.from_toml('marga.toml')
proxy.serve(port=8010)
"
```

## Runtime Detection

marga auto-detects whether it's running locally or in the cloud:

- **Local runtime**: All backends available, loopback addresses work
- **Cloud runtime** (Cloudflare Workers): Loopback backends auto-disabled

Detection:
1. `MARGA_RUNTIME` env override (`local` or `cloud`)
2. `sys.platform == "emscripten"` → cloud
3. `import js` succeeds → cloud (pyodide)
4. Default → local

Backend availability:
- **Local**: HTTP probes with 2s timeout, 60s TTL cache
- **Cloud**: Loopback addresses marked unreachable (no probe)

Check status: `GET http://localhost:8010/health`

```json
{
  "status": "ok",
  "version": "0.2.0",
  "runtime": "local",
  "backends": [
    {
      "name": "llama",
      "url": "http://127.0.0.1:8080/v1",
      "enabled": true,
      "loopback": true,
      "reachable": false,
      "reason": "connection failed: ConnectError"
    },
    {
      "name": "anthropic",
      "url": "https://api.anthropic.com/v1",
      "enabled": true,
      "loopback": false,
      "reachable": true,
      "reason": "backend responding"
    }
  ]
}
```

## Deployment

### Local (Docker)

```bash
cd ~/codebox/marga

# Build and run
docker compose up -d

# Check logs
docker logs -f marga

# Access at http://127.0.0.1:8010
```

### Cloudflare Workers

```bash
# Set secrets (one-time)
wrangler secret put VAULT_URL
wrangler secret put VAULT_TOKEN
wrangler secret put MARGA_AUTH_TOKEN  # Optional: bearer token for public access

# Set backends config as JSON
wrangler secret put MARGA_BACKENDS
# Paste: {"anthropic":{"base_url":"https://api.anthropic.com/v1","api_key_secret":"KEY_ANTHROPIC",...}}

# Deploy
uv run --with cloudflare-workers pywrangler deploy

# Access at https://marga.your-subdomain.workers.dev
```

Worker automatically:
- Detects cloud runtime (`MARGA_RUNTIME=cloud`)
- Disables loopback backends (127.0.0.1, localhost)
- Routes httpx over JS fetch (pyodide.http patch)
- Reads vault credentials from wrangler secrets

## API Endpoints

### OpenAI-compatible

```bash
# Chat completions
curl -X POST http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-5-5",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'

# List models
curl http://localhost:8010/v1/models
```

### Anthropic-compatible

```bash
# Messages API
curl -X POST http://localhost:8010/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-marga-auth-token" \
  -d '{
    "model": "claude-sonnet-5-5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Streaming works for both endpoints (set `"stream": true`).

### Metadata

```bash
# Health check
curl http://localhost:8010/health

# Usage statistics
curl http://localhost:8010/stats?period=month

# Proxy info
curl http://localhost:8010/
```

## Authentication

Set `MARGA_AUTH_TOKEN` to require bearer authentication:

```bash
export MARGA_AUTH_TOKEN=your-secret-token
marga serve
```

Clients send token as:
- `Authorization: Bearer <token>` (OpenAI style)
- `x-api-key: <token>` (Anthropic style)

Public endpoints (`/health`, `/docs`) always allowed.

## CORS

For web chat interfaces:

```bash
# Allow all origins (dev only)
export ALLOWED_ORIGINS="*"

# Allow specific origins
export ALLOWED_ORIGINS="https://chat.example.com,http://localhost:3000"

marga serve
```

## Routing Strategies

```python
from marga import Proxy, least_used, round_robin, cheapest, budget

proxy = Proxy.from_toml("marga.toml")

# Built-in strategies
proxy.route(least_used())  # Fewest calls
proxy.route(round_robin())  # Even distribution
proxy.route(cheapest())  # Lowest cost_per_1k_tokens

# Budget limits
proxy.route(
    budget(
        [
            {"backend": "openai", "limit": 1000000, "period": "month"},
        ]
    )
)


# Custom routing
@proxy.route
async def custom(ctx):
    if ctx.estimated_tokens < 1000:
        return "llama/qwen2.5-coder:7b"  # Fast local
    return "anthropic/claude-haiku-5-5"  # Cloud fallback
```

## Hooks

```python
@proxy.before
async def add_rag_context(ctx):
    """Inject RAG context before LLM."""
    ctx.messages.insert(0, {"role": "system", "content": f"Context: {retrieve_relevant_docs(ctx)}"})


@proxy.after
async def track_usage(ctx, response):
    """Log usage after LLM."""
    usage = response.get("usage", {})
    print(f"[{ctx.model}] {usage.get('total_tokens', 0)} tokens")
    return response


@proxy.error
async def fallback_handler(ctx, error):
    """Handle backend errors."""
    if isinstance(error, TimeoutError):
        return {"error": "Backend timeout, try again"}
```

## Integration

### LibreChat

Add to `librechat.yaml`:

```yaml
endpoints:
  custom:
    - name: 'marga'
      apiKey: "not-needed"
      baseURL: "http://127.0.0.1:8010/v1"
      models:
        fetch: true
      modelDisplayLabel: 'marga (Routing Proxy)'
```

### open-notebook

Set in `docker-compose.yml`:

```yaml
environment:
  - OPENAI_COMPATIBLE_BASE_URL=http://127.0.0.1:8010/v1
```

## Architecture

```
Request (OpenAI or Anthropic format)
    ↓
[Auth Middleware] ← Bearer token check (optional)
    ↓
[CORS] ← Origin validation (optional)
    ↓
[Anthropic Translation] ← If /v1/messages, translate to OpenAI format
    ↓
[Before Hooks] ← RAG injection, prompt modification
    ↓
[Router] ← Strategy selects backend, availability check
    ↓
[Failover Candidates] ← Ordered list with unreachable filtered
    ↓
[Backend HTTP] ← httpx.AsyncClient (fetch in cloud)
    ↓
[After Hooks] ← Response processing, logging
    ↓
[Anthropic Translation] ← If /v1/messages, translate response back
    ↓
Response
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest -v

# Format
uv run ruff format

# Lint
uv run ruff check --fix

# Local server with reload
marga serve --reload --port 8010
```

## Size

- **~3,000 lines** (from 500-line proof-of-concept)
- **8 modules**: proxy, router, strategies, transport, vault, availability, middleware, anthropic
- **Core deps**: httpx, fastapi, pydantic
- **Optional**: uvicorn (local), cloudflare-workers (deploy)

## License

[Your License Here]
