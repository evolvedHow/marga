# marga Repository Structure

```
marga/                         # repository root
├── .git/                      # Git repository
├── .gitignore                 # Python/IDE/local-config ignores
│
├── LICENSE                    # Apache 2.0
├── NOTICE                     # Copyright notice
│
├── README.md                  # Project overview
├── QUICKSTART.md              # 60-second start guide
├── COMPARISON.md              # vs margAI (the full framework)
├── SUMMARY.md                 # Project summary
├── INDEX.md                   # Complete index
├── TRANSFORMATION.md          # The story: 10,389 -> 509 lines
├── STRUCTURE.md               # This file
├── docs/
│   └── contexts.md            # Request contexts: design + client recipes
│
├── pyproject.toml             # Package config (hatchling, `marga` entry point)
├── uv.lock                    # Dependency lock
├── marga.toml.example         # Config template (copy to marga.toml; untracked)
│
├── Dockerfile                 # Container image (binds 127.0.0.1:8010)
├── docker-compose.yml         # Compose service, env from env_file
├── worker.py                  # Cloudflare Worker entry point
├── wrangler.toml              # Cloudflare Worker config
│
├── marga/                     # THE PYTHON PACKAGE (import marga)
│   ├── __init__.py            # Package exports
│   ├── _version.py            # Version string
│   ├── _util.py               # Shared helpers
│   ├── cli.py                 # `marga` command line (serve, ...)
│   ├── config.py              # marga.toml / JSON backend loading
│   ├── contexts.py            # Request contexts + token resolution
│   ├── vault.py               # worker-env vault client
│   ├── models.py              # Pydantic request/response models
│   ├── proxy.py               # Proxy + FastAPI app
│   ├── anthropic.py           # Anthropic Messages API adapter
│   ├── middleware.py          # Context auth + legacy bearer auth + CORS
│   ├── router.py              # Smart routing + usage tracking
│   ├── strategies.py          # Routing strategies
│   ├── availability.py        # Runtime detection + backend availability
│   ├── store.py               # Usage store (+ per-context scoping)
│   ├── hooks.py               # Before/after hook system
│   ├── transport.py           # HTTP backend communication
│   └── errors.py              # Error types + API error envelopes
│
├── examples/
│   └── basic.py               # Working example
└── tests/
    ├── test_marga.py          # Core routing, hooks, accounting
    ├── test_contexts.py       # Context isolation, selectors, tokens
    └── test_compat.py         # OpenAI / Anthropic / embeddings surfaces
```

## Notes

- The inner `marga/` directory is the importable package. `pyproject.toml`
  (`packages = ["marga"]`), the Dockerfile (`COPY marga ./marga`) and the
  `marga` console script all depend on it; it is not a duplicate of the root.
- `marga.toml` is local configuration and is gitignored. Secrets are never
  stored there: backends reference vault keys through `api_key_secret`.
- Generated directories (`.venv/`, `__pycache__/`, `.pytest_cache/`,
  `.ruff_cache/`) are gitignored.
