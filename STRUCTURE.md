# marga Repository Structure

```
marga/
├── .git/                      # Git repository
├── .gitignore                 # Python/IDE ignores
│
├── LICENSE                    # Apache 2.0
├── NOTICE                     # Copyright notice
│
├── README.md                  # Project overview
├── QUICKSTART.md              # 60-second start guide
├── COMPARISON.md              # vs margAI (the full framework)
├── SUMMARY.md                 # Project summary
├── INDEX.md                   # Complete index
├── TRANSFORMATION.md          # The story: 10,389 → 509 lines
│
├── pyproject.toml             # Package config
├── uv.lock                    # Dependency lock
│
├── __init__.py          (14)  # Package exports
├── proxy.py            (208)  # Main proxy + FastAPI
├── router.py           (125)  # Smart routing
├── hooks.py             (70)  # Hook system
├── transport.py         (92)  # HTTP transport
│                       ─────
│                        509   CORE LINES
│
├── example.py                 # Complete working example
└── test_marga.py              # 7 passing tests
```

## File Sizes

```
Total repository: ~240KB (excluding .git, .venv)
Core source code:  ~15KB (5 Python files)
Documentation:     ~50KB (6 markdown files)
```

## Line Counts

```
Core source:        509 lines
Tests:              128 lines
Example:             92 lines
Documentation:    2,000+ lines
```

## Quick Navigation

**New here?**
1. [README.md](README.md) - What is marga?
2. [QUICKSTART.md](QUICKSTART.md) - Run it in 60 seconds
3. [example.py](example.py) - See it in action

**Comparing to margAI?**
1. [TRANSFORMATION.md](TRANSFORMATION.md) - The full story
2. [COMPARISON.md](COMPARISON.md) - Detailed comparison

**Building something?**
1. [INDEX.md](INDEX.md) - API reference
2. [test_marga.py](test_marga.py) - Usage patterns
3. Source files - Only 509 lines total!

## Dependencies

```toml
dependencies = [
    "fastapi>=0.100.0",
    "httpx>=0.27.0",
    "uvicorn>=0.20.0",
]
```

That's it. No pydantic restrictions, no multipart, no extras.

## Installation

```bash
# Clone
git clone <your-repo-url> marga
cd marga

# Install
uv pip install -e .

# Test
pytest test_marga.py -v

# Run example
python example.py
```

## Philosophy

**Small is beautiful.**

- 509 lines of core code
- 3 dependencies
- 2 features (routing + hooks)
- 1 purpose (transparent LLM proxy)
- 0 bloat

**मार्ग (mārga)** - The path is clear.
