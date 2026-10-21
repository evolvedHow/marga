# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY marga ./marga
COPY README.md ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Expose port (8010 to avoid conflict with SurrealDB on 8000)
EXPOSE 8010

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:8010/health', timeout=2.0)" || exit 1

# Run marga
CMD ["uv", "run", "marga", "serve", "--host", "0.0.0.0", "--port", "8010"]
