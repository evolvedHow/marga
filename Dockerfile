# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY marga ./marga
COPY README.md ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Expose port (8010 avoids a conflict with the service on 8000)
EXPOSE 8010

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD /app/.venv/bin/python -c "import httpx; httpx.get('http://127.0.0.1:8010/health', timeout=2.0).raise_for_status()" || exit 1

# Run marga (bind to loopback per homelab networking rule)
CMD ["uv", "run", "marga", "serve", "--host", "127.0.0.1", "--port", "8010"]
