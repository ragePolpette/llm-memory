# Docker image for the HTTP MCP runtime.
FROM python:3.11-slim

LABEL maintainer="LLM Memory"
LABEL description="Local-first MCP memory service with SQLite persistence"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN pip install --upgrade pip && \
    pip install -e .

RUN mkdir -p /data/exchange

ENV MEMORY_STORAGE_BACKEND=sqlite \
    MEMORY_VECTOR_BACKEND=sqlite \
    MEMORY_SQLITE_PATH=/data/memory.db \
    MEMORY_IMPORT_EXPORT_BASE_DIR=/data/exchange \
    EMBEDDING_PROVIDER=hash-local \
    EMBEDDING_MODEL=local-hash-v1 \
    EMBEDDING_DIM=384 \
    MEMORY_ALLOW_OUTBOUND_NETWORK=false \
    MCP_MEMORY_HOST=0.0.0.0 \
    MCP_MEMORY_PORT=8767 \
    MCP_MEMORY_SSE_ENABLED=false

VOLUME ["/data"]

EXPOSE 8767

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import json, urllib.request; data = json.load(urllib.request.urlopen('http://127.0.0.1:8767/health', timeout=5)); raise SystemExit(0 if data.get('status') == 'ok' else 1)"

CMD ["python", "-m", "src.mcp_server.http_server"]
