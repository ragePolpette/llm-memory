# Dockerfile per LLM Memory MCP Server
FROM python:3.11-slim

# Metadata
LABEL maintainer="LLM Memory"
LABEL description="Shared memory system for multi-agent AI with MCP server"

# Variabili di ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Directory di lavoro
WORKDIR /app

# Copia file di configurazione
COPY pyproject.toml .
COPY README.md .

# Installa dipendenze
RUN pip install --upgrade pip && \
    pip install -e .

# Copia il codice sorgente
COPY src/ ./src/
COPY scripts/ ./scripts/

# Crea directory per dati persistenti
RUN mkdir -p /data/memories /data/lancedb /data/logs

# Pre-download del modello embedding (opzionale, commentato per build più veloce)
# RUN python scripts/download_model.py

# Variabili di ambiente di default (possono essere sovrascritte)
ENV MEMORY_STORAGE_DIR=/data/memories \
    LANCEDB_DIR=/data/lancedb \
    EMBEDDING_MODEL=intfloat/multilingual-e5-small \
    EMBEDDING_DIM=384 \
    INDEXING_MODE=sync

# Espone porta per MCP (se in futuro si usa HTTP invece di stdio)
# EXPOSE 8080

# Volume per persistenza dati
VOLUME ["/data"]

# Health check (opzionale)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#   CMD python -c "import sys; sys.exit(0)"

# Comando di avvio
CMD ["python", "-m", "src.mcp_server.server"]
