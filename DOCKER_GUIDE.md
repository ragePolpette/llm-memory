# Guida Docker - llm-memory v2

Questa guida e' allineata al runtime attivo del repo:

- persistenza `SQLite`
- vector store `SQLite`
- endpoint MCP HTTP locale
- health endpoint su `/health`

## Build

```bash
cd <project-root>\llm-memory
docker build -t llm-memory:latest .
```

## Avvio con Docker Compose

```bash
docker compose up -d --build
docker compose logs -f llm-memory
docker compose ps
```

Il container espone:

- `http://127.0.0.1:8767/health`
- `http://127.0.0.1:8767/mcp`

## Stop e Restart

```bash
docker compose down
docker compose restart llm-memory
```

## Persistenza

La directory `./data` dell'host viene montata su `/data` nel container.

File principali:

- `/data/memory.db`
- `/data/exchange/`

## Configurazione Attiva Nel Compose

```yaml
environment:
  - MEMORY_STORAGE_BACKEND=sqlite
  - MEMORY_VECTOR_BACKEND=sqlite
  - MEMORY_SQLITE_PATH=/data/memory.db
  - MEMORY_IMPORT_EXPORT_BASE_DIR=/data/exchange
  - EMBEDDING_PROVIDER=hash-local
  - EMBEDDING_MODEL=local-hash-v1
  - EMBEDDING_DIM=384
  - MEMORY_ALLOW_OUTBOUND_NETWORK=false
  - MCP_MEMORY_HOST=0.0.0.0
  - MCP_MEMORY_PORT=8767
  - MCP_MEMORY_SSE_ENABLED=false
```

## Health Check

```bash
curl http://127.0.0.1:8767/health
```

Risposta attesa:

```json
{
  "status": "ok",
  "server": "llm-memory",
  "api": "v2",
  "mcp_sse_enabled": false
}
```

## MCP HTTP Locale

Il container usa il server HTTP:

```bash
python -m src.mcp_server.http_server
```

Per uso locale questo e' piu pratico dello stdio dentro Docker.

## Troubleshooting

### Il container non parte

```bash
docker compose logs llm-memory
docker compose build --no-cache
```

### Health endpoint non risponde

Verifica che la porta sia libera e che il container abbia esposto `8767`.

```bash
docker compose ps
docker compose logs llm-memory
```

### Reset dati locali

Attenzione: questa operazione elimina i dati persistiti nel bind mount `./data`.

```bash
docker compose down
```

Poi rimuovi manualmente la directory `data/` solo se vuoi davvero un reset completo.

## Note

- Questa configurazione Docker e' pensata per uso locale o team ristretto.
- Non e' pensata come deployment pubblico internet-facing.
- Se vuoi usare stdio MCP, esegui il runtime direttamente fuori dal container.
