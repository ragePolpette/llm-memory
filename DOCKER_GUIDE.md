# Docker Guide

This guide matches the active runtime of the repository:

- `SQLite` persistence
- `SQLite` vector store
- local MCP HTTP endpoint
- health endpoint at `/health`

## Build

```bash
cd <project-root>/llm-memory
docker build -t llm-memory:latest .
```

## Start with Docker Compose

The container binds to `0.0.0.0` internally, so the server requires an API token. Compose refuses to start without it:

```bash
export LLM_MEMORY_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose up -d --build
docker compose logs -f llm-memory
docker compose ps
```

The port is published on the host loopback interface only:

- `http://127.0.0.1:8767/health` (public)
- `http://127.0.0.1:8767/mcp` (requires `Authorization: Bearer $LLM_MEMORY_API_TOKEN`)
- `http://127.0.0.1:8767/admin/*` (requires the token)

The image runs as the unprivileged `app` user (uid 10001). On Linux, make sure the host `./data` directory is writable by that uid, for example `sudo chown -R 10001:10001 ./data`.

## Stop and Restart

```bash
docker compose down
docker compose restart llm-memory
```

## Persistence

The host `./data` directory is mounted at `/data` in the container.

Main files:

- `/data/memory.db`
- `/data/exchange/`

## Active Compose Configuration

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
  - LLM_MEMORY_API_TOKEN=${LLM_MEMORY_API_TOKEN:?set LLM_MEMORY_API_TOKEN to a strong random token}

ports:
  - "127.0.0.1:8767:8767"
```

## Health Check

```bash
curl http://127.0.0.1:8767/health
```

Expected response:

```json
{
  "status": "ok",
  "server": "llm-memory",
  "api": "v2",
  "mcp_sse_enabled": false
}
```

## Local MCP HTTP

The container runs the HTTP server:

```bash
python -m src.mcp_server.http_server
```

For local use this is more practical than stdio inside Docker.

## Troubleshooting

### The container does not start

```bash
docker compose logs llm-memory
docker compose build --no-cache
```

### The health endpoint does not respond

Check that the port is free and that the container exposes `8767`.

```bash
docker compose ps
docker compose logs llm-memory
```

### Reset local data

Warning: removing the `./data` bind mount deletes all persisted data.

```bash
docker compose down
```

Then remove the `data/` directory manually, only if you want a full reset.

## Note

- This Docker configuration is intended for local use or a small trusted team.
- It is not intended as a public, internet-facing deployment. The bearer token is a single shared secret and there is no TLS; see [SECURITY.md](SECURITY.md).
- To use MCP over stdio, run the runtime directly outside the container.
