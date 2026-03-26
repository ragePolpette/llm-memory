# Guida Rapida - llm-memory v2

## 1. Installazione

```bash
cd <project-root>\llm-memory
pip install -e .
```

Il default e' `hash-local`, quindi non serve scaricare un modello esterno per iniziare.

## 2. Configurazione

```bash
copy .env.example .env
```

Default importanti:

- `MEMORY_STORAGE_BACKEND=sqlite`
- `MEMORY_VECTOR_BACKEND=sqlite`
- `MEMORY_SQLITE_PATH=./data/memory.db`
- `MEMORY_IMPORT_EXPORT_BASE_DIR=./data/exchange`
- `EMBEDDING_PROVIDER=hash-local`

## 3. Avvio Runtime

### MCP stdio

```bash
python -m src.mcp_server.server
```

### MCP HTTP locale

```bash
python -m src.mcp_server.http_server
```

Health check:

```bash
curl http://127.0.0.1:8767/health
```

## 4. Test

```bash
pytest -q
```

## 5. Tool MCP Principali

Tool di discovery e amministrazione:

- `memory.about`
- `memory.list_projects`
- `memory.get_project_info`
- `memory.create_project`
- `memory.scope_overview`

Tool operativi:

- `memory.add`
- `memory.search`
- `memory.get`
- `memory.invalidate`
- `memory.promote`
- `memory.reembed`
- `memory.export`
- `memory.import`

## 6. Esempi Payload

### `memory.add`

```json
{
  "agent_id": "local-agent",
  "content": "Il runtime usa SQLite come persistenza locale.",
  "context": "architecture",
  "type": "fact",
  "tier": "tier-2",
  "visibility": "shared"
}
```

### `memory.search`

```json
{
  "agent_id": "local-agent",
  "query": "persistenza locale sqlite",
  "limit": 5,
  "include_project": true,
  "include_workspace": true,
  "include_global": true
}
```

### `memory.reembed`

```json
{
  "agent_id": "local-agent",
  "model_id": "local-hash-v2",
  "dim": 384,
  "activate": true
}
```

## 7. Layout Dati

I dati principali vivono in:

- `./data/memory.db`
- `./data/exchange/`

Non c'e' piu un backend LanceDB o un filesystem markdown come runtime primario.

## 8. Docker

Per il runtime containerizzato usa la guida dedicata in `DOCKER_GUIDE.md`.
