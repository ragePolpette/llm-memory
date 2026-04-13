# Guida Rapida - llm-memory v2

## 1. Installazione

```bash
cd <project-root>\llm-memory
pip install -e ".[dev]"
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

Admin read-only locale:

```bash
curl http://127.0.0.1:8767/admin/summary
curl "http://127.0.0.1:8767/admin/audit?limit=20"
curl "http://127.0.0.1:8767/admin/projects?limit=20"
```

## 4. Test

```bash
pytest -q
```

Golden path verificato:

```bash
pytest tests/test_golden_path.py -q
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
- `memory.log_fast`
- `memory.list_fast`
- `memory.get_fast`
- `memory.rank_fast_candidates`
- `memory.prepare_fast_distillation`
- `memory.apply_fast_distillation`
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

### `memory.log_fast`

```json
{
  "agent_id": "local-agent",
  "content": "Bug: l'utente Y vede solo il menu X. Fix applicato aggiornando la tabella permessi.",
  "kind": "fix",
  "product_area": "menu",
  "component": "permissions",
  "action_taken": "Aggiornata la tabella permessi per riallineare le voci abilitate.",
  "outcome": "Menu corretto dopo refresh sessione.",
  "generalizable": true
}
```

### `memory.rank_fast_candidates`

```json
{
  "agent_id": "local-agent",
  "limit": 5
}
```

### `memory.prepare_fast_distillation`

```json
{
  "agent_id": "local-agent",
  "reason": "Distill top candidate into strong project memory",
  "top_k": 1
}
```

### `memory.apply_fast_distillation`

```json
{
  "agent_id": "local-agent",
  "reason": "Apply reviewed distillation output",
  "dry_run": true,
  "payload": {
    "decisions": []
  }
}
```

### `memory.export`

```json
{
  "agent_id": "local-agent",
  "path": "golden-path.jsonl",
  "format": "jsonl"
}
```

### `memory.import`

```json
{
  "agent_id": "local-agent",
  "path": "golden-path.jsonl",
  "format": "jsonl"
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

## 9. Development Workflow

Per il workflow di sviluppo e contribution usa `CONTRIBUTING.md`.

## 10. Distillazione Fast Memory

La distillazione agentica e' protetta e disabilitata di default.

Per abilitarla localmente:

```bash
set FAST_MEMORY_AGENT_DISTILLATION_ENABLED=true
set FAST_MEMORY_AGENT_DISTILLATION_APPLY_ENABLED=true
```

Workflow minimo:

1. scrivi note episodiche con `memory.log_fast`
2. ordina i candidati con `memory.rank_fast_candidates`
3. prepara il candidate pack con `memory.prepare_fast_distillation`
4. lancia la CLI locale:

```bash
llm-memory-fast-distill run --agent-id local-cli --reason "distill top candidate" --top-k 1 --harness codex
```

5. applica l'output JSON risultante:

```bash
llm-memory-fast-distill apply --input result.json
```
