# Quickstart

## 1. Installation

```bash
cd <project-root>/llm-memory
pip install -e ".[dev]"
```

The default embedding provider is `hash-local`, so no external model download is needed to get started.

## 2. Configuration

```bash
cp .env.example .env
```

On Windows `cmd`, use `copy .env.example .env`.

Important defaults:

- `MEMORY_STORAGE_BACKEND=sqlite`
- `MEMORY_VECTOR_BACKEND=sqlite`
- `MEMORY_SQLITE_PATH=./data/memory.db`
- `MEMORY_IMPORT_EXPORT_BASE_DIR=./data/exchange`
- `EMBEDDING_PROVIDER=hash-local`

## 3. Running the Runtime

### MCP stdio

```bash
python -m src.mcp_server.server
```

### Local MCP HTTP

```bash
python -m src.mcp_server.http_server
```

Health check:

```bash
curl http://127.0.0.1:8767/health
```

Local read-only admin endpoints:

```bash
curl http://127.0.0.1:8767/admin/summary
curl "http://127.0.0.1:8767/admin/audit?limit=20"
curl "http://127.0.0.1:8767/admin/projects?limit=20"
```

## 4. Test

```bash
pytest -q
```

Verified golden path:

```bash
pytest tests/test_golden_path.py -q
```

## 5. Main MCP Tools

Discovery and administration tools:

- `memory.about`
- `memory.list_projects`
- `memory.get_project_info`
- `memory.create_project`
- `memory.scope_overview`

Operational tools:

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

## 6. Example Payloads

### `memory.add`

```json
{
  "agent_id": "local-agent",
  "content": "The runtime uses SQLite for local persistence.",
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
  "query": "local sqlite persistence",
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
  "content": "Bug: user Y only sees menu X. Fixed by updating the permissions table.",
  "kind": "fix",
  "product_area": "menu",
  "component": "permissions",
  "action_taken": "Updated the permissions table to realign the enabled entries.",
  "outcome": "Menu correct after session refresh.",
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

## 7. Data Layout

Main data lives in:

- `./data/memory.db`
- `./data/exchange/`

The runtime uses SQLite only; there is no LanceDB backend and no markdown filesystem as the primary store.

## 8. Docker

For the containerized runtime see [DOCKER_GUIDE.md](DOCKER_GUIDE.md).

## 9. Development Workflow

For the development and contribution workflow see [CONTRIBUTING.md](CONTRIBUTING.md).

## 10. Fast Memory Distillation

Agent distillation is protected and disabled by default.

To enable it locally:

```bash
export FAST_MEMORY_AGENT_DISTILLATION_ENABLED=true
export FAST_MEMORY_AGENT_DISTILLATION_APPLY_ENABLED=true
```

On Windows `cmd`, use `set NAME=true` instead of `export`.

Minimal workflow:

1. write episodic notes with `memory.log_fast`
2. rank candidates with `memory.rank_fast_candidates`
3. prepare the candidate pack with `memory.prepare_fast_distillation`
4. run the local CLI:

```bash
llm-memory-fast-distill run --agent-id local-cli --reason "distill top candidate" --top-k 1 --harness codex
```

5. apply the resulting JSON output:

```bash
llm-memory-fast-distill apply --input result.json
```
