# llm-memory

Local-first MCP memory server that gives agent workflows persistent, governed, scope-aware operational knowledge.

[![CI](https://github.com/ragePolpette/llm-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/ragePolpette/llm-memory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

It keeps two kinds of knowledge per project: raw notes of what happened (episodes) and structured memories that should guide future work (decisions, lessons, facts, constraints, preferences). Recall is hybrid: BM25 for exact identifiers, vectors for meaning. It works the same with any MCP client and any model.

## Overview

Agents forget between sessions. `llm-memory` gives them a durable, inspectable memory per project, with a small tool surface an agent can use correctly without training: note what happened, save what matters, recall before acting, supersede what is no longer true.

## Features

- One SQLite file: episodes, memories with validity intervals and provenance, audit log, vectors, FTS5 indexes
- Hybrid recall: BM25 and vector rankings fused with Reciprocal Rank Fusion, cut to a character budget
- Remote embeddings through any OpenAI-compatible `/v1/embeddings` provider (default Fireworks `qwen3-embedding-8b`); switching provider is configuration and re-embeds everything in the background; recall falls back to text when the provider is down
- Writes rejected when they contain a secret (keys, tokens, passwords, credentials in URLs)
- Every write and its audit row in one transaction; rejected writes audited without their content
- Contradictions close the old memory instead of deleting it, so history stays queryable
- Episodes expire after 60 days unless a memory cites them
- Bearer token authentication for the HTTP transport

## Architecture

```text
MCP tools (src/mcp_server/tools_v2.py)
   |
   v
MemoryServiceV2 (src/store_v2/service.py)   validation, secret rejection, audit payloads
   |
   +--> HybridRetriever (retrieval.py)       BM25 + vectors, RRF, budget
   +--> EmbeddingIndexer (indexer.py)        background embedding queue
   +--> OpenAICompatibleEmbeddings (embeddings.py)
   |
   v
MemoryStore (store.py, schema.py)            SQLite, FTS5, optional sqlite-vec
```

The modules under `src/service/`, `src/storage/`, `src/vectordb/`, `src/embedding/` and `src/interop/` are the v1 runtime. The servers no longer use them; they are kept only for the one-shot migration and will be removed.

## Quick Start

Requires Python 3.11 or newer.

```bash
pip install -e ".[dev]"
cp .env.example .env
pytest -q
```

On Windows `cmd`, use `copy .env.example .env`.

Run as MCP stdio:

```bash
python -m src.mcp_server.server
```

Run the local HTTP transport:

```bash
python -m src.mcp_server.http_server
curl http://127.0.0.1:8767/health
```

`scripts/serve.py` is a cross-platform launcher (`--transport stdio`, `--port 9000`). Stop the server with Ctrl+C.

Container usage is described in [DOCKER_GUIDE.md](DOCKER_GUIDE.md). QUICKSTART.md still describes the v1 tools and will be rewritten.

## Configuration

Configuration is read from environment variables:

- `MEMORY_STORE_PATH=./data/memory-v2.db`
- `MEMORY_SQLITE_PATH=./data/memory.db`: a v1 database found there is migrated once into `MEMORY_STORE_PATH` at first start (ids and audit ids are kept; the v1 file is only read). The migration can also be run by hand: `python -m src.store_v2.migrate_v1 --from data/memory.db --to data/memory-v2.db [--dry-run]`
- `MEMORY_EMBEDDING=on` (`off` for text-only recall)
- `MEMORY_EMBEDDING_BASE_URL=https://api.fireworks.ai/inference/v1`, `MEMORY_EMBEDDING_MODEL=fireworks/qwen3-embedding-8b`, `MEMORY_EMBEDDING_DIM=1024`, `MEMORY_EMBEDDING_API_KEY_ENV=FIREWORKS_API_KEY` (the name of the variable holding the key, not the key), `MEMORY_EMBEDDING_SEND_DIMENSIONS=true`. A local Ollama works too: `MEMORY_EMBEDDING_BASE_URL=http://localhost:11434/v1`, `MEMORY_EMBEDDING_MODEL=nomic-embed-text`, `MEMORY_EMBEDDING_DIM=768`, `MEMORY_EMBEDDING_API_KEY_ENV=`
- `MEMORY_EPISODE_TTL_DAYS=60`
- `MEMORY_SQLITE_VEC=auto` (`pip install -e ".[vec]"` for the vector index; without it vectors are scanned)
- `MCP_MEMORY_HOST=127.0.0.1` and `MCP_MEMORY_PORT=8767`
- `LLM_MEMORY_API_TOKEN` unset (see [Authentication](#authentication))

## Usage

### MCP Tools

- `memory.note`: record what happened (error, test run, observation). Cheap; expires unless cited.
- `memory.save`: save a decision, lesson, fact, constraint or preference. `why` is required for decisions and lessons; `evidence` cites episodes, memories or external references; `supersedes` replaces an older memory.
- `memory.supersede`: close a memory that is no longer true, with a reason.
- `memory.recall`: hybrid search in a project plus the global project, within `budget_chars`; `include_closed` for history.
- `memory.get`: one memory (with sources and validity) or episode by id.
- `memory.projects`: list projects, or create one. Writes to unknown projects are rejected.

Errors come back as `{"ok": false, "error": {"code", "message", "details"}}` with `isError` set. Codes: `invalid_argument`, `not_found`, `conflict`, `secret_rejected`, `unknown_tool`.

### HTTP Endpoints

`/health` is public; `/admin/*` requires the bearer token and is disabled when `LLM_MEMORY_API_TOKEN` is not set:

- `GET /health`
- `GET /admin/audit` (filters: `entry_id`, `action`, `actor`, `reason`, `since`; `after_id=<id>` returns rows with a greater id oldest first, for gap-free polling)
- `GET /admin/projects` (with counts of active and closed memories and episodes)

Audit actions: `note`, `save` (`outcome`: `created`, `duplicate`, `rejected`), `supersede`, `create_project`, `expire_episodes`. Every write payload carries `project_id` at the top level.

## Authentication

Set `LLM_MEMORY_API_TOKEN` to a strong random value (for example `python -c "import secrets; print(secrets.token_urlsafe(32))"`) to enable bearer token authentication on the HTTP transport:

- `/mcp`, `/sse` and `/messages/` require `Authorization: Bearer <token>` when the token is set, and stay open on loopback when it is not.
- `/admin/*` always requires the token and answers `403` when no token is configured.
- `/health` stays public, but only returns diagnostics to authenticated callers when a token is set.
- The server refuses to start when `MCP_MEMORY_HOST` is not a loopback address and no token is configured.

## Security

Memory text is sent to the configured embedding provider: choose it accordingly, or use a local one, or turn embeddings off. Writes containing secrets are rejected before they reach the store or the provider. The provider key is read from the environment at request time and never logged. The HTTP transport binds to loopback by default. See [SECURITY.md](SECURITY.md) for the threat model and how to report vulnerabilities.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
pytest --cov=src --cov-report=term-missing --cov-report=xml -q
```

Coverage is enforced at 82% through `pyproject.toml`. See [CONTRIBUTING.md](CONTRIBUTING.md), [docs/QUALITY_GATE.md](docs/QUALITY_GATE.md), and [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md).

## Project Status

Active development. The core runtime is functional and covered by automated tests. The current direction is hardening the local operational model rather than expanding into a cloud-first memory service. See [CHANGELOG.md](CHANGELOG.md).

Built with AI-assisted workflows; architecture, tradeoffs, integration, review, and validation were directed by the author.

## License

MIT. See [LICENSE](LICENSE).
