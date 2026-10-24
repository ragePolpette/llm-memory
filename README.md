# llm-memory

Local-first MCP memory server that gives agent workflows persistent, governed, scope-aware operational knowledge.

[![CI](https://github.com/ragePolpette/llm-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/ragePolpette/llm-memory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

It is designed for reusable facts, decisions, assumptions, invalidations, and project-level conventions. It is not a repository code-context retriever. The focus is durable memory with governance, privacy controls, auditability, and explicit scope boundaries.

## Overview

Most agent workflows handle short-term context well but have no durable memory. `llm-memory` provides a persistent layer that stays inspectable, governed, portable, and local-first. It targets workstation or small-team use, where memory quality, privacy, and explicit control matter more than chat-history convenience.

## Features

- Local persistence with SQLite-backed metadata and vectors
- Tiered memory: strong memory for durable knowledge, fast memory for episodic operational notes
- Scope hierarchy: retrieval can compose `project`, `workspace`, and `global`
- Governance: promotion, invalidation, deduplication, and an audit trail close to the write path
- Optional encryption for sensitive payloads (install the `crypto` extra) and a configurable privacy policy
- Import/export through JSONL, Markdown (`memory.md`), and SQLite-backed data, limited to the entries the caller can read
- Optional bearer token authentication for the HTTP transport
- No cloud service required for the default setup

## Architecture

```text
MCP tool surface
   |
   v
MemoryService
   |
   +--> persistence policy
   +--> importance scoring
   +--> privacy controls
   +--> audit trail
   |
   +--> SQLite store
   +--> SQLite vector store
   +--> embedding provider
```

Main runtime areas:

- `src/mcp_server/`: MCP tool surface and transport entrypoints
- `src/service/`: persistence policy, scoring, and orchestration logic
- `src/storage/`: SQLite-backed record persistence
- `src/vectordb/`: local vector storage
- `src/security/`: privacy and optional encryption helpers
- `src/interop/`: import/export helpers such as `memory.md`

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

`scripts/serve.py` is a cross-platform launcher that applies the local-only defaults (hash embeddings, SQLite, outbound network blocked) and works the same on Linux, macOS and Windows:

```bash
python scripts/serve.py                     # HTTP on 127.0.0.1:8767
python scripts/serve.py --transport stdio
python scripts/serve.py --port 9000
```

Stop the server with Ctrl+C. The former Windows `.bat` launchers have been removed.

To enable at-rest encryption of sensitive payloads, install the optional extra: `pip install -e ".[crypto]"`.

More detail, including request payload examples, is in [QUICKSTART.md](QUICKSTART.md). Container usage is described in [DOCKER_GUIDE.md](DOCKER_GUIDE.md).

## Configuration

Configuration is read from environment variables. [`.env.example`](.env.example) lists every key with its default. Important defaults:

- `MEMORY_STORAGE_BACKEND=sqlite` and `MEMORY_VECTOR_BACKEND=sqlite`
- `MEMORY_SQLITE_PATH=./data/memory.db`
- `MEMORY_IMPORT_EXPORT_BASE_DIR=./data/exchange`
- `EMBEDDING_PROVIDER=hash-local` (local hash-based embedding, no model download)
- `MEMORY_ALLOW_OUTBOUND_NETWORK=false`
- `MEMORY_ENCRYPTION_ENABLED=false`
- `MEMORY_MULTI_PROJECT_ENABLED=false`
- `MCP_MEMORY_HOST=127.0.0.1` and `MCP_MEMORY_PORT=8767`
- `LLM_MEMORY_API_TOKEN` unset (see [Authentication](#authentication))

## Usage

### MCP Tools

Discovery and administration:

- `memory.about`
- `memory.list_projects`
- `memory.get_project_info`
- `memory.create_project`
- `memory.scope_overview`

Operational memory tools:

- `memory.add`
- `memory.log_fast`
- `memory.list_fast`
- `memory.get_fast`
- `memory.rank_fast_candidates`
- `memory.prepare_fast_distillation`
- `memory.apply_fast_distillation`
- `memory.summarize_fast`
- `memory.discard_fast`
- `memory.promote_fast`
- `memory.search`
- `memory.get`
- `memory.invalidate`
- `memory.promote`
- `memory.reembed`
- `memory.export`
- `memory.import`

### Fast-memory CLI

Prepare a fast-memory distillation pack from the CLI:

```bash
llm-memory-fast-distill prepare --agent-id local-cli --reason "prepare top candidate" --top-k 1
```

Launch a local harness with the fixed prompt:

```bash
llm-memory-fast-distill run --agent-id local-cli --reason "distill top candidate" --top-k 1 --harness codex
```

Apply a distillation result from a local JSON file:

```bash
llm-memory-fast-distill apply --input result.json
```

Tool `limit` arguments are capped at 100 (50 for `search_inference_memory`).

### HTTP Endpoints

Local admin and health endpoints. `/health` is always public; the `/admin/*` endpoints require the bearer token and are disabled when `LLM_MEMORY_API_TOKEN` is not set:

- `GET /health`
- `GET /admin/summary`
- `GET /admin/audit` (filters: `entry_id`, `action`, `actor`, `reason`, `since`; `after_id=<id>` returns rows with a greater id oldest first, for gap-free polling)
- `GET /admin/projects`
- `GET /admin/fast-memory`
- `GET /admin/fast-memory/candidates`
- `GET /admin/fast-memory/distillation/runs`
- `GET /admin/fast-memory/distillation/runs/{run_id}`
- `GET /admin/fast-memory/{entry_id}`
- `POST /admin/fast-memory/distillation/prepare`
- `POST /admin/fast-memory/distillation/apply`

### Fast Memory Flow

The intended fast-memory workflow is:

1. write episodic notes with `memory.log_fast`
2. inspect ranked candidates with `memory.rank_fast_candidates`
3. prepare a protected distillation pack with `memory.prepare_fast_distillation`
4. run a local harness or CLI against the prepared pack
5. apply the resulting JSON with `memory.apply_fast_distillation`

The distillation flow is disabled by default.
Enable it explicitly with:

- `FAST_MEMORY_AGENT_DISTILLATION_ENABLED=true`
- `FAST_MEMORY_AGENT_DISTILLATION_APPLY_ENABLED=true`

## Authentication

Set `LLM_MEMORY_API_TOKEN` to a strong random value (for example `python -c "import secrets; print(secrets.token_urlsafe(32))"`) to enable bearer token authentication on the HTTP transport:

- `/mcp`, `/sse` and `/messages/` require `Authorization: Bearer <token>` when the token is set, and stay open on loopback when it is not.
- `/admin/*` always requires the token and answers `403` when no token is configured.
- `/health` stays public, but only returns diagnostics to authenticated callers when a token is set.
- The server refuses to start when `MCP_MEMORY_HOST` is not a loopback address and no token is configured.

## Embedding Versions

Each stored vector is tagged with the embedding version that produced it. At startup the server compares the active version with the configured provider. When they differ it logs an `embedding_version_mismatch` warning and `memory.search` is refused with an explicit error until you run `memory.reembed` without `model_id`/`dim`, which re-embeds existing entries with the configured provider and activates it. Writes keep working in the meantime. A write is rejected (`reason: embedding-failed`) when the provider cannot produce a vector, so every stored entry is searchable.

## Security

Outbound network access is blocked by default, the HTTP transport binds to loopback by default, and import/export paths are confined to `MEMORY_IMPORT_EXPORT_BASE_DIR`. Exports only contain entries the caller is allowed to read. The activity log on stderr records content hashes and lengths, never content. See [SECURITY.md](SECURITY.md) for the threat model and how to report vulnerabilities.

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
