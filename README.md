# llm-memory

`llm-memory` is a local-first MCP memory system for persistent operational knowledge shared across agent workflows.

It is designed for reusable facts, decisions, assumptions, invalidations, and project-level conventions. It is not a repository code-context retriever. The focus is durable memory with governance, privacy controls, auditability, and explicit scope boundaries.

## What It Does

- stores persistent operational memory locally with SQLite-backed metadata and vectors
- supports tiered memory for short-lived, project, and curated long-term knowledge
- provides explicit scope composition across `project`, `workspace`, and `global`
- exposes MCP tools for add, search, promote, invalidate, import, export, and re-embed workflows
- keeps audit trails and persistence policies close to the write path

## Why It Exists

Most agent workflows are good at short-term context but weak at durable memory.

`llm-memory` is built to give local agents a persistent memory layer that remains:

- inspectable
- governed
- portable
- local-first

The project is intended for serious workstation or small-team environments where memory quality, privacy, and explicit control matter more than “chat history” convenience.

## Core Concepts

- Tiering: session-like, project-level, and curated long-term memory buckets
- Scope hierarchy: retrieval can compose `project`, `workspace`, and `global`
- Governance: promotion, invalidation, deduplication, and audit trail
- Local-first operation: no cloud service required for the default setup
- Import/export: deterministic interchange through JSONL, Markdown, and SQLite-backed data

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

## MCP Surface

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
- `memory.search`
- `memory.get`
- `memory.invalidate`
- `memory.promote`
- `memory.reembed`
- `memory.export`
- `memory.import`

## Local Run

Quick setup:

```bash
pip install -e ".[dev]"
copy .env.example .env
pytest -q
```

Run as MCP stdio:

```bash
python -m src.mcp_server.server
```

Run local HTTP transport:

```bash
python -m src.mcp_server.http_server
```

Useful local endpoints:

- `GET /health`
- `GET /admin/summary`
- `GET /admin/audit`
- `GET /admin/projects`
- `GET /admin/fast-memory`
- `GET /admin/fast-memory/{entry_id}`

## Runtime Characteristics

- default storage backend: SQLite
- default vector backend: SQLite
- default embedding path: local hash-based embedding, with optional external local embedding providers
- optional multi-project mode with explicit project registration
- optional encryption for sensitive payloads
- outbound network blocked by default in the intended local-first setup

## Project Status

This repository is in active development, but the core runtime is already functional and covered by automated tests. The current direction is focused on hardening the local operational model rather than expanding into a cloud-first memory service.

## Documentation

- [QUICKSTART.md](QUICKSTART.md)
- [DOCKER_GUIDE.md](DOCKER_GUIDE.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CHANGELOG.md](CHANGELOG.md)
- [docs/QUALITY_GATE.md](docs/QUALITY_GATE.md)
- [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md)

## Development Process

Built with AI-assisted workflows, while architecture, tradeoffs, integration, review, and validation were directed by the author.
