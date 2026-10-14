# Changelog

All meaningful user-visible changes should be summarized here when cutting a public-ready release.

## Unreleased

### Fixed

- HTTP server no longer crashes at startup: the lifespan logged startup diagnostics through the standard library logger with keyword arguments and raised `TypeError`.
- The stdio server writes logs to stderr only; stdout is reserved for JSON-RPC messages.
- `memory.export` only exports entries the caller can read. Private entries of other agents are no longer included, and `format=sqlite` now writes a standalone database with only those entries (plus their embeddings and the project record) instead of copying the whole database, audit log and fast memory included.
- SQLite connections use a 30 second busy timeout and enable foreign keys on every connection, not only at schema creation.
- An entry and its vector are written in a single transaction, so a failure while storing the vector no longer leaves an entry behind.
- The `sentence-transformers` embedding fingerprint no longer includes the vector dimension, which changed once the model was loaded and produced a different embedding version between calls.
- Writes no longer silently switch the active embedding version to the current provider.

### Changed

- Behavior change: `memory.add` (and imports) reject a write with `reason: embedding-failed` when the embedding provider fails or returns an empty vector, instead of storing an entry without a vector. `memory.promote_fast` aborts with an error in the same situation.
- Behavior change: when the active embedding version differs from the configured provider, the server logs an `embedding_version_mismatch` warning at startup and `memory.search` is refused until `memory.reembed` is run without `model_id`/`dim`. Re-embedding with an explicit model and `activate=true` switches the running process to that provider; update `EMBEDDING_MODEL`/`EMBEDDING_DIM` to keep it after a restart.
- Behavior change: existing `sentence-transformers` deployments get a new embedding version id because of the fingerprint fix and must run `memory.reembed` once before semantic search works again.
- Behavior change: `/admin/*` HTTP endpoints require `LLM_MEMORY_API_TOKEN` and return `403` when it is not configured. `/mcp`, `/sse` and `/messages/` require the bearer token when it is set. `/health` hides diagnostics from unauthenticated callers when a token is set.
- Behavior change: the HTTP server refuses to start on a non-loopback `MCP_MEMORY_HOST` without `LLM_MEMORY_API_TOKEN`.
- Docker Compose publishes the port on `127.0.0.1` only and requires `LLM_MEMORY_API_TOKEN`; the image runs as an unprivileged `app` user (uid 10001) and installs the `crypto` extra.
- Passphrase-derived encryption keys use a random per-installation salt stored next to the database (`<MEMORY_SQLITE_PATH>.salt`, mode 0600). Payloads encrypted with the previous static salt remain readable. Copy the salt file together with the database when moving encrypted data with a passphrase key.
- `cryptography` is now declared as the optional `crypto` extra (`pip install "llm-memory[crypto]"`).
- Tool `limit` arguments are capped at 100 (`search_inference_memory` at 50).
- The stderr activity log records a short SHA-256 prefix and the length of content, queries, reasons and summaries instead of a text preview.
- Tool descriptions, the `memory.about` payload, docstrings and messages are in English.
- Removed the unused `fastmcp` and `pyyaml` dependencies.
- All use of `mcp` transport internals is isolated in `src/mcp_server/mcp_transport.py`; the legacy SSE endpoint no longer relies on the private `request._send`.
- Replaced the Windows `.bat` launchers with the cross-platform `scripts/serve.py`.

### Docs

- README and quickstart synced with the active fast-memory distillation workflow
- Removed internal planning and prompt markdown files from the public docs surface
- Release checklist added in `docs/RELEASE_CHECKLIST.md`
- Coverage support and a documented local quality gate added in `docs/QUALITY_GATE.md`
