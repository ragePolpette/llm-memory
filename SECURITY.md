# Security Policy

## Scope

`llm-memory` is a local-first MCP memory service intended for a single workstation or a small trusted team. This policy covers the code in this repository: the MCP server (stdio and HTTP transports), the service layer, the SQLite storage and vector stores, the import/export paths, and the optional privacy and encryption helpers.

## Threat Model

The design assumes:

- the host running the service is trusted
- clients that can reach the service are trusted
- the service is not exposed to the public internet

What the code does to reduce risk:

- Outbound network calls are blocked by default (`MEMORY_ALLOW_OUTBOUND_NETWORK=false`); loopback connections remain allowed.
- The HTTP transport binds to `127.0.0.1` by default and restricts accepted `Host` and `Origin` headers through `MCP_MEMORY_ALLOWED_HOSTS` and `MCP_MEMORY_ALLOWED_ORIGINS`.
- Optional bearer token authentication (`LLM_MEMORY_API_TOKEN`) protects `/mcp`, `/sse` and `/messages/`; the `/admin/*` endpoints always require it. The server refuses to bind to a non-loopback host without a token.
- Exports contain only entries the caller is allowed to read.
- The stderr activity log records hashes and lengths, not content.
- `memory.import` and `memory.export` paths must resolve under `MEMORY_IMPORT_EXPORT_BASE_DIR`.
- A privacy policy can drop configured metadata keys and flag sensitive tags (`MEMORY_PRIVACY_*`).
- Sensitive payloads can optionally be encrypted at rest with a key read from an environment variable (`MEMORY_ENCRYPTION_ENABLED`, `MEMORY_ENCRYPTION_KEY_ENV`).
- Writes are recorded in an audit trail.

## Not Covered

- Fine-grained authorization: the bearer token is a single shared secret. Anyone holding it can act as any `agent_id`, because agent identity is self-declared in tool arguments. TLS is not provided; put a TLS-terminating reverse proxy in front of the server if traffic leaves the host. The provided Docker Compose file publishes port 8767 on `127.0.0.1` only and requires a token.
- Encryption is off by default, requires the optional `crypto` extra, and applies only to payloads selected by the privacy policy. Metadata and the SQLite file itself are not encrypted as a whole. Passphrase-derived keys use a random salt stored in `<MEMORY_SQLITE_PATH>.salt`.
- Protection against a compromised host, malicious local users, or a malicious MCP client.
- Content quality: the service does not verify that stored memories are accurate or free of prompt injection.
- Third-party dependencies and any external embedding provider you configure.
- Legacy scripts under `scripts/legacy/`.

## Reporting a Vulnerability

Please do not open a public issue for security problems.

Report privately through GitHub Security Advisories: use the "Report a vulnerability" button on the Security tab of https://github.com/ragePolpette/llm-memory. If that is unavailable, open a minimal issue that asks for a private contact channel and does not include details.

Include the affected version or commit, a description of the issue, and steps to reproduce. Do not include real secrets or private memory content.

This is a small project maintained on a best-effort basis; no response time is guaranteed.

## Supported Versions

Only the latest commit on `main` is supported.
