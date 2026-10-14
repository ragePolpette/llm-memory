# Contributing

## Scope

This project is a local-first MCP memory tool. The target is a single-user or small-team codebase, not a public SaaS.

## Local Setup

```bash
pip install -e ".[dev]"
cp .env.example .env
pytest -q
```

On Windows `cmd`, use `copy .env.example .env`.

Useful entrypoints:

- `python -m src.mcp_server.server`
- `python -m src.mcp_server.http_server`

## Workflow

1. Create a branch from `main`: `feature/<short-name>` or `fix/<short-name>`.
2. Keep the change scoped to one coherent slice.
3. Add or update tests.
4. Run the quality gate below.
5. Push the branch and open a pull request using the template.
6. Merge after review.

## Quality Gate

Before opening a PR:

```bash
ruff check src tests
pytest -q
pytest --cov=src --cov-report=term-missing --cov-report=xml -q
```

Coverage is enforced at 82% through `pyproject.toml`. See [docs/QUALITY_GATE.md](docs/QUALITY_GATE.md).

The golden path test covers `memory.add`, `memory.search`, `memory.export`, and `memory.import`:

```bash
pytest tests/test_golden_path.py -q
```

Also:

- update docs when behavior changes
- keep the documented MCP tool surface aligned with the runtime shipped in this repo
- avoid introducing legacy config or outdated docs

## Security Rules

- Never commit secrets, real `.env` files, encryption keys, or local databases. `.gitignore` excludes them; do not weaken it.
- Use `.env.example` for new configuration keys, with safe defaults.
- Prefer local-first defaults and do not add cloud assumptions to code, docs, or examples.
- Keep outbound network blocking and import/export path confinement intact.
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Releases

Use [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) when preparing a release.
