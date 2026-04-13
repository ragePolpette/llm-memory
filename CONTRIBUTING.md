# Contributing

## Scope

This project is built as a local-first MCP memory tool.
The target is a serious single-user or small-team codebase, not a public SaaS.

## Local Setup

```bash
pip install -e ".[dev]"
copy .env.example .env
pytest -q
```

Useful entrypoints:

- `python -m src.mcp_server.server`
- `python -m src.mcp_server.http_server`

## Development Workflow

1. create a branch: `feature/<short-name>`
2. keep the change scoped to one coherent slice
3. implement the change
4. add or update tests
5. run local verification
6. push branch
7. open PR
8. merge

## Quality Bar

Before opening a PR:

- run `ruff check src tests`
- run `pytest -q`
- run `pytest --cov=src --cov-report=term-missing --cov-report=xml -q`
- update docs when behavior changes
- keep changes scoped to one roadmap item or one coherent block
- avoid introducing legacy config or outdated docs

The documented local quality gate is:

- `ruff check src tests`
- `pytest -q`
- `pytest --cov=src --cov-report=term-missing --cov-report=xml -q`

Coverage is currently enforced at `82%` through project configuration in `pyproject.toml`.

## Golden Path

The repository includes a verified golden path test for:

- `memory.add`
- `memory.search`
- `memory.export`
- `memory.import`

Run it with:

```bash
pytest tests/test_golden_path.py -q
```

## Notes

- prefer local-first defaults
- avoid cloud assumptions in docs and examples
- keep the MCP tool surface aligned with the runtime actually shipped in this repo
- use `docs/RELEASE_CHECKLIST.md` when preparing a public-ready release
