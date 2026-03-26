# Quality Gate

This repository uses a local-first quality gate suitable for a serious portfolio project and small-team workflow.

## Required Setup

Install the development extras before running the gate:

```bash
pip install -e ".[dev]"
```

## Gate Commands

Run these commands before opening a PR or cutting a release:

```bash
ruff check src tests
pytest -q
pytest --cov=src --cov-report=term-missing --cov-report=xml -q
```

The lint scope intentionally targets the maintained runtime and test surface.
Legacy migration helpers under `scripts/legacy/` are kept outside this gate.

## Coverage Threshold

The project coverage threshold is currently `82%`.

The threshold is configured in `pyproject.toml` and is based on the current tested surface of the repository.
It should only be raised intentionally after coverage improves, not lowered casually to pass a change.

## Current Baseline

At the time this gate was introduced, the full suite measured `83%` total coverage on `src/`.

The threshold is set one point below that baseline to keep the gate stable while still protecting against regression.
