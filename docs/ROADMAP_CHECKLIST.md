# llm-memory Roadmap Checklist

## Product Target

This roadmap is intentionally scoped for a:

- local-first tool
- single-user or small team workflow
- public GitHub portfolio project
- enterprise-lean codebase, not public internet SaaS

The goal is not "big enterprise complexity".
The goal is "serious, reliable, well-structured, safe, and credible".

## Working Workflow

For every roadmap item:

1. create a branch: `feature/<short-name>`
2. implement the change
3. add or update tests
4. run verification locally
5. push branch
6. open PR
7. merge
8. update this file with status, date, and PR/reference

## Definition Of Done

An item is done only when:

- code is implemented
- tests are added or updated when relevant
- docs are aligned
- manual verification is noted when relevant
- this checklist is updated

## Status Legend

- `[ ]` not started
- `[~]` in progress
- `[x]` done

## Baseline

- [x] Initial audit completed on `2026-03-25`
- [x] Select first `P0` item and start implementation workflow

## Milestone 1: Portfolio-Ready Core

These items have the highest priority because they improve real quality and also how the repo looks to an external reviewer.

- [ ] `P0` Fix Dockerfile build order and make image build consistent with the active runtime
- [ ] `P0` Remove legacy LanceDB and stale env references from Docker/docs/quickstart
- [x] `P0` Fix `memory.reembed` so `model_id` and `dim` overrides are real, or remove the override contract
- [ ] `P0` Replace raw SQLite file copy export with a consistent backup/export approach compatible with WAL
- [ ] `P0` Add GitHub Actions for tests and lint
- [ ] `P0` Add a verified "golden path" quickstart: install, start, add, search, export, import
- [ ] `P1` Add a project license before public GitHub publication
- [ ] `P1` Add a concise contribution/development guide for local workflow
- [ ] `P1` Add a release checklist for tagging a public-ready version

## Milestone 2: Team-Ready Maintainability

These items matter once the repo is already coherent and usable.

- [ ] `P1` Split `MemoryService` into smaller modules: write pipeline, retrieval, import/export, governance
- [ ] `P1` Introduce explicit DB schema versioning and migrations
- [ ] `P1` Improve startup diagnostics and config validation
- [ ] `P1` Reduce broad `except Exception` handling where real error types should be preserved
- [ ] `P1` Add coverage support and a documented quality gate
- [ ] `P1` Add a local admin surface for audit inspection, either MCP read-only tool or CLI
- [ ] `P2` Add benchmark scripts for search and reembed on realistic local datasets
- [ ] `P2` Add maintenance utilities for cleanup, compaction, and backup verification

## Milestone 3: Enterprise-Lean Direction

These items are intentionally lower priority. They matter if the tool starts being used by more colleagues or becomes a shared internal service.

- [ ] `P2` Harden caller identity boundaries for shared non-public use
- [ ] `P2` Improve privacy handling beyond tag-based redaction only
- [ ] `P2` Add backup and restore smoke tests
- [ ] `P2` Add packaging/release automation for reproducible local distribution
- [ ] `P3` Re-evaluate the vector backend only if dataset size or team usage justifies it

## Progress Log

- [x] `2026-03-25` Created roadmap checklist from repo audit.
- [x] `2026-03-25` Fix `memory.reembed` override contract - branch: `feature/fix-reembed-contract` - PR: `#9` - notes: `reembed` now resolves a provider coherent with requested overrides and regression coverage verifies generated vector dimension.

## Update Rule

When an item is completed, append a short note in this format:

- `[x] YYYY-MM-DD - item name - branch: feature/... - PR: ... - notes: ...`
