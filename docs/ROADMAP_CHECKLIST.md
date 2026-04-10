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

- [x] `P0` Fix Dockerfile build order and make image build consistent with the active runtime
- [x] `P0` Remove legacy LanceDB and stale env references from Docker/docs/quickstart
- [x] `P0` Fix `memory.reembed` so `model_id` and `dim` overrides are real, or remove the override contract
- [x] `P0` Replace raw SQLite file copy export with a consistent backup/export approach compatible with WAL
- [ ] `P0` Add GitHub Actions for tests and lint
- [x] `P0` Add a verified "golden path" quickstart: install, start, add, search, export, import
- [x] `P1` Add a project license before public GitHub publication
- [x] `P1` Add a concise contribution/development guide for local workflow
- [x] `P1` Add a release checklist for tagging a public-ready version

## Milestone 2: Team-Ready Maintainability

These items matter once the repo is already coherent and usable.

- [ ] `P1` Split `MemoryService` into smaller modules: write pipeline, retrieval, import/export, governance
- [ ] `P1` Introduce explicit DB schema versioning and migrations
- [x] `P1` Improve startup diagnostics and config validation
- [x] `P1` Reduce broad `except Exception` handling where real error types should be preserved
- [x] `P1` Add coverage support and a documented quality gate
- [x] `P1` Add a local admin surface for audit inspection, either MCP read-only tool or CLI
- [ ] `P2` Add benchmark scripts for search and reembed on realistic local datasets
- [ ] `P2` Add maintenance utilities for cleanup, compaction, and backup verification

## Milestone 2.5: Dual Memory And Distillation

These items extend the product from durable memory only to a deliberate two-layer
model: strong reusable knowledge plus fast episodic material for later compression.

- [ ] `P1` Add a separate fast-memory layer for episodic/log-like writes without polluting semantic retrieval
- [ ] `P1` Add a dedicated fast write API, keeping direct structured writes available
- [ ] `P1` Add read-only admin visibility for fast-memory counts, recent events, and distillation status
- [ ] `P2` Add a manual local distillation pipeline from fast memory to structured candidates
- [ ] `P2` Add recurrence-aware scoring for fast-memory promotion, with damping against noisy loops
- [ ] `P2` Link promoted structured memories back to fast-memory evidence for auditability
- [ ] `P3` Add a unified router that can choose `structured`, `raw`, or `drop`
- [ ] `P3` Keep "weights edit" or ROME-like memory mutation out of scope until the raw-memory workflow proves useful

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
- [x] `2026-03-26` Align Docker setup and runtime docs - branch: `feature/docker-docs-runtime-alignment` - PR: `#10` - notes: Docker now starts the HTTP MCP runtime and the Docker/quickstart docs no longer reference LanceDB or obsolete tool names.
- [x] `2026-03-26` Replace raw SQLite export with backup API - branch: `feature/sqlite-export-backup` - PR: `#11` - notes: SQLite export now uses a consistent database backup instead of copying the raw database file while WAL is active.
- [x] `2026-03-26` Verify and document the golden path, add license, and add contribution guide - branch: `feature/golden-path-license-dev-guide` - PR: `#12` - notes: Added a golden-path MCP tool test, refined quickstart examples, added MIT license, and documented the development workflow.
- [x] `2026-03-26` Add release checklist for public-ready tagging - branch: `feature/release-checklist` - PR: `#13` - notes: Added a concrete release checklist, linked it from the docs surface, and introduced a minimal changelog file.
- [x] `2026-03-26` Improve startup diagnostics and config validation - branch: `feature/startup-diagnostics-config-validation` - PR: `#14` - notes: Added stronger config validation, ensured runtime directories exist at startup, and exposed a reusable diagnostics summary in startup logs and HTTP health.
- [x] `2026-03-26` Narrow broad exception handling around decrypt paths - branch: `feature/narrow-exception-handling` - PR: `#15` - notes: Introduced a typed decrypt error and replaced generic catches in the encrypted payload paths with explicit handling.
- [x] `2026-03-26` Add coverage support and a documented quality gate - branch: `feature/coverage-quality-gate` - PR: `#16` - notes: Added coverage configuration, documented the local quality gate, and aligned release/dev docs with an explicit coverage command and threshold.
- [x] `2026-03-26` Add local admin HTTP surface for audit inspection - branch: `feature/memory-admin-http-surface` - PR: `#17` - notes: Added read-only HTTP admin endpoints for summary, audit, and projects, with filtered audit queries and route-level test coverage.
- [x] `2026-04-10` Integrate dual-memory roadmap extension - branch: `docs/dual-memory-roadmap` - PR: `pending` - notes: Added a concrete plan for raw episodic memory, batch distillation, recurrence-aware promotion, and future router integration without polluting the strong-memory path.
- [x] `2026-04-10` Add fast-memory foundation operational plan - branch: `docs/fast-memory-foundation-plan` - PR: `pending` - notes: Formalized the first implementation slice, renamed the episodic layer to fast memory, and fixed the v1 selection scoring approach around static score, recurrence, and noise penalty.

## Update Rule

When an item is completed, append a short note in this format:

- `[x] YYYY-MM-DD - item name - branch: feature/... - PR: ... - notes: ...`
