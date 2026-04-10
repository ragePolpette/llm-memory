# Fast Memory Foundation Plan

## Scope

This document defines the first implementation slice for the new two-layer model:

- strong memory: curated reusable knowledge
- fast memory: episodic operational residue

This phase does not implement the full router.
This phase does not change the semantics of `memory.add`.
This phase creates the minimum viable base for fast memory.

## Product Rules

### Strong memory

Strong memory remains the destination layer.

It keeps:

- facts
- decisions
- reusable assumptions
- invalidations
- stable project conventions

It is the default retrieval layer.

### Fast memory

Fast memory is not knowledge by default.

It holds:

- attempts
- fixes
- incidents
- temporary notes
- repeated issues
- raw operational observations

It is an analysis layer.
It must not pollute normal semantic retrieval.

## Foundation Goal

At the end of this phase, the repo must support:

1. writing fast-memory records explicitly
2. storing them separately from strong memory
3. inspecting them through admin surfaces
4. computing a first recurrence-aware selection score
5. preparing later distillation without changing the strong-memory path

## Phase 1 Deliverables

### D1. New data model

Add a separate model, for example `FastMemoryEntry`.

Required fields:

- `id`
- `workspace_id`
- `project_id`
- `agent_id`
- `user_id`
- `session_id`
- `event_type`
- `content`
- `context`
- `tags`
- `metadata`
- `source`
- `created_at`
- `updated_at`
- `resolved`
- `distillation_status`
- `distilled_at`

Optional fields for recurrence and later clustering:

- `cluster_id`
- `recurrence_count`
- `first_seen_at`
- `last_seen_at`
- `selection_score`

### D2. Separate persistence

Add separate SQLite storage.

Recommended tables:

- `fast_memory_entries`
- optional later: `fast_memory_clusters`

Do not reuse the main `entries` table.
Do not attach embeddings to fast memory in phase 1.

### D3. Service surface

Add dedicated service methods:

- `log_fast(...)`
- `list_fast(...)`
- `get_fast(...)`
- `summarize_fast_admin(...)`

Phase 1 does not require semantic search over fast memory.

### D4. MCP surface

Add one new explicit write tool:

- `memory.log_fast`

This tool is for low-confidence or episodic notes.

Do not weaken `memory.add`.
Do not auto-route `memory.add` into fast memory in phase 1.

### D5. Admin visibility

Extend admin read-only visibility with:

- fast-memory counts
- recent fast-memory events
- event-type distribution
- resolved vs unresolved counts
- distillation status counts

This should be visible both in HTTP admin and later in the dashboard.

## Selection Score V1

Phase 1 should formalize the scoring foundation for later distillation.

Use the score only for fast memory.
Do not apply it to strong memory entries already accepted.

### Formula

```text
S_select(m) = S_meta(m) + alpha * R(m) * (1 - S_meta(m)) * (1 - N(m))
```

Where:

- `S_meta(m)` = normalized static score in `[0, 1]`
- `R(m)` = normalized recurrence score in `[0, 1]`
- `N(m)` = normalized noise penalty in `[0, 1]`
- `alpha` = trust weight assigned to recurrence

### Interpretation

- if a memory is already strong, recurrence adds little
- if a memory is weak but keeps coming back, recurrence can raise it
- if repetition looks noisy, the penalty suppresses the boost

This better matches the product goal than a damping term based purely on the
absolute gap between static score and recurrence.

### Recurrence decomposition

Use:

```text
R(m) = F(m) * Q(m)
```

Where:

- `F(m)` = pure frequency, preferably compressed with `log1p`
- `Q(m)` = recurrence quality

Phase 1 quality signals:

- repeated across different sessions
- repeated across different tasks
- repeated across different days

Phase 1 negative signals:

- burst retries in the same session
- exact-duplicate spam
- very short intervals from the same actor

### Noise penalty

`N(m)` should start simple.

Phase 1 suggested contributors:

- duplicate burst ratio
- same-session concentration
- same-actor concentration
- excessive short-window repetition

## Operational Sequence

Implement in this order.

### Step 1. Model and store

- add `FastMemoryEntry` to `src/models.py`
- add store interface methods in `src/storage/base.py`
- add SQLite schema in `src/storage/sqlite_store.py`
- add store tests

Definition of done:

- fast-memory records can be inserted, listed, and fetched locally
- schema is created automatically on startup

### Step 2. Service methods

- add `log_fast` path in `src/service/memory_service.py`
- add validation rules for fast-memory payloads
- add audit events for fast-memory writes
- keep privacy and encryption compatibility where relevant

Definition of done:

- service can write and read fast-memory entries
- writes are auditable

### Step 3. MCP tool

- add `memory.log_fast` in `src/mcp_server/tools.py`
- document it in `README.md` and `QUICKSTART.md`
- add end-to-end tests

Definition of done:

- tool is visible and usable from MCP
- it does not affect standard `memory.search`

### Step 4. Admin read-only surface

- extend admin summary with fast-memory counts
- add `GET /admin/fast-memory`
- support simple filters:
  - `limit`
  - `event_type`
  - `agent_id`
  - `resolved`
  - `distillation_status`

Definition of done:

- local operators can inspect fast memory without opening SQLite directly

### Step 5. Selection score foundation

- implement `S_select` utilities in a dedicated module
- compute score metadata on fast-memory writes or admin aggregation
- do not auto-promote yet

Definition of done:

- each fast-memory entry can expose:
  - `static_score`
  - `recurrence_score`
  - `noise_penalty`
  - `selection_score`

## Suggested File Layout

Recommended additions:

- `src/service/fast_memory_scoring.py`
- `tests/test_fast_memory_scoring.py`
- `tests/test_fast_memory_store.py`
- `tests/test_fast_memory_api.py`

Possible later refactor target:

- `src/service/fast_memory_service.py`

## Explicit Non-Goals

Not in this phase:

- full batch distillation
- automatic promotion to strong memory
- unified router
- dashboard implementation
- vector search over fast memory
- weight editing or model-memory mutation

## First Branches

Recommended execution order:

1. `feature/fast-memory-store`
2. `feature/fast-memory-service`
3. `feature/fast-memory-mcp-tool`
4. `feature/fast-memory-admin-surface`
5. `feature/fast-memory-selection-score`

## Success Criteria

This foundation is successful if:

- strong memory remains clean
- fast memory becomes writable and inspectable
- the system can measure recurrence without promoting noise
- the repo is ready for a later distillation pass without redesigning the model
