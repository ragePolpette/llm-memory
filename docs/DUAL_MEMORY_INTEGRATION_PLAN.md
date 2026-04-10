# Dual Memory Integration Plan

## Goal

Extend `llm-memory` from a single durable-memory plane into a two-layer system:

- structured memory: dense, reusable, governed knowledge
- fast memory: noisy episodic material for later analysis and distillation

The intent is to preserve the current quality bar of the strong memory path while
adding a safe place for operational residue, attempts, fixes, and repeated issues.

## Current Fit

The repository already has three building blocks that make this extension natural:

- a deny-by-default persistence policy in `src/service/persistence_policy.py`
- an activity stream and audit trail in `src/service/memory_service.py`
- importance metadata and novelty scoring in `src/service/importance_scoring.py`

What is missing is not scoring infrastructure. What is missing is an explicit
storage and workflow boundary between:

- durable knowledge worth retrieval now
- episodic raw material worth distilling later

## Product Decision

Keep both entry points.

- direct structured write stays available for high-confidence reusable knowledge
- fast write becomes a separate path for low-confidence, messy, or transient material

Do not force everything through batch distillation.

That keeps the current `memory.add` path valuable and intentional, while giving the
agent a new "working residue" sink that does not pollute retrieval quality.

## Recommended Architecture

### 1. Structured memory stays as-is

The current `MemoryEntry` path remains the canonical strong-memory layer.

It should continue to store only:

- facts
- decisions
- assumptions worth keeping
- curated invalidations
- reusable conventions and project rules

This is the retrieval layer.

### 2. Fast memory becomes a separate layer

Do not overload `MemoryEntry` with an `episodic` flag.
Do not store raw notes in the same retrieval table.

Recommended approach:

- add a separate fast-memory model
- add a separate SQLite table
- add separate CRUD/query methods in the store
- keep it out of semantic retrieval by default

Fast memory should hold:

- "today X happened"
- fix attempts
- temporary debugging notes
- operational incidents
- repeated errors
- unresolved work fragments

This is an analysis layer, not a retrieval layer.

### 3. Distillation is a periodic pipeline

Add a batch process that reads raw memory and produces one of three outcomes:

- promote to structured memory
- compress into a summary or cluster record
- discard as noise

This can start as a local CLI or internal service method before becoming an MCP tool.

Initial implementation should be deterministic-first:

- group by recurrence and similarity
- detect repeated failures and repeated fixes
- compute frequency windows
- emit candidate summaries

LLM-based distillation can sit on top of that, not replace it.

### 4. Router integration comes after fast-memory exists

The router concept fits well, but should be added after the raw layer exists.

Realistic routing outputs for this repo:

- `structured`
- `raw`
- `drop`

A later extension may add:

- `weight_candidate`

But direct weight editing such as ROME-style mutation is not a first implementation
target for this codebase. It is too far from the current operational model and would
blur product scope too early.

### 5. Recurrence becomes a second-order importance signal

Your recurrence rule is good, but it belongs in the distillation pipeline and router,
not in the current direct-write path alone.

Recommended interpretation:

- static importance remains the first signal for `memory.add`
- recurrence becomes a promotion signal for fast memory
- low static importance plus repeated recurrence should raise review priority
- recurrence should be damped when repetition looks loop-like or low-diversity

That avoids promoting spammy loops while still rescuing low-salience patterns that
keep returning.

## Suggested Data Model

Introduce a new fast-memory record model, separate from `MemoryEntry`.

Suggested fields:

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
- `created_at`
- `updated_at`
- `resolved`
- `distilled_at`
- `distillation_status`
- `source`

Suggested distillation metadata:

- `recurrence_count`
- `first_seen_at`
- `last_seen_at`
- `cluster_id`
- `static_importance_score`
- `recurrence_boost`
- `distillation_score`

## Suggested API Surface

Phase 1:

- `memory.add` remains the strong-memory write path
- add `memory.log_fast`
- add admin read-only endpoints for fast-memory counts and recent events

Phase 2:

- add `memory.distill_fast` for local manual runs
- add CLI support for scheduled distillation
- add admin visibility for distillation candidates and outcomes

Phase 3:

- add a router mode that can suggest `structured`, `raw`, or `drop`
- optionally support auto-routing for trusted callers

## Suggested Rollout

### Phase A: Fast Memory Foundation

- define fast-memory model and store
- add fast-memory write API
- add tests and local admin visibility

### Phase B: Distillation Pipeline

- add recurrence counters and clustering
- add deterministic compression rules
- add manual distillation command

### Phase C: Structured Promotion

- emit structured candidates from fast-memory clusters
- preserve links from promoted structured records back to fast-memory evidence
- record audit events for every distillation outcome

### Phase D: Router

- centralize write decisioning across `structured`, `raw`, and `drop`
- keep direct strong-memory writes allowed
- introduce recurrence-aware promotion logic

## Non-Goals For First Iteration

- editing model weights directly
- replacing the current structured-memory path
- making fast memory searchable in the main semantic retrieval flow
- fully autonomous distillation without visibility or auditability

## Why This Fits The Repo

This direction strengthens the original product thesis:

- durable memory stays curated
- operational residue gets a first-class home
- repeated evidence becomes promotable instead of noisy
- the dashboard admin surface gains meaningful new observability

In short:

- structured memory remains clean
- fast memory absorbs the mess
- distillation turns repetition into knowledge
