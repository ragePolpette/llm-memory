# llm-memory: Future Shared Project Catalog Risk

## Context

`llm-memory` now has an explicit project concept, including:

- `project_id`
- project discovery and creation
- scope hierarchy integration
- workspace/project memory isolation

This is correct for the current phase of the architecture.

## Current Risk

The same high-level domain concept, "project", now also exists in another repository:

- `llm-context`

But the two project models are currently defined independently.

Today this is acceptable because:

- each MCP still owns its own local responsibility
- there is no central control-plane enforcing a shared project identity model

## Why This Becomes a Problem

This duplication becomes a real architectural problem when a third actor appears, for example:

- dashboard / admin plane
- orchestration layer
- deployment/runtime controller
- project bootstrap workflow
- another MCP that needs project-aware behavior

At that point, there is a risk of having:

- two different sources of truth for project existence
- accidental project drift between context and memory systems
- multiple creation paths for what should be the same project
- naming mismatches and duplicate logical projects
- ambiguity about which metadata is canonical

## Architectural Distinction

There are two different layers that should not be confused.

### Shared platform concept

Project identity as a shared platform entity:

- `workspace_id`
- `project_id`
- `display_name`
- `description`
- `status`
- `created_at`
- `updated_at`
- optional shared metadata

### Local MCP-specific concept

Project-specific metadata owned by one MCP:

- in `llm-memory`: scope behavior, memory counts, memory-specific metadata
- in `llm-context`: root path, ingest/index status, retrieval profile

The future mistake to avoid is allowing both repositories to independently redefine the shared identity layer.

## Recommended Future Direction

Introduce a shared project catalog as the canonical identity layer.

That catalog should define:

- whether a project exists
- its canonical identifiers
- display metadata
- lifecycle state

Each MCP should then keep only its own extension metadata.

This keeps the model clean:

- shared project identity is centralized
- memory-specific and context-specific metadata remain local

## Minimal Future Shared Model

Recommended canonical project catalog fields:

- `workspace_id`
- `project_id`
- `display_name`
- `description`
- `status`
- `metadata`
- `created_at`
- `updated_at`

Everything else should remain service-specific unless a stronger platform use case emerges.

## Why Add This Note Now

This is not yet a blocker.

However, it will become important as soon as:

- project creation is used operationally by LLM/tooling
- dashboard wants a unified view of projects
- more project-aware services are added

The risk is not in having two local models today.

The risk is letting a third system create a third incompatible project definition later.

## Conclusion

For now:

- `llm-memory` can keep its local project registry
- `llm-context` can keep its local project registry

But the next cross-repo architectural step should be:

- a shared project catalog
- with local service metadata layered above it

That will avoid fragmentation of the project identity model as the MCP ecosystem grows.
