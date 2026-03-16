from __future__ import annotations

import pytest

from src.config import MemoryScope, ScopeLevel, Tier
from src.models import EntryType, MemoryEntry, ScopeRef
from src.service.memory_service import ActorContext


@pytest.mark.asyncio
async def test_workspace_scope_is_visible_across_projects_same_workspace(service):
    actor_a = ActorContext(agent_id="agent-a", user_id="user-a", workspace_id="ws-test", project_id="project-a")
    actor_b = ActorContext(agent_id="agent-b", user_id="user-b", workspace_id="ws-test", project_id="project-b")
    service.create_project(actor=actor_a, project_id="project-a")
    service.create_project(actor=actor_b, project_id="project-b")

    await service.add(
        {
            "content": "La convenzione workspace usa commit squash prima del merge.",
            "agent_id": actor_a.agent_id,
            "visibility": "shared",
            "tier": "tier-2",
            "type": "fact",
            "scope": {
                "workspace_id": "ws-test",
                "scope_level": "workspace",
            },
        },
        actor_a,
    )

    results = await service.search("commit squash merge", actor_b, limit=5)

    assert any(bundle.scope.scope_level.value == "workspace" for bundle in results)


@pytest.mark.asyncio
async def test_global_scope_is_visible_across_workspaces(service):
    actor_a = ActorContext(agent_id="agent-a", user_id="user-a", workspace_id="ws-test", project_id="project-a")
    actor_b = ActorContext(agent_id="agent-b", user_id="user-b", workspace_id="ws-other", project_id="project-b")
    service.create_project(actor=actor_a, project_id="project-a")
    service.create_project(actor=actor_b, project_id="project-b")

    entry = MemoryEntry(
        tier=Tier.TIER_2,
        scope=ScopeRef(
            workspace_id=service._GLOBAL_BUCKET_WORKSPACE_ID,
            project_id=service._GLOBAL_BUCKET_PROJECT_ID,
            scope_level=ScopeLevel.GLOBAL,
        ),
        visibility=MemoryScope.GLOBAL,
        type=EntryType.FACT,
        content="Preferenza globale: risposte concise e orientate al task.",
        context="global-style",
    )
    service.store.add_entry(entry)
    version = service._current_embedding_version()
    vector = (await service.embedding_provider.embed([entry.content]))[0]
    service.vector_store.upsert(entry.id, version.version_id, vector, entry.updated_at.isoformat())

    results = await service.search("risposte concise", actor_b, limit=5)

    assert any(bundle.scope.scope_level.value == "global" for bundle in results)


@pytest.mark.asyncio
async def test_search_can_exclude_workspace_and_global_scopes(service):
    actor_a = ActorContext(agent_id="agent-a", user_id="user-a", workspace_id="ws-test", project_id="project-a")
    actor_b = ActorContext(agent_id="agent-b", user_id="user-b", workspace_id="ws-test", project_id="project-b")
    service.create_project(actor=actor_a, project_id="project-a")
    service.create_project(actor=actor_b, project_id="project-b")

    await service.add(
        {
            "content": "Convenzione workspace: usare changelog sintetico.",
            "agent_id": actor_a.agent_id,
            "visibility": "shared",
            "tier": "tier-2",
            "type": "fact",
            "scope": {
                "workspace_id": "ws-test",
                "scope_level": "workspace",
            },
        },
        actor_a,
    )

    results = await service.search(
        "changelog sintetico",
        actor_b,
        limit=5,
        include_project=True,
        include_workspace=False,
        include_global=False,
    )

    assert results == []


def test_scope_overview_returns_counts_by_bucket(service):
    actor = ActorContext(agent_id="agent-a", user_id="user-a", workspace_id="ws-test", project_id="prj-test")

    overview = service.scope_overview(actor)

    assert set(overview.keys()) == {"project", "workspace", "global"}
    assert overview["project"]["project_id"] == "prj-test"
