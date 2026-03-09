"""Unit test: storage + dedup + scope isolation."""

from __future__ import annotations

import pytest

from src.service.memory_service import ActorContext


@pytest.mark.asyncio
async def test_dedup_hash(service):
    actor = ActorContext(agent_id="agent-a", user_id="user-a", workspace_id="ws-test", project_id="prj-test")

    first = await service.add(
        {
            "content": "Il progetto usa SQLite locale per metadata.",
            "context": "arch",
            "agent_id": actor.agent_id,
            "visibility": "shared",
            "tier": "tier-2",
            "type": "fact",
        },
        actor,
    )

    second = await service.add(
        {
            "content": "Il progetto usa SQLite locale per metadata.",
            "context": "arch",
            "agent_id": actor.agent_id,
            "visibility": "shared",
            "tier": "tier-2",
            "type": "fact",
        },
        actor,
    )

    assert first["success"] is True
    assert second["duplicate_of"] == first["entry_id"]


@pytest.mark.asyncio
async def test_scope_isolation(service):
    actor_a = ActorContext(agent_id="agent-a", user_id="user-a", workspace_id="ws-test", project_id="project-a")
    actor_b = ActorContext(agent_id="agent-b", user_id="user-b", workspace_id="ws-test", project_id="project-b")

    await service.add(
        {
            "content": "Dato privato progetto A",
            "agent_id": actor_a.agent_id,
            "scope": {
                "workspace_id": "ws-test",
                "project_id": "project-a",
                "agent_id": "agent-a",
            },
            "visibility": "private",
            "tier": "tier-2",
            "type": "fact",
        },
        actor_a,
    )

    results_a = await service.search("Dato privato", actor_a, limit=5)
    results_b = await service.search("Dato privato", actor_b, limit=5)

    assert len(results_a) >= 1
    assert len(results_b) == 0
