from __future__ import annotations

import pytest

from src.service.memory_service import ActorContext, MemoryInputError


def test_log_fast_persists_entry_and_audits(service):
    actor = ActorContext(agent_id="agent-fast", user_id="user-fast", workspace_id="ws-test", project_id="prj-test")

    result = service.log_fast(
        {
            "content": "Fix temporaneo sul parser YAML ancora da consolidare.",
            "context": "debugging parser",
            "agent_id": actor.agent_id,
            "event_type": "fix_attempt",
            "session_id": "session-1",
            "tags": ["debug", "yaml"],
            "metadata": {"ticket": "FAST-1"},
        },
        actor,
    )

    assert result["success"] is True
    entry = service.get_fast(result["entry_id"], actor)
    assert entry is not None
    assert entry.event_type == "fix_attempt"
    assert entry.session_id == "session-1"
    assert entry.metadata["ticket"] == "FAST-1"

    audits = service.store.list_audit(entry_id=result["entry_id"], limit=10)
    assert any(audit.action == "fast_write" and audit.reason == "log_fast" for audit in audits)


def test_log_fast_rejects_invalid_payload(service):
    actor = ActorContext(agent_id="agent-fast", user_id="user-fast", workspace_id="ws-test", project_id="prj-test")

    with pytest.raises(MemoryInputError, match="INVALID_CONTENT"):
        service.log_fast(
            {
                "content": "   ",
                "agent_id": actor.agent_id,
            },
            actor,
        )


@pytest.mark.asyncio
async def test_list_fast_is_project_isolated(service):
    actor_a = ActorContext(agent_id="agent-a", user_id="user-a", workspace_id="ws-test", project_id="project-a")
    actor_b = ActorContext(agent_id="agent-b", user_id="user-b", workspace_id="ws-test", project_id="project-b")
    service.create_project(actor=actor_a, project_id="project-a")
    service.create_project(actor=actor_b, project_id="project-b")

    service.log_fast(
        {
            "content": "Errore ricorrente sull'import batch.",
            "agent_id": actor_a.agent_id,
            "event_type": "incident",
            "scope": {
                "workspace_id": "ws-test",
                "project_id": "project-a",
            },
        },
        actor_a,
    )

    rows_a = service.list_fast(actor_a, limit=20)
    rows_b = service.list_fast(actor_b, limit=20)

    assert len(rows_a) == 1
    assert rows_a[0].project_id == "project-a"
    assert rows_b == []


def test_get_fast_denies_cross_project_read(service):
    actor_a = ActorContext(agent_id="agent-a", user_id="user-a", workspace_id="ws-test", project_id="project-a")
    actor_b = ActorContext(agent_id="agent-b", user_id="user-b", workspace_id="ws-test", project_id="project-b")
    service.create_project(actor=actor_a, project_id="project-a")
    service.create_project(actor=actor_b, project_id="project-b")

    result = service.log_fast(
        {
            "content": "Nota veloce su retry rumorosi.",
            "agent_id": actor_a.agent_id,
            "event_type": "note",
            "scope": {
                "workspace_id": "ws-test",
                "project_id": "project-a",
            },
        },
        actor_a,
    )

    with pytest.raises(PermissionError, match="Read denied by scope policy"):
        service.get_fast(result["entry_id"], actor_b)
