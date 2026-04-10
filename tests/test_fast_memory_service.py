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
    assert entry.selection_score is not None
    assert entry.metadata["fast_memory_scoring"]["formula_version"] == "fast-memory-v1"
    assert entry.metadata["fast_memory_scoring"]["selection_score"] == entry.selection_score

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


def test_log_fast_rejects_invalid_recurrence_count(service):
    actor = ActorContext(agent_id="agent-fast", user_id="user-fast", workspace_id="ws-test", project_id="prj-test")

    with pytest.raises(MemoryInputError, match="INVALID_RECURRENCE_COUNT"):
        service.log_fast(
            {
                "content": "Burst di retry non normalizzato.",
                "agent_id": actor.agent_id,
                "recurrence_count": 0,
            },
            actor,
        )


def test_log_fast_computes_higher_score_for_recurrent_cross_session_pattern(service):
    actor = ActorContext(agent_id="agent-fast", user_id="user-fast", workspace_id="ws-test", project_id="prj-test")

    single = service.log_fast(
        {
            "content": "Nota singola su un errore raro.",
            "agent_id": actor.agent_id,
            "event_type": "note",
            "metadata": {"importance_score": 20},
        },
        actor,
    )
    recurrent = service.log_fast(
        {
            "content": "Errore ricorrente riapparso su sessioni diverse.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "recurrence_count": 5,
            "metadata": {
                "importance_score": 20,
                "distinct_session_count": 3,
                "distinct_day_count": 2,
            },
        },
        actor,
    )

    single_entry = service.get_fast(single["entry_id"], actor)
    recurrent_entry = service.get_fast(recurrent["entry_id"], actor)

    assert single_entry is not None
    assert recurrent_entry is not None
    assert recurrent_entry.selection_score > single_entry.selection_score
    assert recurrent_entry.metadata["fast_memory_scoring"]["recurrence_boost"] > 0.0


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
