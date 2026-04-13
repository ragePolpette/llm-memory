from __future__ import annotations

import pytest

from src.config import MemoryScope, Tier
from src.models import EntryType, FastMemoryDistillationStatus
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


def test_log_fast_normalizes_structured_metadata(service):
    actor = ActorContext(agent_id="agent-fast", user_id="user-fast", workspace_id="ws-test", project_id="prj-test")

    result = service.log_fast(
        {
            "content": "L'utente y vedeva solo il menu x.",
            "context": "menu troubleshooting",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "kind": "bug",
            "product_area": "authorization",
            "component": "menu-engine",
            "feature": "dynamic-menu",
            "entity_refs": ["user:y", "menu:x", "user:y"],
            "symptoms": ["utente vede solo menu x", "permessi incompleti"],
            "action_taken": "Update sulla tabella dei permessi utente.",
            "outcome": "Menu corretti dopo refresh autorizzazioni.",
            "root_cause_hypothesis": "Join incompleto tra profilo utente e permessi menu.",
            "resolution_confidence": 0.92,
            "generalizable": "yes",
            "evidence_refs": ["ticket:AUTH-17", "sql:perm-fix-1"],
            "sql_patch": "update table t set ... where utente = 'y'",
            "commands": ["sqlcmd -i perm_fix.sql", "refresh-menu-cache"],
            "observed_by": "support",
            "affected_user_scope": "single-user",
        },
        actor,
    )

    entry = service.get_fast(result["entry_id"], actor)
    assert entry is not None
    structured = entry.metadata["structured_context"]
    assert structured["kind"] == "bug"
    assert structured["product_area"] == "authorization"
    assert structured["component"] == "menu-engine"
    assert structured["entity_refs"] == ["user:y", "menu:x"]
    assert structured["generalizable"] == "yes"
    assert structured["resolution_confidence"] == 0.92
    assert structured["commands"] == ["sqlcmd -i perm_fix.sql", "refresh-menu-cache"]


def test_log_fast_rejects_invalid_structured_metadata_type(service):
    actor = ActorContext(agent_id="agent-fast", user_id="user-fast", workspace_id="ws-test", project_id="prj-test")

    with pytest.raises(MemoryInputError, match="INVALID_FIELD_TYPE"):
        service.log_fast(
            {
                "content": "Payload strutturato non valido.",
                "agent_id": actor.agent_id,
                "entity_refs": "user:y",
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


def test_summarize_fast_updates_status_and_audits(service):
    actor = ActorContext(agent_id="agent-fast", user_id="user-fast", workspace_id="ws-test", project_id="prj-test")
    created = service.log_fast(
        {
            "content": "Sequenza sporca di retry sul parser CSV.",
            "agent_id": actor.agent_id,
            "event_type": "retry",
        },
        actor,
    )

    result = service.summarize_fast(
        entry_id=created["entry_id"],
        actor=actor,
        summary="Retry rumorosi sul parser CSV senza nuovi segnali utili.",
        reason="manual triage",
        cluster_id="cluster-csv",
        resolved=True,
    )

    entry = service.get_fast(created["entry_id"], actor)
    assert entry is not None
    assert result["distillation_status"] == FastMemoryDistillationStatus.SUMMARIZED.value
    assert entry.distillation_status == FastMemoryDistillationStatus.SUMMARIZED
    assert entry.cluster_id == "cluster-csv"
    assert entry.resolved is True
    assert entry.metadata["distillation_summary"].startswith("Retry rumorosi")
    assert entry.metadata["last_fast_memory_distillation"]["action"] == "summarize"

    audits = service.store.list_audit(entry_id=created["entry_id"], limit=10)
    assert any(audit.action == "fast_summarize" and audit.reason == "manual triage" for audit in audits)


def test_discard_fast_marks_entry_discarded(service):
    actor = ActorContext(agent_id="agent-fast", user_id="user-fast", workspace_id="ws-test", project_id="prj-test")
    created = service.log_fast(
        {
            "content": "Burst ripetuto dello stesso fallback locale.",
            "agent_id": actor.agent_id,
            "event_type": "retry",
        },
        actor,
    )

    result = service.discard_fast(
        entry_id=created["entry_id"],
        actor=actor,
        reason="noise-only pattern",
        resolved=False,
    )

    entry = service.get_fast(created["entry_id"], actor)
    assert entry is not None
    assert result["distillation_status"] == FastMemoryDistillationStatus.DISCARDED.value
    assert entry.distillation_status == FastMemoryDistillationStatus.DISCARDED
    assert entry.metadata["last_fast_memory_distillation"]["action"] == "discard"

    audits = service.store.list_audit(entry_id=created["entry_id"], limit=10)
    assert any(audit.action == "fast_discard" and audit.reason == "noise-only pattern" for audit in audits)


@pytest.mark.asyncio
async def test_promote_fast_creates_strong_memory_and_marks_source(service):
    actor = ActorContext(agent_id="agent-fast", user_id="user-fast", workspace_id="ws-test", project_id="prj-test")
    created = service.log_fast(
        {
            "content": "Errore ricorrente sull'import incrementale dopo resume.",
            "context": "incident review",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "recurrence_count": 4,
            "component": "import-engine",
            "kind": "bug",
            "metadata": {"importance_score": 35, "distinct_session_count": 3},
        },
        actor,
    )

    result = await service.promote_fast(
        entry_id=created["entry_id"],
        actor=actor,
        reason="recurring import issue became reusable knowledge",
        target_tier=Tier.TIER_2,
        memory_type=EntryType.FACT,
        visibility=MemoryScope.SHARED,
        summary="L'import incrementale fallisce dopo resume se il checkpoint locale non viene riallineato.",
        confidence=0.85,
    )

    fast_entry = service.get_fast(created["entry_id"], actor)
    strong_entry = service.get(result["promoted_entry_id"], actor)

    assert fast_entry is not None
    assert strong_entry is not None
    assert result["distillation_status"] == FastMemoryDistillationStatus.PROMOTED.value
    assert fast_entry.distillation_status == FastMemoryDistillationStatus.PROMOTED
    assert fast_entry.metadata["promoted_entry_id"] == strong_entry.id
    assert strong_entry.tier == Tier.TIER_2
    assert strong_entry.type == EntryType.FACT
    assert strong_entry.source == "internal_governance"
    assert strong_entry.metadata["fast_memory_origin"]["entry_id"] == fast_entry.id
    assert strong_entry.metadata["structured_context"]["component"] == "import-engine"
    assert strong_entry.metadata["persistence_decision"]["write_path"] == "promote_fast"
    assert strong_entry.content.startswith("L'import incrementale fallisce")

    audits = service.store.list_audit(entry_id=created["entry_id"], limit=20)
    assert any(audit.action == "fast_promote" for audit in audits)


@pytest.mark.asyncio
async def test_promote_fast_denies_cross_project_promotion(service):
    actor_a = ActorContext(agent_id="agent-a", user_id="user-a", workspace_id="ws-test", project_id="project-a")
    actor_b = ActorContext(agent_id="agent-b", user_id="user-b", workspace_id="ws-test", project_id="project-b")
    service.create_project(actor=actor_a, project_id="project-a")
    service.create_project(actor=actor_b, project_id="project-b")

    created = service.log_fast(
        {
            "content": "Nota veloce riservata al progetto A.",
            "agent_id": actor_a.agent_id,
            "scope": {
                "workspace_id": "ws-test",
                "project_id": "project-a",
            },
        },
        actor_a,
    )

    with pytest.raises(PermissionError, match="Write denied by scope policy"):
        await service.promote_fast(
            entry_id=created["entry_id"],
            actor=actor_b,
            reason="should not cross projects",
        )


@pytest.mark.asyncio
async def test_apply_fast_distillation_dry_run_does_not_mutate_entries(service):
    actor = ActorContext(agent_id="agent-fast", user_id="user-fast", workspace_id="ws-test", project_id="prj-test")
    first = service.log_fast(
        {
            "content": "L'utente y vedeva solo il menu x.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "component": "menu-engine",
        },
        actor,
    )
    second = service.log_fast(
        {
            "content": "L'utente z vedeva solo il menu x.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "component": "menu-engine",
        },
        actor,
    )

    result = await service.apply_fast_distillation(
        actor=actor,
        reason="preview agentic output",
        dry_run=True,
        payload={
            "decisions": [
                {
                    "cluster_id": "cluster-menu",
                    "action": "promote",
                    "summary": "La gestione menu dipende dai permessi utente consolidati.",
                    "confidence": 0.88,
                    "source_entry_ids": [first["entry_id"], second["entry_id"]],
                    "strong_memory": {
                        "content": "La gestione menu dipende dal corretto allineamento dei permessi utente.",
                        "context": "menu troubleshooting",
                        "type": "fact",
                        "tier": "tier-2",
                        "visibility": "shared",
                        "tags": ["menu", "permissions"],
                        "metadata": {"kind": "distilled-knowledge"},
                    },
                }
            ]
        },
    )

    first_entry = service.get_fast(first["entry_id"], actor)
    second_entry = service.get_fast(second["entry_id"], actor)
    assert result["success"] is True
    assert result["dry_run"] is True
    assert result["results"][0]["action"] == "promote"
    assert first_entry is not None and first_entry.distillation_status == FastMemoryDistillationStatus.PENDING
    assert second_entry is not None and second_entry.distillation_status == FastMemoryDistillationStatus.PENDING


@pytest.mark.asyncio
async def test_apply_fast_distillation_promotes_anchor_and_summarizes_remaining(service):
    actor = ActorContext(agent_id="agent-fast", user_id="user-fast", workspace_id="ws-test", project_id="prj-test")
    first = service.log_fast(
        {
            "content": "L'utente y vedeva solo il menu x.",
            "context": "menu troubleshooting",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "kind": "bug",
            "component": "menu-engine",
        },
        actor,
    )
    second = service.log_fast(
        {
            "content": "L'utente z vedeva solo il menu x.",
            "context": "menu troubleshooting",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "kind": "bug",
            "component": "menu-engine",
        },
        actor,
    )

    result = await service.apply_fast_distillation(
        actor=actor,
        reason="apply reviewed distillation output",
        dry_run=False,
        payload={
            "decisions": [
                {
                    "cluster_id": "cluster-menu",
                    "action": "promote",
                    "summary": "La gestione menu dipende dai permessi utente consolidati.",
                    "confidence": 0.91,
                    "source_entry_ids": [first["entry_id"], second["entry_id"]],
                    "strong_memory": {
                        "content": "La gestione menu funziona solo se il profilo utente e la tabella permessi sono riallineati.",
                        "context": "menu troubleshooting",
                        "type": "fact",
                        "tier": "tier-2",
                        "visibility": "shared",
                        "tags": ["menu", "permissions"],
                        "metadata": {"kind": "distilled-knowledge"},
                    },
                }
            ]
        },
    )

    first_entry = service.get_fast(first["entry_id"], actor)
    second_entry = service.get_fast(second["entry_id"], actor)
    promoted_entry = service.get(result["results"][0]["promoted_entry_id"], actor)
    assert result["dry_run"] is False
    assert first_entry is not None and first_entry.distillation_status == FastMemoryDistillationStatus.PROMOTED
    assert second_entry is not None and second_entry.distillation_status == FastMemoryDistillationStatus.SUMMARIZED
    assert promoted_entry is not None
    assert promoted_entry.content.startswith("La gestione menu funziona")
    assert promoted_entry.metadata["fast_memory_origin"]["entry_id"] == first["entry_id"]


@pytest.mark.asyncio
async def test_apply_fast_distillation_rejects_invalid_contract(service):
    actor = ActorContext(agent_id="agent-fast", user_id="user-fast", workspace_id="ws-test", project_id="prj-test")

    with pytest.raises(ValueError, match="payload.decisions must contain at least one decision"):
        await service.apply_fast_distillation(
            actor=actor,
            reason="invalid payload",
            dry_run=True,
            payload={"decisions": []},
        )
