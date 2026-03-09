from __future__ import annotations

import pytest
from mcp.server import Server
from mcp.types import CallToolRequest

from src.mcp_server.tools import register_tools
from src.models import EntryType
from src.service.memory_service import ActorContext


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "context", "category", "reason_code"),
    [
        ("ciao", "", "small_talk", "SMALL_TALK"),
        ("Ho finito il parser e ora chiudo il task.", "status", "task_progress", "TASK_PROGRESS"),
        ("Debug note: stack trace temporaneo da ricontrollare.", "debug", "working_memory", "DEBUG_NOTE"),
        ("```python\nprint('hello')\n```", "snippet", "code_snippet", "CODE_SNIPPET"),
        ("Ricordamelo in questa sessione corrente.", "session", "transient_context", "TRANSIENT_CONTEXT"),
    ],
)
async def test_rejects_non_persistible_content(service, content: str, context: str, category: str, reason_code: str):
    actor = ActorContext(agent_id="agent-policy", user_id="user-policy", workspace_id="ws-test", project_id="prj-test")

    result = await service.add(
        {
            "content": content,
            "context": context,
            "agent_id": actor.agent_id,
            "visibility": "shared",
        },
        actor,
    )

    assert result["success"] is False
    assert result["rejected"] is True
    assert result["decision"]["category"] == category
    assert reason_code in result["decision"]["reason_codes"]
    assert service.list_entries(actor, limit=10) == []


@pytest.mark.asyncio
async def test_reject_audit_contains_reason_codes(service):
    actor = ActorContext(agent_id="agent-audit", user_id="user-audit", workspace_id="ws-test", project_id="prj-test")

    result = await service.add(
        {
            "content": "ok",
            "context": "",
            "agent_id": actor.agent_id,
        },
        actor,
    )

    assert result["success"] is False

    audits = service.store.list_audit(limit=10)
    assert audits
    latest = audits[0]
    assert latest.action == "write_attempt"
    assert latest.payload["decision"] == "rejected"
    assert "reason_codes" in latest.payload
    assert latest.payload["write_path"] == "add"


@pytest.mark.asyncio
async def test_accepts_semantic_memory(service):
    actor = ActorContext(agent_id="agent-sem", user_id="user-sem", workspace_id="ws-test", project_id="prj-test")

    result = await service.add(
        {
            "content": "Il progetto usa SQLite locale come backend predefinito per i metadata persistenti.",
            "context": "architecture",
            "agent_id": actor.agent_id,
            "type": "fact",
            "visibility": "shared",
        },
        actor,
    )

    assert result["success"] is True
    assert result["decision"]["category"] == "semantic_memory"


@pytest.mark.asyncio
async def test_accepts_stable_preference(service):
    actor = ActorContext(agent_id="agent-pref", user_id="user-pref", workspace_id="ws-test", project_id="prj-test")

    result = await service.add(
        {
            "content": "Preferisco usare payload JSON compatti e evitare campi ridondanti nei tool MCP.",
            "context": "preference",
            "agent_id": actor.agent_id,
            "visibility": "shared",
        },
        actor,
    )

    assert result["success"] is True
    assert result["decision"]["category"] == "stable_preference"


@pytest.mark.asyncio
async def test_accepts_architectural_decision(service):
    actor = ActorContext(agent_id="agent-arch", user_id="user-arch", workspace_id="ws-test", project_id="prj-test")

    result = await service.add(
        {
            "content": "Decisione architetturale: usiamo MemoryService come unico orchestratore della persistenza.",
            "context": "decision",
            "agent_id": actor.agent_id,
            "type": "decision",
            "visibility": "shared",
        },
        actor,
    )

    assert result["success"] is True
    assert result["decision"]["category"] == "architectural_decision"


@pytest.mark.asyncio
async def test_import_rejects_formally_valid_but_non_persistible_entry(service):
    actor = ActorContext(agent_id="agent-import", user_id="user-import", workspace_id="ws-test", project_id="prj-test")

    jsonl_path = service.config.import_export_base_dir / "reject-import.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        '{"id":"e-noise","tier":"tier-2","scope":{"workspace_id":"ws-test","project_id":"prj-test"},"visibility":"shared","source":"test","type":"fact","status":"active","content":"ciao","context":"chat","tags":[],"sensitivity_tags":[],"metadata":{},"links":[],"confidence":0.7,"created_at":"2026-01-01T00:00:00+00:00","updated_at":"2026-01-01T00:00:00+00:00","content_hash":"abc","embedding_version_id":null,"encrypted":false,"redacted":false}\n',
        encoding="utf-8",
    )

    result = await service.import_data(jsonl_path, "jsonl", actor)

    assert result.imported == 0
    assert result.rejected == 1


@pytest.mark.asyncio
async def test_mcp_runtime_rejects_removed_legacy_tool(runtime):
    server = Server("llm-memory-test")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    request = CallToolRequest(
        params={
            "name": "memory_write",
            "arguments": {
                "content": "ciao",
                "context": "legacy",
                "agent_id": "agent-legacy",
            },
        }
    )
    result = await handler(request)

    assert result.root.isError is True
    assert "Unknown tool: memory_write" in result.root.content[0].text


@pytest.mark.asyncio
async def test_invalidation_entries_are_internal_governance(service):
    actor = ActorContext(agent_id="agent-internal", user_id="user-internal", workspace_id="ws-test", project_id="prj-test")

    created = await service.add(
        {
            "content": "Il sistema usa invalidation entries per tracciare correzioni persistenti.",
            "context": "architecture",
            "agent_id": actor.agent_id,
            "visibility": "shared",
            "type": "fact",
        },
        actor,
    )
    result = service.invalidate([created["entry_id"]], actor=actor, reason="obsolete fact")

    assert result["count"] == 1

    entries = service.store.export_entries()
    invalidation_entries = [entry for entry in entries if entry.type == EntryType.INVALIDATED]
    assert invalidation_entries
    internal = invalidation_entries[0]
    assert internal.source == "internal_governance"
    decision = internal.metadata["persistence_decision"]
    assert decision["decision"] == "accepted_internal"
    assert decision["source"] == "internal_governance"
    assert decision["write_path"] == "invalidate"
    assert decision["source_type"] == "internal_governance"
    assert internal.metadata["internal_reason"] == "obsolete fact"
    audits = service.store.list_audit(entry_id=internal.id, limit=10)
    assert any(
        audit.action == "write_attempt"
        and audit.payload["decision"] == "accepted_internal"
        and audit.payload["internal_reason"] == "obsolete fact"
        and audit.payload["write_path"] == "invalidate"
        for audit in audits
    )


@pytest.mark.asyncio
async def test_promote_merge_creates_auditable_internal_record(service):
    actor = ActorContext(agent_id="agent-merge", user_id="user-merge", workspace_id="ws-test", project_id="prj-test")

    entry_ids = []
    for content in (
        "Il progetto usa MemoryService come orchestratore principale della persistenza.",
        "Il progetto usa SQLite come backend persistente del runtime v2.",
    ):
        result = await service.add(
            {
                "content": content,
                "context": "architecture",
                "agent_id": actor.agent_id,
                "visibility": "shared",
                "type": "fact",
                "tier": "tier-2",
            },
            actor,
        )
        entry_ids.append(result["entry_id"])

    result = service.promote(
        entry_ids=entry_ids,
        actor=actor,
        target_tier=service.config.promotion_default_target_tier,
        reason="consolidate promoted facts",
        merge=True,
        summary="Summary interno di consolidamento delle memorie promosse.",
    )

    assert result["merged_entry_id"] is not None
    merged = service.store.get_entry(result["merged_entry_id"])
    assert merged is not None
    assert merged.source == "internal_governance"
    decision = merged.metadata["persistence_decision"]
    assert decision["decision"] == "accepted_internal"
    assert decision["source"] == "internal_governance"
    assert decision["write_path"] == "promote_merge"
    assert decision["category"] == "internal_merge_summary"
    assert decision["source_type"] == "internal_governance"
    assert decision["policy_version"] == "internal-persistence-v1"
    assert merged.metadata["internal_reason"] == "consolidate promoted facts"

    audits = service.store.list_audit(entry_id=merged.id, limit=10)
    assert any(
        audit.action == "write_attempt"
        and audit.payload["decision"] == "accepted_internal"
        and audit.payload["internal_reason"] == "consolidate promoted facts"
        and audit.payload["write_path"] == "promote_merge"
        for audit in audits
    )
