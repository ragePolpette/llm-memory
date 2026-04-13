from __future__ import annotations

import json

import pytest
from mcp.server import Server
from mcp.types import CallToolRequest

from src.mcp_server.tools import register_tools


def _tool_payload(result) -> dict:
    return json.loads(result.root.content[0].text)


@pytest.mark.asyncio
async def test_log_and_list_fast_memory_tools(runtime):
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    log_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.log_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "content": "Tentativo di fix rapido sul parser Markdown.",
                    "context": "debug markdown parser",
                    "event_type": "fix_attempt",
                    "session_id": "session-fast-1",
                    "tags": ["debug", "markdown"],
                },
            }
        )
    )
    log_payload = _tool_payload(log_result)

    assert log_payload["success"] is True
    assert log_payload["event_type"] == "fix_attempt"

    list_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.list_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "event_type": "fix_attempt",
                    "limit": 10,
                },
            }
        )
    )
    list_payload = _tool_payload(list_result)

    assert list_payload["count"] >= 1
    assert any(entry["id"] == log_payload["entry_id"] for entry in list_payload["entries"])


@pytest.mark.asyncio
async def test_get_fast_memory_tool_returns_entry(runtime):
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    log_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.log_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "content": "Errore ricorrente nel mapping degli allegati.",
                    "event_type": "incident",
                },
            }
        )
    )
    log_payload = _tool_payload(log_result)

    get_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.get_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "entry_id": log_payload["entry_id"],
                },
            }
        )
    )
    get_payload = _tool_payload(get_result)

    assert get_payload["entry"]["id"] == log_payload["entry_id"]
    assert get_payload["entry"]["event_type"] == "incident"


@pytest.mark.asyncio
async def test_log_fast_memory_tool_accepts_structured_metadata(runtime):
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    log_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.log_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "content": "L'utente y vedeva solo il menu x.",
                    "event_type": "incident",
                    "kind": "bug",
                    "product_area": "authorization",
                    "component": "menu-engine",
                    "entity_refs": ["user:y", "menu:x"],
                    "action_taken": "Update tabella permessi utente.",
                    "outcome": "Menu ripristinati.",
                    "generalizable": "yes",
                    "commands": ["sqlcmd -i perm_fix.sql"],
                },
            }
        )
    )
    log_payload = _tool_payload(log_result)

    get_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.get_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "entry_id": log_payload["entry_id"],
                },
            }
        )
    )
    get_payload = _tool_payload(get_result)
    structured = get_payload["entry"]["metadata"]["structured_context"]

    assert structured["kind"] == "bug"
    assert structured["product_area"] == "authorization"
    assert structured["component"] == "menu-engine"
    assert structured["entity_refs"] == ["user:y", "menu:x"]
    assert structured["generalizable"] == "yes"


@pytest.mark.asyncio
async def test_log_fast_tool_returns_json_validation_error(runtime):
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    result = await handler(
        CallToolRequest(
            params={
                "name": "memory.log_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "content": "   ",
                },
            }
        )
    )
    payload = _tool_payload(result)

    assert result.root.isError is True
    assert payload["error_type"] == "memory_input_error"
    assert payload["code"] == "INVALID_CONTENT"


@pytest.mark.asyncio
async def test_memory_about_exposes_fast_memory_tools(runtime):
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    result = await handler(CallToolRequest(params={"name": "memory.about", "arguments": {}}))
    payload = _tool_payload(result)

    assert "memory.log_fast" in payload["tool_map"]["generic"]
    assert "memory.list_fast" in payload["tool_map"]["generic"]
    assert "memory.get_fast" in payload["tool_map"]["generic"]
    assert "memory.rank_fast_candidates" in payload["tool_map"]["generic"]
    assert "memory.prepare_fast_distillation" in payload["tool_map"]["generic"]
    assert "memory.apply_fast_distillation" in payload["tool_map"]["generic"]
    assert "memory.summarize_fast" in payload["tool_map"]["generic"]
    assert "memory.discard_fast" in payload["tool_map"]["generic"]
    assert "memory.promote_fast" in payload["tool_map"]["generic"]


@pytest.mark.asyncio
async def test_summarize_fast_memory_tool_updates_entry(runtime):
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    log_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.log_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "content": "Retry ripetuti sul parser CSV.",
                    "event_type": "retry",
                },
            }
        )
    )
    log_payload = _tool_payload(log_result)

    summarize_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.summarize_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "entry_id": log_payload["entry_id"],
                    "summary": "Retry rumorosi sul parser CSV senza nuovi segnali.",
                    "reason": "manual triage",
                    "cluster_id": "cluster-csv",
                    "resolved": True,
                },
            }
        )
    )
    summarize_payload = _tool_payload(summarize_result)

    assert summarize_payload["success"] is True
    assert summarize_payload["distillation_status"] == "summarized"
    assert summarize_payload["cluster_id"] == "cluster-csv"
    assert summarize_payload["resolved"] is True


@pytest.mark.asyncio
async def test_discard_fast_memory_tool_marks_entry_discarded(runtime):
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    log_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.log_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "content": "Burst locale dello stesso fallback.",
                    "event_type": "retry",
                },
            }
        )
    )
    log_payload = _tool_payload(log_result)

    discard_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.discard_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "entry_id": log_payload["entry_id"],
                    "reason": "noise-only pattern",
                },
            }
        )
    )
    discard_payload = _tool_payload(discard_result)

    assert discard_payload["success"] is True
    assert discard_payload["distillation_status"] == "discarded"


@pytest.mark.asyncio
async def test_promote_fast_memory_tool_creates_strong_memory(runtime):
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    log_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.log_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "content": "Errore ricorrente nell'import incrementale dopo resume.",
                    "context": "incident review",
                    "event_type": "incident",
                    "recurrence_count": 4,
                    "metadata": {"importance_score": 35, "distinct_session_count": 3},
                },
            }
        )
    )
    log_payload = _tool_payload(log_result)

    promote_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.promote_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "entry_id": log_payload["entry_id"],
                    "reason": "recurring issue became reusable knowledge",
                    "target_tier": "tier-2",
                    "memory_type": "fact",
                    "summary": "L'import incrementale fallisce dopo resume se il checkpoint locale non viene riallineato.",
                    "confidence": 0.85,
                },
            }
        )
    )
    promote_payload = _tool_payload(promote_result)

    assert promote_payload["success"] is True
    assert promote_payload["distillation_status"] == "promoted"
    assert promote_payload["target_tier"] == "tier-2"
    assert promote_payload["promoted_entry_id"]


@pytest.mark.asyncio
async def test_rank_fast_candidates_tool_returns_clustered_candidates(runtime):
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    for session_id, user_ref in (("session-a", "user:y"), ("session-b", "user:z")):
        await handler(
            CallToolRequest(
                params={
                    "name": "memory.log_fast",
                    "arguments": {
                        "agent_id": "agent-fast-tool",
                        "content": f"L'utente {user_ref} vedeva solo il menu x.",
                        "event_type": "incident",
                        "session_id": session_id,
                        "kind": "bug",
                        "product_area": "authorization",
                        "component": "menu-engine",
                        "entity_refs": [user_ref, "menu:x"],
                        "recurrence_count": 2,
                    },
                }
            )
        )

    result = await handler(
        CallToolRequest(
            params={
                "name": "memory.rank_fast_candidates",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "limit": 10,
                },
            }
        )
    )
    payload = _tool_payload(result)

    assert payload["candidates"]["count"] >= 1
    first = payload["candidates"]["items"][0]
    assert first["member_count"] >= 2
    assert first["component"] == "menu-engine"


@pytest.mark.asyncio
async def test_prepare_fast_distillation_tool_requires_feature_flag(runtime):
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    result = await handler(
        CallToolRequest(
            params={
                "name": "memory.prepare_fast_distillation",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "reason": "prepare top candidate pack",
                },
            }
        )
    )
    payload = _tool_payload(result)

    assert result.root.isError is True
    assert payload["error_type"] == "permission_error"
    assert "FAST_MEMORY_AGENT_DISTILLATION_ENABLED" in payload["message"]


@pytest.mark.asyncio
async def test_prepare_fast_distillation_tool_returns_candidate_pack_when_enabled(runtime):
    runtime.service.config.fast_memory_agent_distillation_enabled = True
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    await handler(
        CallToolRequest(
            params={
                "name": "memory.log_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "content": "L'utente y vedeva solo il menu x.",
                    "context": "menu troubleshooting",
                    "event_type": "incident",
                    "session_id": "session-a",
                    "kind": "bug",
                    "product_area": "authorization",
                    "component": "menu-engine",
                    "entity_refs": ["user:y", "menu:x"],
                    "action_taken": "Update tabella permessi utente.",
                    "outcome": "Menu ripristinati.",
                    "generalizable": "yes",
                    "recurrence_count": 2,
                },
            }
        )
    )

    result = await handler(
        CallToolRequest(
            params={
                "name": "memory.prepare_fast_distillation",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "reason": "prepare agentic distillation",
                    "top_k": 1,
                },
            }
        )
    )
    payload = _tool_payload(result)

    assert payload["prepared_count"] == 1
    assert payload["protection"]["mode"] == "review_then_apply"
    assert "Return JSON only" in payload["prompt"]
    assert payload["candidates"][0]["source_entries"][0]["structured_context"]["component"] == "menu-engine"


@pytest.mark.asyncio
async def test_apply_fast_distillation_tool_requires_feature_flag(runtime):
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    result = await handler(
        CallToolRequest(
            params={
                "name": "memory.apply_fast_distillation",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "reason": "apply reviewed distillation",
                    "payload": {"decisions": []},
                },
            }
        )
    )
    payload = _tool_payload(result)

    assert result.root.isError is True
    assert payload["error_type"] == "permission_error"
    assert "FAST_MEMORY_AGENT_DISTILLATION_APPLY_ENABLED" in payload["message"]


@pytest.mark.asyncio
async def test_apply_fast_distillation_tool_dry_run_by_default(runtime):
    runtime.service.config.fast_memory_agent_distillation_apply_enabled = True
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    first = await handler(
        CallToolRequest(
            params={
                "name": "memory.log_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "content": "L'utente y vedeva solo il menu x.",
                    "event_type": "incident",
                    "component": "menu-engine",
                },
            }
        )
    )
    second = await handler(
        CallToolRequest(
            params={
                "name": "memory.log_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "content": "L'utente z vedeva solo il menu x.",
                    "event_type": "incident",
                    "component": "menu-engine",
                },
            }
        )
    )
    first_id = _tool_payload(first)["entry_id"]
    second_id = _tool_payload(second)["entry_id"]

    result = await handler(
        CallToolRequest(
            params={
                "name": "memory.apply_fast_distillation",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "reason": "preview reviewed distillation",
                    "payload": {
                        "decisions": [
                            {
                                "cluster_id": "cluster-menu",
                                "action": "promote",
                                "summary": "La gestione menu dipende dai permessi utente consolidati.",
                                "confidence": 0.9,
                                "source_entry_ids": [first_id, second_id],
                                "strong_memory": {
                                    "content": "La gestione menu dipende dal corretto allineamento dei permessi utente.",
                                    "context": "menu troubleshooting",
                                    "type": "fact",
                                    "tier": "tier-2",
                                    "visibility": "shared",
                                    "tags": ["menu", "permissions"],
                                    "metadata": {},
                                },
                            }
                        ]
                    },
                },
            }
        )
    )
    payload = _tool_payload(result)

    assert payload["success"] is True
    assert payload["dry_run"] is True
    assert payload["results"][0]["action"] == "promote"


@pytest.mark.asyncio
async def test_apply_fast_distillation_tool_applies_cluster_mutation(runtime):
    runtime.service.config.fast_memory_agent_distillation_apply_enabled = True
    server = Server("llm-memory-fast-tools")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    first = await handler(
        CallToolRequest(
            params={
                "name": "memory.log_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "content": "L'utente y vedeva solo il menu x.",
                    "context": "menu troubleshooting",
                    "event_type": "incident",
                    "kind": "bug",
                    "component": "menu-engine",
                },
            }
        )
    )
    second = await handler(
        CallToolRequest(
            params={
                "name": "memory.log_fast",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "content": "L'utente z vedeva solo il menu x.",
                    "context": "menu troubleshooting",
                    "event_type": "incident",
                    "kind": "bug",
                    "component": "menu-engine",
                },
            }
        )
    )
    first_id = _tool_payload(first)["entry_id"]
    second_id = _tool_payload(second)["entry_id"]

    result = await handler(
        CallToolRequest(
            params={
                "name": "memory.apply_fast_distillation",
                "arguments": {
                    "agent_id": "agent-fast-tool",
                    "reason": "apply reviewed distillation",
                    "dry_run": False,
                    "payload": {
                        "decisions": [
                            {
                                "cluster_id": "cluster-menu",
                                "action": "promote",
                                "summary": "La gestione menu dipende dai permessi utente consolidati.",
                                "confidence": 0.9,
                                "source_entry_ids": [first_id, second_id],
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
                },
            }
        )
    )
    payload = _tool_payload(result)

    assert payload["success"] is True
    assert payload["dry_run"] is False
    assert payload["results"][0]["promoted_entry_id"]
