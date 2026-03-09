"""Tests for deterministic self-eval scoring metadata."""

from __future__ import annotations

import json

import pytest
from mcp.server import Server
from mcp.types import CallToolRequest

from src.config import MemoryScope
from src.mcp_server.tools import register_tools
from src.models import ScopeRef
from src.service.importance_scoring import build_importance_metadata
from src.service.memory_service import ActorContext, MemoryInputError


def test_context_hash_is_deterministic_for_equivalent_payload():
    scope = ScopeRef(workspace_id="ws-a", project_id="prj-a", user_id="u-1", agent_id="agent-1")

    payload_a = {
        "writer_model": "sonnet-4",
        "context_fingerprint": {
            "conversation_id": "conv-9",
            "task_id": "task-7",
            "retrieved_ids": ["c", "a", "b", "a"],
            "tool_trace_fingerprint": {"step2": "y", "step1": "x"},
            "prompt_fingerprint": ["part-2", "part-1"],
        },
        "importance": {"confidence": 0.2, "inference_level": 3},
    }
    payload_b = {
        "writer_model": "sonnet-4",
        "context_fingerprint": {
            "conversation_id": "conv-9",
            "task_id": "task-7",
            "retrieved_ids": ["b", "c", "a"],
            "tool_trace_fingerprint": {"step1": "x", "step2": "y"},
            "prompt_fingerprint": ["part-2", "part-1"],
        },
        "importance": {"confidence": 0.2, "inference_level": 3},
    }

    meta_a = build_importance_metadata(
        payload=payload_a,
        scope=scope,
        visibility=MemoryScope.SHARED,
        top_similarities=[0.1],
        novelty_computed=True,
        event_ts_utc="2026-03-05T12:00:00+00:00",
        actor_agent_id="agent-1",
    )
    meta_b = build_importance_metadata(
        payload=payload_b,
        scope=scope,
        visibility=MemoryScope.SHARED,
        top_similarities=[0.1],
        novelty_computed=True,
        event_ts_utc="2026-03-05T12:00:00+00:00",
        actor_agent_id="agent-1",
    )

    assert meta_a["context_hash"] == meta_b["context_hash"]
    assert len(meta_a["context_hash"]) == 16
    assert meta_a["context_fingerprint"]["retrieved_ids"] == ["a", "b", "c"]


def test_confidence_path_scoring_with_negative_impact():
    scope = ScopeRef(workspace_id="ws", project_id="prj")
    payload = {
        "writer_model": "gpt-5",
        "importance": {
            "confidence": 0.2,
            "tool_steps": 10,
            "correction_count": 5,
            "inference_level": 5,
            "negative_impact": 1.0,
        },
    }

    meta = build_importance_metadata(
        payload=payload,
        scope=scope,
        visibility=MemoryScope.SHARED,
        top_similarities=[0.3, 0.1],
        novelty_computed=True,
        event_ts_utc="2026-03-05T12:00:00+00:00",
        actor_agent_id="agent-score",
    )

    assert meta["surprise_source"] == "confidence"
    assert meta["importance_score"] == 100
    assert meta["importance_class"] == "high"
    assert meta["inference_score"] == 1.0


@pytest.mark.asyncio
async def test_service_add_enforcement_rejects_missing_fields(service):
    actor = ActorContext(agent_id="agent-enf", user_id="user-enf", workspace_id="ws-test", project_id="prj-test")
    service.config.self_eval_enforced = True

    with pytest.raises(MemoryInputError) as exc:
        await service.add(
            {
                "content": "Memoria incompleta da rifiutare",
                "context": "invalid",
                "agent_id": actor.agent_id,
                "tier": "tier-1",
                "type": "fact",
                "visibility": "shared",
            },
            actor,
        )
    assert exc.value.code == "MISSING_REQUIRED_FIELDS"
    assert "context_fingerprint" in exc.value.missing_fields
    assert "importance" in exc.value.missing_fields


@pytest.mark.asyncio
async def test_service_add_persists_self_eval_metadata(service):
    actor = ActorContext(agent_id="agent-meta", user_id="user-meta", workspace_id="ws-test", project_id="prj-test")

    result = await service.add(
        {
            "content": "Evento critico osservato in pipeline.",
            "context": "ops",
            "agent_id": actor.agent_id,
            "tier": "tier-2",
            "type": "fact",
            "visibility": "shared",
            "writer_model": "gpt-5",
            "context_fingerprint": {
                "conversation_id": "conv-meta",
                "task_id": "task-meta",
                "retrieved_ids": ["doc-2", "doc-1"],
                "tool_trace_fingerprint": {"tool": "db_read"},
                "prompt_fingerprint": "prompt-meta",
            },
            "importance": {
                "confidence": 0.3,
                "tool_steps": 4,
                "correction_count": 1,
                "inference_level": 3,
                "negative_impact": 0.6,
            },
            "is_external": True,
        },
        actor,
    )

    entry = service.get(result["entry_id"], actor)
    assert entry is not None
    assert entry.metadata["writer_model"] == "gpt-5"
    assert entry.metadata["writer_agent_id"] == actor.agent_id
    assert len(entry.metadata["context_hash"]) == 16
    assert entry.metadata["importance_score"] >= 0
    assert entry.metadata["novelty_computed"] in {True, False}
    assert entry.metadata["is_external"] is True


@pytest.mark.asyncio
async def test_service_add_sets_novelty_computed_false_on_similarity_failure(service):
    actor = ActorContext(agent_id="agent-nov", user_id="user-nov", workspace_id="ws-test", project_id="prj-test")
    service.config.dedup_semantic_enabled = False

    def _boom(*args, **kwargs):
        raise RuntimeError("sim-fail")

    service.vector_store.search = _boom  # type: ignore[method-assign]

    result = await service.add(
        {
            "content": "Memoria con fallback novelty",
            "context": "fallback",
            "agent_id": actor.agent_id,
            "tier": "tier-1",
            "type": "fact",
            "visibility": "shared",
            "writer_model": "gpt-5",
            "context_fingerprint": {
                "conversation_id": "conv-fallback",
                "task_id": "task-fallback",
                "retrieved_ids": [],
                "tool_trace_fingerprint": "trace",
                "prompt_fingerprint": "prompt",
            },
            "importance": {
                "self_rating": 0.4,
                "inference_level": 1,
            },
        },
        actor,
    )

    entry = service.get(result["entry_id"], actor)
    assert entry is not None
    assert entry.metadata["novelty_computed"] is False
    assert entry.metadata["novelty_score"] == 1.0


@pytest.mark.asyncio
async def test_call_tool_returns_json_error_payload_for_input_validation(runtime):
    runtime.service.config.self_eval_enforced = True
    server = Server("llm-memory-test")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    request = CallToolRequest(
        params={
            "name": "memory.add",
            "arguments": {
                "content": "memoria incompleta",
                "agent_id": "agent-call-tool",
                "visibility": "shared",
            },
        }
    )
    result = await handler(request)

    assert result.root.isError is True
    payload = json.loads(result.root.content[0].text)
    assert payload["error_type"] == "memory_input_error"
    assert payload["code"] == "MISSING_REQUIRED_FIELDS"
    assert "context_fingerprint" in payload["missing_fields"]


@pytest.mark.asyncio
async def test_memory_about_exposes_self_eval_and_usage_guide(runtime):
    runtime.service.config.self_eval_enforced = True
    server = Server("llm-memory-test")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    request = CallToolRequest(params={"name": "memory.about", "arguments": {}})
    result = await handler(request)
    payload = json.loads(result.root.content[0].text)

    assert payload["self_eval_enforced"] is True
    assert "what_to_store" in payload
    assert "how_to_fill_memory_add" in payload
