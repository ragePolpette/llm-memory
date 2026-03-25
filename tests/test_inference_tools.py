from __future__ import annotations

import json

import pytest
from mcp.server import Server
from mcp.types import CallToolRequest

from src.mcp_server.tools import register_tools
from src.service.memory_service import ActorContext


@pytest.mark.asyncio
async def test_memory_about_exposes_harness_tool_map(runtime):
    server = Server("llm-memory-test")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    result = await handler(CallToolRequest(params={"name": "memory.about", "arguments": {}}))
    payload = json.loads(result.root.content[0].text)

    assert "tool_map" in payload
    assert "capture_inference_memory" in payload["tool_map"]["harness"]
    assert payload["harness_scope_guide"]["global"].startswith("Usa per conoscenza trasversale")


@pytest.mark.asyncio
async def test_capture_and_search_inference_memory_project_scope(runtime):
    server = Server("llm-memory-test")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    capture_result = await handler(
        CallToolRequest(
            params={
                "name": "capture_inference_memory",
                "arguments": {
                    "namespace": "ticket-harness",
                    "phase": "triage",
                    "ticket_key": "BPO-101",
                    "product_target": "legacy",
                    "repo_target": "bpopilot",
                    "content": "La logica ordini vive nel controller storico e nel relativo service.",
                    "tags": ["orders", "mapping"],
                    "scope": "project",
                    "agent_id": "agent-harness",
                },
            }
        )
    )
    capture_payload = json.loads(capture_result.root.content[0].text)

    assert capture_payload["stored"] is True
    assert capture_payload["memory_id"]

    search_result = await handler(
        CallToolRequest(
            params={
                "name": "search_inference_memory",
                "arguments": {
                    "namespace": "ticket-harness",
                    "query": "ordini controller storico",
                    "product_target": "legacy",
                    "scope": "project",
                    "agent_id": "agent-harness",
                },
            }
        )
    )
    search_payload = json.loads(search_result.root.content[0].text)

    assert search_payload["count"] == 1
    item = search_payload["items"][0]
    assert item["content"].startswith("La logica ordini vive")
    assert item["product_target"] == "legacy"
    assert item["scope"] == "project"


@pytest.mark.asyncio
async def test_capture_inference_memory_global_scope_uses_global_bucket(runtime):
    server = Server("llm-memory-test")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    result = await handler(
        CallToolRequest(
            params={
                "name": "capture_inference_memory",
                "arguments": {
                    "namespace": "ticket-harness",
                    "phase": "execution",
                    "product_target": "shared",
                    "repo_target": "cross-product",
                    "content": "Per PR discovery preferire sempre il lookup esplicito del branch sorgente.",
                    "scope": "global",
                    "agent_id": "agent-harness",
                },
            }
        )
    )
    payload = json.loads(result.root.content[0].text)

    assert payload["stored"] is True
    entry = runtime.service.get(
        payload["memory_id"],
        ActorContext(
            agent_id="agent-harness",
            user_id=None,
            workspace_id=runtime.service.config.default_workspace_id,
            project_id=runtime.service.config.default_project_id,
        ),
    )
    assert entry is not None
    assert entry.scope.scope_level.value == "global"
    assert entry.visibility.value == "global"
