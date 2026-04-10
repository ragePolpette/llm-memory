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
