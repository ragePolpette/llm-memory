from __future__ import annotations

import json

import pytest
from mcp.server import Server
from mcp.types import CallToolRequest

from src.bootstrap import build_runtime
from src.mcp_server.tools import register_tools


def _tool_payload(result) -> dict:
    return json.loads(result.root.content[0].text)


@pytest.mark.asyncio
async def test_verified_golden_path_for_core_memory_tools(runtime, test_config):
    server = Server("llm-memory-golden-path")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]

    add_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.add",
                "arguments": {
                    "agent_id": "golden-agent",
                    "content": "Il golden path deve coprire add search export e import.",
                    "context": "golden-path",
                    "type": "fact",
                    "tier": "tier-2",
                    "visibility": "shared",
                },
            }
        )
    )
    add_payload = _tool_payload(add_result)
    assert add_payload["success"] is True
    entry_id = add_payload["entry_id"]

    search_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.search",
                "arguments": {
                    "agent_id": "golden-agent",
                    "query": "golden path add search export import",
                    "limit": 5,
                },
            }
        )
    )
    search_payload = _tool_payload(search_result)
    assert search_payload["count"] >= 1
    assert any(bundle["entry_id"] == entry_id for bundle in search_payload["bundles"])

    export_result = await handler(
        CallToolRequest(
            params={
                "name": "memory.export",
                "arguments": {
                    "agent_id": "golden-agent",
                    "path": "golden-path.jsonl",
                    "format": "jsonl",
                },
            }
        )
    )
    export_payload = _tool_payload(export_result)
    assert export_payload["count"] >= 1

    second_config = test_config.model_copy(
        update={
            "sqlite_db_path": test_config.sqlite_db_path.parent / "golden-import.db",
        }
    )
    second_runtime = build_runtime(second_config)
    second_server = Server("llm-memory-golden-import")
    register_tools(second_server, second_runtime.service)
    second_handler = second_server.request_handlers[CallToolRequest]

    import_result = await second_handler(
        CallToolRequest(
            params={
                "name": "memory.import",
                "arguments": {
                    "agent_id": "golden-agent",
                    "path": "golden-path.jsonl",
                    "format": "jsonl",
                },
            }
        )
    )
    import_payload = _tool_payload(import_result)
    assert import_payload["imported"] >= 1

    imported_search_result = await second_handler(
        CallToolRequest(
            params={
                "name": "memory.search",
                "arguments": {
                    "agent_id": "golden-agent",
                    "query": "golden path add search export import",
                    "limit": 5,
                },
            }
        )
    )
    imported_search_payload = _tool_payload(imported_search_result)
    assert imported_search_payload["count"] >= 1
