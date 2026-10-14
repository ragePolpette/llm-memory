from __future__ import annotations

import json

import pytest
from mcp.server import Server
from mcp.types import CallToolRequest, ListToolsRequest

from src.mcp_server.tools import MAX_LIMIT, _bounded_limit, register_tools
from src.service.memory_service import ActorContext

SECRET = "Il progetto usa SQLite locale come backend persistente del runtime."


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [({}, 10), ({"limit": 5}, 5), ({"limit": 10_000}, MAX_LIMIT), ({"limit": 0}, 1), ({"limit": -3}, 1), ({"limit": "7"}, 7)],
)
def test_bounded_limit_clamps(arguments, expected):
    assert _bounded_limit(arguments, 10) == expected


@pytest.mark.parametrize("value", ["abc", True, [1]])
def test_bounded_limit_rejects_non_integers(value):
    with pytest.raises(ValueError, match="limit must be an integer"):
        _bounded_limit({"limit": value}, 10)


@pytest.mark.asyncio
async def test_search_tool_never_passes_more_than_max_limit(runtime, monkeypatch):
    server = Server("limits")
    register_tools(server, runtime.service)
    handler = server.request_handlers[CallToolRequest]
    seen: list[int] = []

    async def fake_search(**kwargs):
        seen.append(kwargs["limit"])
        return []

    monkeypatch.setattr(runtime.service, "search", fake_search)
    for name, extra in (
        ("memory.search", {"query": "x"}),
        ("memory.list_fast", {}),
        ("memory.rank_fast_candidates", {}),
    ):
        result = await handler(
            CallToolRequest(params={"name": name, "arguments": {"agent_id": "a", "limit": 100_000, **extra}})
        )
        if not result.root.isError:
            json.loads(result.root.content[0].text)
    assert all(limit <= MAX_LIMIT for limit in seen)

    listed = await server.request_handlers[ListToolsRequest](ListToolsRequest(method="tools/list"))
    tools = listed.root.tools
    for tool in tools:
        limit_schema = tool.inputSchema.get("properties", {}).get("limit")
        if limit_schema:
            assert limit_schema["maximum"] <= MAX_LIMIT


@pytest.mark.asyncio
async def test_activity_log_does_not_leak_content(service, capsys):
    actor = ActorContext(agent_id="agent-log", user_id="u", workspace_id="ws-test", project_id="prj-test")
    result = await service.add(
        {"content": SECRET, "context": "architecture", "agent_id": actor.agent_id, "visibility": "shared"},
        actor,
    )
    service.log_fast({"content": "Errore episodico riservato nel parser.", "agent_id": actor.agent_id}, actor)
    await service.search("backend persistente riservato", actor)
    service.invalidate(target_ids=[result["entry_id"]], actor=actor, reason="motivo riservato dettagliato")

    stderr = capsys.readouterr().err
    activity = [json.loads(line.split(" ", 1)[1]) for line in stderr.splitlines() if line.startswith("[MCP_ACTIVITY]")]
    assert activity
    assert "SQLite locale" not in stderr
    assert "Errore episodico" not in stderr
    assert "backend persistente riservato" not in stderr
    assert "motivo riservato" not in stderr
    write_in = next(record for record in activity if record["event"] == "write_in")
    assert set(write_in["content"]) == {"sha256_12", "chars"}
    assert write_in["content"]["chars"] == len(SECRET)
