"""Contract tests of the v2 surface, end to end through the real HTTP app.

What clients (Yesod first) depend on:
- GET /admin/audit: auth, response shape, filters, after_id cursor, ids in commit order;
- audit actions and payloads: note, save, supersede, create_project, every write payload
  with project_id at the top level, rejected writes audited without their content;
- the seven MCP tools: memory.note, memory.save, memory.supersede, memory.recall,
  memory.get, memory.list, memory.projects; errors as {ok: false, error: {code, message}}.

Change these tests only together with the clients.
"""

from __future__ import annotations

import itertools
import json

import pytest
from starlette.testclient import TestClient

import src.mcp_server.http_server as http_server

TOKEN = "contract-token-0123456789abcdef"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json", **AUTH}
PROJECT = "contract-prj"
AUDIT_ITEM_KEYS = {"id", "entry_id", "action", "actor", "reason", "payload", "created_at", "payload_preview"}
TOOLS = {"memory.note", "memory.save", "memory.supersede", "memory.recall", "memory.get", "memory.list",
         "memory.projects"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_STORE_PATH", str(tmp_path / "memory-v2.db"))
    monkeypatch.setenv("MEMORY_SQLITE_PATH", str(tmp_path / "no-v1.db"))
    monkeypatch.setenv("MEMORY_EMBEDDING", "off")
    monkeypatch.setenv("MEMORY_IMPORT_EXPORT_BASE_DIR", str(tmp_path / "exchange"))
    monkeypatch.setenv("MCP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("MCP_MEMORY_HOST", "127.0.0.1")
    monkeypatch.setenv("LLM_MEMORY_API_TOKEN", TOKEN)
    with TestClient(http_server.app, base_url="http://127.0.0.1:8767") as test_client:
        yield test_client
    for name in ("runtime", "mcp_server", "sse_transport", "streamable_session_manager", "app_config"):
        monkeypatch.setattr(http_server, name, None)


class Mcp:
    """Minimal streamable-HTTP MCP client."""

    def __init__(self, client: TestClient):
        self.client = client
        self.ids = itertools.count(1)
        self.session = None
        response = self._post({
            "jsonrpc": "2.0", "id": next(self.ids), "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                       "clientInfo": {"name": "contract", "version": "0.0.1"}},
        })
        assert response.status_code == 200, response.text
        self.session = response.headers.get("mcp-session-id")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _post(self, body: dict):
        headers = dict(MCP_HEADERS)
        if self.session:
            headers["mcp-session-id"] = self.session
        return self.client.post("/mcp", json=body, headers=headers)

    def rpc(self, method: str, params: dict | None = None) -> dict:
        response = self._post({"jsonrpc": "2.0", "id": next(self.ids), "method": method, "params": params or {}})
        assert response.status_code == 200, response.text
        return response.json()["result"]

    def raw(self, name: str, arguments: dict) -> tuple[bool, dict]:
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        return bool(result.get("isError")), json.loads(result["content"][0]["text"])

    def call(self, name: str, arguments: dict) -> dict:
        error, payload = self.raw(name, {"agent_id": "yesod", **arguments})
        assert not error and payload["ok"], payload
        return payload

    def fail(self, name: str, arguments: dict) -> dict:
        error, payload = self.raw(name, {"agent_id": "yesod", **arguments})
        assert error and payload["ok"] is False, payload
        return payload["error"]


@pytest.fixture
def mcp(client):
    session = Mcp(client)
    session.call("memory.projects", {"action": "create", "project_id": PROJECT, "display_name": "Contract"})
    return session


def audit(client: TestClient, **params) -> dict:
    response = client.get("/admin/audit", params=params, headers=AUTH)
    assert response.status_code == 200, response.text
    return response.json()["audit"]


def rows(client: TestClient, **params) -> list[dict]:
    """Audit rows oldest first."""
    return list(reversed(audit(client, limit=500, **params)["items"]))


# -- tools ------------------------------------------------------------------------


def test_exactly_seven_tools_and_no_legacy_names(mcp):
    assert {tool["name"] for tool in mcp.rpc("tools/list")["tools"]} == TOOLS
    assert mcp.fail("memory.add", {})["code"] == "unknown_tool"


def test_note_then_recall_and_get(mcp, client):
    noted = mcp.call("memory.note", {"project_id": PROJECT, "content": "Il job notturno è andato in timeout.",
                                     "event_type": "error", "tags": ["nightly"]})
    assert noted["kind"] == "episode" and noted["expires_at"] and noted["seen_before"] == 0
    again = mcp.call("memory.note", {"project_id": PROJECT, "content": "Il job notturno è andato in timeout.",
                                     "event_type": "error"})
    assert again["seen_before"] == 1 and again["id"] != noted["id"]

    found = mcp.call("memory.recall", {"project_id": PROJECT, "query": "timeout job notturno"})
    assert found["mode"] == "text" and found["degraded"]  # embeddings off in this test
    assert noted["id"] in {item["id"] for item in found["items"]}

    item = mcp.call("memory.get", {"id": noted["id"]})["item"]
    assert item["kind"] == "episode" and item["agent_id"] == "yesod" and item["tags"] == ["nightly"]

    row = rows(client, action="note")[0]
    assert row["entry_id"] == noted["id"] and row["actor"] == "yesod"
    assert row["payload"]["project_id"] == PROJECT and row["payload"]["outcome"] == "created"


def test_save_supersede_and_history(mcp, client):
    note = mcp.call("memory.note", {"project_id": PROJECT, "content": "Le notifiche via email arrivano in ritardo."})
    old = mcp.call("memory.save", {"project_id": PROJECT, "type": "decision",
                                   "statement": "Le notifiche urgenti passano da email.", "why": "già disponibile"})
    new = mcp.call("memory.save", {
        "project_id": PROJECT, "type": "decision", "statement": "Le notifiche urgenti passano da ntfy.",
        "why": "email arriva in ritardo", "impact": "medium", "supersedes": old["id"],
        "evidence": [{"kind": "episode", "ref": note["id"]}, {"kind": "external", "ref": "https://example.org/pr/1"}],
    })
    assert new["created"] and new["superseded"] == old["id"]
    memory = mcp.call("memory.get", {"id": new["id"]})["item"]
    assert {"kind": "episode", "ref": note["id"]} in memory["sources"]
    assert mcp.call("memory.get", {"id": old["id"]})["item"]["superseded_by"] == new["id"]

    current = mcp.call("memory.recall", {"project_id": PROJECT, "query": "notifiche urgenti", "kinds": ["memory"]})
    assert [item["id"] for item in current["items"]] == [new["id"]]
    history = mcp.call("memory.recall", {"project_id": PROJECT, "query": "notifiche urgenti", "kinds": ["memory"],
                                         "include_closed": True})
    assert {item["id"] for item in history["items"]} == {new["id"], old["id"]}

    duplicate = mcp.call("memory.save", {"project_id": PROJECT, "type": "decision",
                                         "statement": "Le notifiche urgenti passano da ntfy.", "why": "x"})
    assert duplicate["created"] is False and duplicate["id"] == new["id"]
    outcomes = [(r["payload"]["outcome"], r["entry_id"]) for r in rows(client, action="save")]
    assert outcomes[-1] == ("duplicate", new["id"]) and ("created", new["id"]) in outcomes

    fact = mcp.call("memory.save", {"project_id": PROJECT, "type": "fact", "statement": "Il db è PostgreSQL 15."})
    closed = mcp.call("memory.supersede", {"memory_id": fact["id"], "reason": "migrato a PostgreSQL 16"})
    assert closed["closed"]
    row = rows(client, action="supersede")[-1]
    assert row["entry_id"] == fact["id"] and row["reason"] == "migrato a PostgreSQL 16"
    assert row["payload"]["project_id"] == PROJECT
    assert mcp.fail("memory.supersede", {"memory_id": fact["id"], "reason": "di nuovo"})["code"] == "conflict"


def test_validation_errors_tell_what_to_change(mcp):
    assert mcp.fail("memory.save", {"project_id": PROJECT, "type": "decision",
                                    "statement": "Usiamo ntfy."})["message"] == "why is required"
    assert mcp.fail("memory.save", {"project_id": PROJECT, "type": "opinion", "statement": "x"})["code"] == \
        "invalid_argument"
    unknown = mcp.fail("memory.note", {"project_id": "nope", "content": "x"})
    assert unknown["code"] == "not_found" and "memory.projects" in unknown["message"]
    assert mcp.fail("memory.note", {"project_id": PROJECT})["code"] == "invalid_argument"
    assert mcp.fail("memory.get", {"id": "missing"})["code"] == "not_found"
    assert mcp.fail("memory.recall", {"project_id": PROJECT, "query": "x", "limit": 0})["code"] == "invalid_argument"


def test_secrets_are_rejected_and_never_stored(mcp, client):
    key = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0" * 2
    error = mcp.fail("memory.note", {"project_id": PROJECT, "content": f"La chiave OpenAI è {key}"})
    assert error["code"] == "secret_rejected" and error["details"]["fields"] == ["content"]
    assert error["details"]["kinds"] == ["openai_key"] and key not in json.dumps(error)
    error = mcp.fail("memory.save", {"project_id": PROJECT, "type": "fact", "statement": "Il db risponde.",
                                     "how": "psql postgres://app:" + "Sup3rS3cret" + "@db.local/app"})
    assert error["details"]["fields"] == ["how"]

    rejected = rows(client, reason="secret_detected")
    assert [r["action"] for r in rejected] == ["note", "save"]
    assert all(r["payload"]["outcome"] == "rejected" and r["payload"]["project_id"] == PROJECT for r in rejected)
    assert key not in json.dumps(rows(client)) and "Sup3rS3cret" not in json.dumps(rows(client))
    assert mcp.call("memory.recall", {"project_id": PROJECT, "query": "chiave OpenAI db"})["items"] == []

    fine = mcp.call("memory.note", {"project_id": PROJECT, "content": "La chiave sta nella variabile FIREWORKS_API_KEY."})
    assert fine["id"]


def test_projects_create_is_idempotent_and_audited(mcp, client):
    again = mcp.call("memory.projects", {"action": "create", "project_id": PROJECT})
    assert again["created"] is False
    listed = {p["project_id"] for p in mcp.call("memory.projects", {})["projects"]}
    assert {PROJECT, "__global__"} <= listed
    assert [r["payload"] for r in rows(client, action="create_project") if r["payload"]["project_id"] == PROJECT] \
        == [{"project_id": PROJECT}]
    assert mcp.fail("memory.projects", {"action": "create", "project_id": "Bad Id"})["code"] == "invalid_argument"


def test_global_rules_are_recalled_in_every_project(mcp):
    mcp.call("memory.save", {"project_id": "__global__", "type": "constraint",
                             "statement": "Le dipendenze devono avere almeno 45 giorni."})
    found = mcp.call("memory.recall", {"project_id": PROJECT, "query": "dipendenze giorni"})
    assert found["items"][0]["project_id"] == "__global__"
    alone = mcp.call("memory.recall", {"project_id": PROJECT, "query": "dipendenze giorni", "include_global": False})
    assert alone["items"] == []


# -- /admin/audit -----------------------------------------------------------------


def test_audit_requires_token(client):
    assert client.get("/admin/audit").status_code == 401
    assert client.get("/admin/audit", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_audit_shape_and_filters(mcp, client):
    mcp.call("memory.note", {"project_id": PROJECT, "content": "Uno."})
    payload = audit(client, limit=10)
    assert set(payload) == {"count", "limit", "filters", "items"}
    assert set(payload["filters"]) == {"entry_id", "action", "actor", "reason", "since", "after_id"}
    assert all(set(item) == AUDIT_ITEM_KEYS for item in payload["items"])
    ids = [item["id"] for item in payload["items"]]
    assert ids == sorted(ids, reverse=True)  # newest first
    assert all(item["actor"] == "yesod" for item in audit(client, actor="yesod", limit=50)["items"])
    future = audit(client, since="2999-01-01T00:00:00Z")
    assert future["items"] == [] and future["filters"]["since"].startswith("2999-01-01")


def test_audit_after_id_is_a_forward_cursor(mcp, client):
    start = max(item["id"] for item in rows(client))
    for i in range(5):
        mcp.call("memory.note", {"project_id": PROJECT, "content": f"Evento {i}."})
    first = audit(client, after_id=start, limit=2)
    assert first["filters"]["after_id"] == start
    ids = [item["id"] for item in first["items"]]
    assert ids == sorted(ids) and ids[0] > start and len(ids) == 2
    rest = audit(client, after_id=ids[-1], limit=500)["items"]
    assert len(rest) == 3 and rest[0]["id"] > ids[-1]
    assert audit(client, after_id=rest[-1]["id"])["items"] == []


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 501}, {"limit": "x"}, {"after_id": -1},
                                    {"after_id": "x"}, {"since": "ieri"}])
def test_audit_rejects_invalid_parameters(client, params):
    response = client.get("/admin/audit", params=params, headers=AUTH)
    assert response.status_code == 400 and response.json()["error"]["type"] == "bad_request"


def test_admin_projects_counts(mcp, client):
    mcp.call("memory.note", {"project_id": PROJECT, "content": "Uno."})
    mcp.call("memory.save", {"project_id": PROJECT, "type": "fact", "statement": "Due."})
    response = client.get("/admin/projects", headers=AUTH)
    project = next(p for p in response.json()["projects"]["items"] if p["project_id"] == PROJECT)
    assert project["episodes"] == 1 and project["memories_active"] == 1


def test_list_latest_notes_and_permanent_records(mcp):
    for i in range(3):
        mcp.call("memory.note", {"project_id": PROJECT, "content": f"Run {i} ok.", "event_type": "run_summary",
                                 "metadata": {"run_id": f"r{i}"}})
    kept = mcp.call("memory.note", {"project_id": PROJECT, "content": "accepted: dedupe-1",
                                    "event_type": "proposal_verdict", "metadata": {"verdict": "accepted"},
                                    "ttl_days": 0})
    assert kept["expires_at"] is None
    runs = mcp.call("memory.list", {"project_id": PROJECT, "event_type": "run_summary", "limit": 2})
    assert [item["metadata"]["run_id"] for item in runs["items"]] == ["r2", "r1"]
    assert runs["items"][0]["content"] == "Run 2 ok." and runs["items"][0]["expires_at"]
    verdicts = mcp.call("memory.list", {"project_id": PROJECT, "event_type": "proposal_verdict"})["items"]
    assert verdicts[0]["metadata"] == {"verdict": "accepted"} and verdicts[0]["expires_at"] is None

    mcp.call("memory.save", {"project_id": PROJECT, "type": "constraint", "statement": "Mai toccare main."})
    memories = mcp.call("memory.list", {"project_id": PROJECT, "kind": "memory", "type": "constraint"})["items"]
    assert [m["statement"] for m in memories] == ["Mai toccare main."]
    assert mcp.fail("memory.list", {"project_id": PROJECT, "kind": "memory", "event_type": "x"})["code"] == \
        "invalid_argument"
    assert mcp.fail("memory.note", {"project_id": PROJECT, "content": "x", "ttl_days": -1})["code"] == \
        "invalid_argument"
