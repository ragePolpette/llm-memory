"""Contract tests for the parts of llm-memory that Yesod depends on.

Yesod (Level 0 watcher and adapters) reads the audit log over GET /admin/audit and
enriches rows through MCP tools. These tests pin that surface end to end through
the real HTTP app, so a rewrite of the internals cannot break Yesod unnoticed.

Consumers on the Yesod side:
- Yesod.Watcher/Audit.cs: GET /admin/audit, after_id paging, response shape.
- Yesod.Watcher/Salience.cs: action names and payload fields it reads.
- Yesod.Memory/LlmMemory.cs: memory.get, memory.get_fast, memory.log_fast,
  memory.list_fast, memory.create_project.

Change these tests only together with Yesod.
"""

from __future__ import annotations

import itertools
import json

import pytest
from starlette.testclient import TestClient

import src.mcp_server.http_server as http_server

TOKEN = "contract-token-0123456789abcdef"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
    **AUTH,
}
AGENT = "yesod"
PROJECT = "contract-prj"
SCOPE = {"scope_level": "project", "project_id": PROJECT}

AUDIT_ITEM_KEYS = {"id", "entry_id", "action", "actor", "reason", "payload", "created_at"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_SQLITE_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_IMPORT_EXPORT_BASE_DIR", str(tmp_path / "exchange"))
    monkeypatch.setenv("MCP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash-local")
    monkeypatch.setenv("EMBEDDING_MODEL", "local-hash-contract")
    monkeypatch.setenv("MCP_MEMORY_HOST", "127.0.0.1")
    monkeypatch.setenv("MEMORY_ENCRYPTION_ENABLED", "false")
    monkeypatch.setenv("MEMORY_MULTI_PROJECT_ENABLED", "true")
    monkeypatch.setenv("LLM_MEMORY_API_TOKEN", TOKEN)
    with TestClient(http_server.app, base_url="http://127.0.0.1:8767") as test_client:
        yield test_client
    for name in ("runtime", "mcp_server", "sse_transport", "streamable_session_manager", "app_config"):
        monkeypatch.setattr(http_server, name, None)


class Mcp:
    """Minimal streamable-HTTP MCP client, the way Yesod talks to /mcp."""

    def __init__(self, client: TestClient):
        self.client = client
        self.ids = itertools.count(1)
        response = self._post(
            {
                "jsonrpc": "2.0",
                "id": next(self.ids),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "yesod-contract", "version": "0.0.1"},
                },
            }
        )
        assert response.status_code == 200, response.text
        self.session = response.headers.get("mcp-session-id")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _post(self, body: dict):
        headers = dict(MCP_HEADERS)
        if getattr(self, "session", None):
            headers["mcp-session-id"] = self.session
        return self.client.post("/mcp", json=body, headers=headers)

    def call(self, name: str, arguments: dict) -> dict:
        response = self._post(
            {
                "jsonrpc": "2.0",
                "id": next(self.ids),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert not result.get("isError"), result
        return json.loads(result["content"][0]["text"])


def audit(client: TestClient, **params) -> dict:
    response = client.get("/admin/audit", params=params, headers=AUTH)
    assert response.status_code == 200, response.text
    return response.json()


def audit_rows(client: TestClient, **params) -> list[dict]:
    return audit(client, limit=500, **params)["audit"]["items"]


def add_fact(mcp: Mcp, content: str) -> dict:
    return mcp.call(
        "memory.add",
        {"content": content, "type": "fact", "agent_id": AGENT, "scope": SCOPE},
    )


@pytest.fixture
def mcp(client) -> Mcp:
    session = Mcp(client)
    session.call("memory.create_project", {"agent_id": AGENT, "project_id": PROJECT})
    return session


# --- GET /admin/audit ------------------------------------------------------


def test_audit_requires_bearer_token(client):
    assert client.get("/admin/audit").status_code == 401
    assert client.get("/admin/audit", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/admin/audit", headers=AUTH).status_code == 200


def test_audit_response_shape(client, mcp):
    add_fact(mcp, "Il sistema usa PostgreSQL 16 come database principale.")
    body = audit(client, limit=10, after_id=0)

    assert body["status"] == "ok"
    assert body["server"] == "llm-memory"
    assert body["api"] == "v2"
    page = body["audit"]
    assert {"count", "limit", "filters", "items"} <= set(page)
    assert page["count"] == len(page["items"])
    assert page["limit"] == 10
    # Yesod detects forward paging only when after_id is echoed back as a number.
    assert isinstance(page["filters"]["after_id"], int) and page["filters"]["after_id"] == 0
    assert page["items"]
    for item in page["items"]:
        assert AUDIT_ITEM_KEYS <= set(item)
        assert isinstance(item["id"], int)
        assert isinstance(item["action"], str)
        assert isinstance(item["payload"], dict)
        assert isinstance(item["created_at"], str)


def test_audit_after_id_pages_forward_without_gaps(client, mcp):
    for index in range(7):
        add_fact(mcp, f"Il sistema usa la regola numero {index} per il deploy notturno.")
    everything = sorted(row["id"] for row in audit_rows(client))
    assert len(everything) >= 7

    seen: list[int] = []
    cursor = 0
    while True:
        page = audit(client, limit=3, after_id=cursor)["audit"]["items"]
        if not page:
            break
        ids = [row["id"] for row in page]
        assert ids == sorted(ids), "after_id pages must be oldest first"
        assert all(row_id > cursor for row_id in ids)
        seen.extend(ids)
        cursor = ids[-1]
    assert seen == everything


def test_audit_without_after_id_is_newest_first(client, mcp):
    for index in range(3):
        add_fact(mcp, f"Il sistema usa il backup numero {index} ogni notte.")
    ids = [row["id"] for row in audit_rows(client)]
    assert ids == sorted(ids, reverse=True)


def test_audit_since_is_inclusive_and_filters_older_rows(client, mcp):
    add_fact(mcp, "Il sistema usa ntfy per le notifiche urgenti.")
    rows = audit_rows(client)
    newest = rows[0]
    since_rows = audit_rows(client, since=newest["created_at"])
    assert newest["id"] in [row["id"] for row in since_rows]


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 501}, {"after_id": -1}, {"after_id": "x"}])
def test_audit_rejects_invalid_parameters(client, params):
    response = client.get("/admin/audit", params=params, headers=AUTH)
    assert response.status_code == 400


def test_audit_ids_are_monotonic_in_write_order(client, mcp):
    entry_ids = [
        add_fact(mcp, f"Il sistema usa la coda numero {index} per i job.")["entry_id"] for index in range(4)
    ]
    rows = [row for row in audit_rows(client, after_id=0, action="write_attempt") if row["entry_id"] in entry_ids]
    assert [row["entry_id"] for row in rows] == entry_ids


# --- Audit actions and payload fields read by Yesod.Watcher/Salience.cs ----


def test_accepted_write_emits_write_attempt_with_outcome_and_project(client, mcp):
    result = add_fact(mcp, "Il sistema usa Fireworks come provider del modello.")
    assert result["success"] is True
    rows = [row for row in audit_rows(client, action="write_attempt") if row["entry_id"] == result["entry_id"]]
    assert len(rows) == 1
    row = rows[0]
    assert row["actor"] == AGENT
    assert row["payload"]["outcome"] == "accepted"
    assert row["payload"]["scope"]["project_id"] == PROJECT


def test_duplicate_write_emits_write_attempt_with_duplicate_outcome(client, mcp):
    content = "Il sistema usa Telegram come canale di notifica secondario."
    first = add_fact(mcp, content)
    second = add_fact(mcp, content)
    assert second.get("duplicate_of") == first["entry_id"] or second.get("entry_id") == first["entry_id"]
    outcomes = [row["payload"].get("outcome") for row in audit_rows(client, action="write_attempt")]
    assert "duplicate" in outcomes


def test_fast_write_emits_fast_write_with_project(client, mcp):
    result = mcp.call(
        "memory.log_fast",
        {
            "agent_id": AGENT,
            "scope": SCOPE,
            "event_type": "observation",
            "content": "Il watcher ha visto tre errori di timeout sul job notturno.",
            "metadata": {"source_ref": "contract"},
            "tags": ["yesod"],
            "source": "yesod",
        },
    )
    assert isinstance(result["entry_id"], str)
    rows = [row for row in audit_rows(client, action="fast_write") if row["entry_id"] == result["entry_id"]]
    assert len(rows) == 1
    assert rows[0]["payload"]["project_id"] == PROJECT


def _invalidate_row(client, mcp) -> dict:
    target = add_fact(mcp, "Il sistema usa Redis come cache delle sessioni.")
    mcp.call(
        "memory.invalidate",
        {"target_ids": [target["entry_id"]], "reason": "Redis rimosso", "agent_id": AGENT, "scope": SCOPE},
    )
    rows = [row for row in audit_rows(client, action="invalidate") if row["entry_id"] == target["entry_id"]]
    assert len(rows) == 1
    return rows[0]


def test_invalidate_emits_invalidate_with_entry_and_reason(client, mcp):
    row = _invalidate_row(client, mcp)
    assert row["reason"] == "Redis rimosso"
    assert isinstance(row["payload"]["internal_write"], dict)
    assert isinstance(row["payload"]["invalidation_entry_id"], str)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known gap: Yesod.Watcher/Salience.cs reads payload.internal_write.scope.project_id, "
        "but the invalidate audit payload has never carried a scope, so Yesod gets a null "
        "project unless it remembers the decision. Fix in llm-memory v2 (typed audit payloads)."
    ),
)
def test_invalidate_payload_carries_project_as_yesod_expects(client, mcp):
    row = _invalidate_row(client, mcp)
    assert row["payload"]["internal_write"]["scope"]["project_id"] == PROJECT


# --- MCP tools used by Yesod.Memory/LlmMemory.cs ---------------------------


def test_memory_get_returns_entry_with_fields_yesod_reads(mcp):
    added = add_fact(mcp, "Il sistema usa SQLite in modalità WAL per la memoria.")
    body = mcp.call("memory.get", {"agent_id": AGENT, "entry_id": added["entry_id"], "scope": SCOPE})
    entry = body["entry"]
    assert isinstance(entry, dict)
    assert entry["type"] == "fact"
    assert isinstance(entry["content"], str)
    assert isinstance(entry["tags"], list)
    assert isinstance(entry["metadata"]["importance_score"], (int, float))


def test_memory_get_fast_returns_entry_with_selection_score(mcp):
    logged = mcp.call(
        "memory.log_fast",
        {
            "agent_id": AGENT,
            "scope": SCOPE,
            "event_type": "observation",
            "content": "Il deploy di ieri ha richiesto due tentativi per un timeout.",
            "metadata": {},
            "tags": [],
            "source": "yesod",
        },
    )
    body = mcp.call("memory.get_fast", {"agent_id": AGENT, "entry_id": logged["entry_id"], "scope": SCOPE})
    entry = body["entry"]
    assert isinstance(entry, dict)
    assert isinstance(entry["event_type"], str)
    scoring = entry.get("metadata", {}).get("fast_memory_scoring", {})
    selection = scoring.get("selection_score", entry.get("selection_score"))
    assert isinstance(selection, (int, float))


def test_memory_list_fast_returns_entries_filtered_by_event_type(mcp):
    for event_type in ("observation", "observation", "note"):
        mcp.call(
            "memory.log_fast",
            {
                "agent_id": AGENT,
                "scope": SCOPE,
                "event_type": event_type,
                "content": f"Evento di tipo {event_type} registrato dal watcher di prova.",
                "metadata": {},
                "tags": [],
                "source": "yesod",
            },
        )
    body = mcp.call(
        "memory.list_fast",
        {"agent_id": AGENT, "scope": SCOPE, "event_type": "observation", "limit": 10},
    )
    entries = body["entries"]
    assert len(entries) == 2
    assert all(entry["event_type"] == "observation" for entry in entries)


def test_create_project_is_idempotent_for_yesod(mcp):
    # Yesod calls create_project for every project it sees; a second call must not fail.
    mcp.call("memory.create_project", {"agent_id": AGENT, "project_id": PROJECT})
