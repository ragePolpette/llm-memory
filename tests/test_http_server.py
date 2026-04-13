from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import src.mcp_server.http_server as http_server
from src.mcp_server.http_server import (
    REQUEST_SCHEMA,
    legacy_sse_method_not_allowed,
    routes,
    sse_legacy_endpoint,
    streamable_http_app,
    validateRequest,
)
from src.models import AuditEvent
from src.service.memory_service import ActorContext


@pytest.mark.asyncio
async def test_sse_write_methods_return_explicit_405():
    response = await legacy_sse_method_not_allowed(None)  # type: ignore[arg-type]

    assert response.status_code == 405
    assert response.headers["Allow"] == "GET"


def test_sse_routes_are_explicit_and_not_streamable_http():
    sse_routes = [route for route in routes if getattr(route, "path", None) == "/sse"]

    assert len(sse_routes) == 2
    assert any(route.endpoint is sse_legacy_endpoint and "GET" in route.methods for route in sse_routes)
    assert any(
        route.endpoint is legacy_sse_method_not_allowed and route.methods == {"POST", "DELETE"}
        for route in sse_routes
    )
    assert all(route.endpoint is not streamable_http_app for route in sse_routes)


def test_request_schema_is_strict():
    assert REQUEST_SCHEMA["type"] == "object"
    assert REQUEST_SCHEMA["additionalProperties"] is False
    assert REQUEST_SCHEMA["required"] == ["jsonrpc", "method"]


def test_validate_request_reports_missing_enum_type_and_extra_fields():
    result = validateRequest(
        {
            "jsonrpc": 2,
            "method": 42,
            "extra": True,
        }
    )

    assert result["valid"] is False
    assert {"field": "jsonrpc", "error": "type mismatch: got int", "expected": "2.0"} in result["errors"]
    assert {"field": "method", "error": "type mismatch: got int", "expected": "string"} in result["errors"]
    assert {"field": "extra", "error": "unexpected field", "expected": "no additional fields"} in result["errors"]


def test_validate_request_reports_invalid_enum_value():
    result = validateRequest({"jsonrpc": "1.0", "method": "initialize"})

    assert result["valid"] is False
    assert result["errors"] == [
        {
            "field": "jsonrpc",
            "error": "invalid enum value: '1.0'",
            "expected": "2.0",
        }
    ]


@pytest.mark.asyncio
async def test_streamable_http_returns_structured_validation_error(monkeypatch):
    http_server.app_config = SimpleNamespace(mcp_sse_enabled=False)
    http_server.streamable_session_manager = None

    body = json.dumps({"jsonrpc": "2.0", "params": {}, "unexpected": True}).encode("utf-8")
    scope = {
        "type": "http",
        "path": "/mcp",
        "method": "POST",
        "headers": [],
        "query_string": b"",
        "http_version": "1.1",
        "scheme": "http",
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8767),
    }
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent: list[dict] = []

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await streamable_http_app(scope, receive, send)

    start = next(message for message in sent if message["type"] == "http.response.start")
    body_message = next(message for message in sent if message["type"] == "http.response.body")
    payload = json.loads(body_message["body"].decode("utf-8"))

    assert start["status"] == 400
    assert payload == {
        "success": False,
        "error": {
            "type": "validation_error",
            "message": "Request validation failed",
            "details": [
                {"field": "method", "error": "is required", "expected": "string"},
                {"field": "unexpected", "error": "unexpected field", "expected": "no additional fields"},
            ],
        },
    }


@pytest.mark.asyncio
async def test_admin_summary_exposes_local_runtime_state(monkeypatch, service):
    actor = ActorContext(agent_id="agent-admin", user_id="user-admin", workspace_id="ws-test", project_id="prj-test")
    service.create_project(actor=actor, project_id="project-alpha", display_name="Project Alpha")
    await service.add(
        {
            "content": "Il progetto mantiene audit trail locali per le operazioni di memoria.",
            "context": "admin-summary",
            "agent_id": actor.agent_id,
            "visibility": "shared",
        },
        actor,
    )
    service.log_fast(
        {
            "content": "Fix rapido applicato al parser di import.",
            "agent_id": actor.agent_id,
            "event_type": "fix_attempt",
        },
        actor,
    )

    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service))

    response = await http_server.admin_summary(None)  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["summary"]["counts"]["projects_total"] >= 2
    assert payload["summary"]["counts"]["active_entries"] >= 1
    assert payload["summary"]["counts"]["fast_entries_total"] >= 1
    assert payload["summary"]["counts"]["audit_events_total"] >= 1
    assert payload["summary"]["fast_memory"]["pending"] >= 1


@pytest.mark.asyncio
async def test_admin_audit_supports_filters(monkeypatch, service):
    service.store.add_audit(
        AuditEvent(
            action="export",
            actor="agent-a",
            reason="manual_export",
            payload={"path": "one.jsonl"},
        )
    )
    service.store.add_audit(
        AuditEvent(
            action="import",
            actor="agent-b",
            reason="manual_import",
            payload={"path": "two.jsonl"},
        )
    )

    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service))
    request = SimpleNamespace(query_params={"action": "import", "actor": "agent-b", "limit": "10"})

    response = await http_server.admin_audit(request)  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["audit"]["count"] == 1
    assert payload["audit"]["items"][0]["action"] == "import"
    assert payload["audit"]["items"][0]["actor"] == "agent-b"
    assert "payload_preview" in payload["audit"]["items"][0]


@pytest.mark.asyncio
async def test_admin_audit_rejects_invalid_limit(monkeypatch, service):
    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service))
    request = SimpleNamespace(query_params={"limit": "0"})

    response = await http_server.admin_audit(request)  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert payload["error"]["type"] == "bad_request"
    assert "limit must be between" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_admin_projects_returns_entry_counts(monkeypatch, service):
    actor = ActorContext(agent_id="agent-admin", user_id="user-admin", workspace_id="ws-test", project_id="project-alpha")
    service.create_project(actor=actor, project_id="project-alpha", display_name="Project Alpha")
    await service.add(
        {
            "content": "Project Alpha usa endpoint admin locali per audit e summary.",
            "context": "admin-projects",
            "agent_id": actor.agent_id,
            "visibility": "shared",
        },
        actor,
    )
    service.log_fast(
        {
            "content": "Errore episodico in fase di export locale.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
        },
        actor,
    )

    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service))
    request = SimpleNamespace(query_params={"workspace_id": "ws-test", "limit": "20"})

    response = await http_server.admin_projects(request)  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    project = next(item for item in payload["projects"]["items"] if item["project_id"] == "project-alpha")
    assert project["entry_count"] >= 1
    assert project["active_entry_count"] >= 1
    assert project["fast_entry_count"] >= 1
    assert project["open_fast_entry_count"] >= 1


@pytest.mark.asyncio
async def test_admin_fast_memory_supports_filters(monkeypatch, service):
    actor = ActorContext(agent_id="agent-admin", user_id="user-admin", workspace_id="ws-test", project_id="project-alpha")
    service.create_project(actor=actor, project_id="project-alpha", display_name="Project Alpha")
    service.log_fast(
        {
            "content": "Retry rumoroso sul parser.",
            "agent_id": actor.agent_id,
            "event_type": "retry",
            "resolved": True,
        },
        actor,
    )
    second = service.log_fast(
        {
            "content": "Incidente reale sull'import incrementale.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "session_id": "session-fast-admin",
        },
        actor,
    )

    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service))
    request = SimpleNamespace(query_params={"project_id": "project-alpha", "event_type": "incident", "resolved": "false"})

    response = await http_server.admin_fast_memory(request)  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["fast_memory"]["count"] == 1
    assert payload["fast_memory"]["filters"]["resolved"] is False
    assert payload["fast_memory"]["items"][0]["id"] == second["entry_id"]
    assert payload["fast_memory"]["items"][0]["content_preview"].startswith("Incidente reale")


@pytest.mark.asyncio
async def test_admin_fast_memory_entry_returns_detail(monkeypatch, service):
    actor = ActorContext(agent_id="agent-admin", user_id="user-admin", workspace_id="ws-test", project_id="project-alpha")
    service.create_project(actor=actor, project_id="project-alpha", display_name="Project Alpha")
    logged = service.log_fast(
        {
            "content": "Tentativo di fix temporaneo sul deduplicatore.",
            "context": "cluster duplicate warnings",
            "agent_id": actor.agent_id,
            "event_type": "fix_attempt",
        },
        actor,
    )

    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service))
    request = SimpleNamespace(path_params={"entry_id": logged["entry_id"]})

    response = await http_server.admin_fast_memory_entry(request)  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["entry"]["id"] == logged["entry_id"]
    assert payload["entry"]["event_type"] == "fix_attempt"
    assert "content_preview" in payload["entry"]


@pytest.mark.asyncio
async def test_admin_fast_memory_rejects_invalid_resolved_filter(monkeypatch, service):
    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service))
    request = SimpleNamespace(query_params={"resolved": "maybe"})

    response = await http_server.admin_fast_memory(request)  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert payload["error"]["type"] == "bad_request"
    assert "resolved must be a boolean value" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_admin_fast_memory_candidates_clusters_recurring_patterns(monkeypatch, service):
    actor = ActorContext(agent_id="agent-admin", user_id="user-admin", workspace_id="ws-test", project_id="project-alpha")
    service.create_project(actor=actor, project_id="project-alpha", display_name="Project Alpha")
    service.log_fast(
        {
            "content": "L'utente y vedeva solo il menu x nel portale.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "session_id": "session-a",
            "kind": "bug",
            "product_area": "authorization",
            "component": "menu-engine",
            "entity_refs": ["user:y", "menu:x"],
            "recurrence_count": 2,
        },
        actor,
    )
    service.log_fast(
        {
            "content": "L'utente z vedeva solo il menu x nel portale.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "session_id": "session-b",
            "kind": "bug",
            "product_area": "authorization",
            "component": "menu-engine",
            "entity_refs": ["user:z", "menu:x"],
            "recurrence_count": 2,
        },
        actor,
    )

    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service))
    request = SimpleNamespace(query_params={"project_id": "project-alpha", "limit": "10"})

    response = await http_server.admin_fast_memory_candidates(request)  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["candidates"]["count"] >= 1
    first = payload["candidates"]["items"][0]
    assert first["member_count"] >= 2
    assert first["distinct_session_count"] >= 2
    assert first["component"] == "menu-engine"
    assert "cross_session_signal" in first["reasons"]


@pytest.mark.asyncio
async def test_admin_fast_memory_candidates_support_include_resolved(monkeypatch, service):
    actor = ActorContext(agent_id="agent-admin", user_id="user-admin", workspace_id="ws-test", project_id="project-alpha")
    service.create_project(actor=actor, project_id="project-alpha", display_name="Project Alpha")
    service.log_fast(
        {
            "content": "Fix già chiuso sul parser markdown.",
            "agent_id": actor.agent_id,
            "event_type": "fix_attempt",
            "resolved": True,
            "component": "markdown-parser",
        },
        actor,
    )

    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service))
    request = SimpleNamespace(query_params={"project_id": "project-alpha", "include_resolved": "true"})

    response = await http_server.admin_fast_memory_candidates(request)  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["candidates"]["filters"]["include_resolved"] is True
    assert payload["candidates"]["source_count"] >= 1


@pytest.mark.asyncio
async def test_admin_fast_memory_candidates_merges_semantic_variants(monkeypatch, service):
    actor = ActorContext(agent_id="agent-admin", user_id="user-admin", workspace_id="ws-test", project_id="project-alpha")
    service.create_project(actor=actor, project_id="project-alpha", display_name="Project Alpha")
    service.log_fast(
        {
            "content": "Il profilo y nel portale mostrava solo il menu x.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "session_id": "session-a",
            "kind": "bug",
            "product_area": "authorization",
            "component": "menu-engine",
            "entity_refs": ["user:y", "menu:x"],
            "recurrence_count": 2,
        },
        actor,
    )
    service.log_fast(
        {
            "content": "Nel portale il profilo z esponeva esclusivamente il menu x.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "session_id": "session-b",
            "kind": "bug",
            "product_area": "authorization",
            "component": "menu-engine",
            "entity_refs": ["user:z", "menu:x"],
            "recurrence_count": 2,
        },
        actor,
    )

    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service))
    request = SimpleNamespace(query_params={"project_id": "project-alpha", "limit": "10"})

    response = await http_server.admin_fast_memory_candidates(request)  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["candidates"]["count"] >= 1
    first = payload["candidates"]["items"][0]
    assert first["member_count"] >= 2
    assert first["base_cluster_count"] >= 2
    assert first["merge_version"] == "fast-cluster-v2"
    assert "semantic_cluster_merge" in first["reasons"]


@pytest.mark.asyncio
async def test_admin_fast_memory_candidates_do_not_merge_weakly_related_scope_only_entries(monkeypatch, service):
    actor = ActorContext(agent_id="agent-admin", user_id="user-admin", workspace_id="ws-test", project_id="project-alpha")
    service.create_project(actor=actor, project_id="project-alpha", display_name="Project Alpha")
    service.log_fast(
        {
            "content": "Timeout sull'import batch del catalogo fornitori.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "kind": "incident",
            "product_area": "operations",
            "component": "job-runner",
        },
        actor,
    )
    service.log_fast(
        {
            "content": "Permessi errati nella sincronizzazione dei ruoli backoffice.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "kind": "incident",
            "product_area": "operations",
            "component": "job-runner",
        },
        actor,
    )

    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service))
    request = SimpleNamespace(query_params={"project_id": "project-alpha", "limit": "10"})

    response = await http_server.admin_fast_memory_candidates(request)  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["candidates"]["count"] >= 2
    assert all(item["base_cluster_count"] == 1 for item in payload["candidates"]["items"][:2])


@pytest.mark.asyncio
async def test_admin_prepare_fast_distillation_returns_candidate_pack(monkeypatch, service):
    service.config.fast_memory_agent_distillation_enabled = True
    actor = ActorContext(agent_id="agent-admin", user_id="user-admin", workspace_id="ws-test", project_id="project-alpha")
    service.create_project(actor=actor, project_id="project-alpha", display_name="Project Alpha")
    service.log_fast(
        {
            "content": "L'utente y vedeva solo il menu x nel portale.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "kind": "bug",
            "product_area": "authorization",
            "component": "menu-engine",
            "entity_refs": ["user:y", "menu:x"],
            "recurrence_count": 2,
        },
        actor,
    )

    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service, config=service.config))

    class _Request:
        async def json(self):
            return {
                "agent_id": actor.agent_id,
                "user_id": actor.user_id,
                "workspace_id": actor.workspace_id,
                "project_id": actor.project_id,
                "reason": "prepare from dashboard",
                "top_k": 1,
            }

    response = await http_server.admin_prepare_fast_distillation(_Request())  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["distillation_prepare"]["prepared_count"] >= 1
    assert payload["distillation_prepare"]["candidates"][0]["component"] == "menu-engine"


@pytest.mark.asyncio
async def test_admin_prepare_fast_distillation_requires_feature_flag(monkeypatch, service):
    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service, config=service.config))

    class _Request:
        async def json(self):
            return {
                "agent_id": "agent-admin",
                "workspace_id": "ws-test",
                "project_id": "project-alpha",
                "reason": "prepare from dashboard",
            }

    response = await http_server.admin_prepare_fast_distillation(_Request())  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 403
    assert payload["error"]["type"] == "forbidden"
    assert "FAST_MEMORY_AGENT_DISTILLATION_ENABLED" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_admin_apply_fast_distillation_supports_dry_run(monkeypatch, service):
    service.config.fast_memory_agent_distillation_apply_enabled = True
    actor = ActorContext(agent_id="agent-admin", user_id="user-admin", workspace_id="ws-test", project_id="project-alpha")
    service.create_project(actor=actor, project_id="project-alpha", display_name="Project Alpha")
    first = service.log_fast(
        {
            "content": "L'utente y vedeva solo il menu x nel portale.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "component": "menu-engine",
        },
        actor,
    )
    second = service.log_fast(
        {
            "content": "L'utente z vedeva solo il menu x nel portale.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "component": "menu-engine",
        },
        actor,
    )

    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service, config=service.config))

    class _Request:
        async def json(self):
            return {
                "agent_id": actor.agent_id,
                "user_id": actor.user_id,
                "workspace_id": actor.workspace_id,
                "project_id": actor.project_id,
                "reason": "preview apply from dashboard",
                "payload": {
                    "decisions": [
                        {
                            "cluster_id": "cluster-menu",
                            "action": "promote",
                            "summary": "La gestione menu dipende dai permessi utente consolidati.",
                            "confidence": 0.9,
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
            }

    response = await http_server.admin_apply_fast_distillation(_Request())  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["distillation_apply"]["success"] is True
    assert payload["distillation_apply"]["dry_run"] is True


@pytest.mark.asyncio
async def test_admin_apply_fast_distillation_rejects_invalid_contract(monkeypatch, service):
    service.config.fast_memory_agent_distillation_apply_enabled = True
    monkeypatch.setattr(http_server, "runtime", SimpleNamespace(service=service, config=service.config))

    class _Request:
        async def json(self):
            return {
                "agent_id": "agent-admin",
                "workspace_id": "ws-test",
                "project_id": "project-alpha",
                "reason": "invalid apply from dashboard",
                "payload": {"decisions": []},
            }

    response = await http_server.admin_apply_fast_distillation(_Request())  # type: ignore[arg-type]
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert payload["error"]["type"] == "bad_request"
    assert "payload.decisions must contain at least one decision" in payload["error"]["message"]
