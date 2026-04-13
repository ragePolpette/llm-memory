"""HTTP server MCP per LLM Memory v2 (streamable + SSE legacy)."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, TypedDict

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from src.bootstrap import build_runtime
from src.config import get_config
from src.mcp_server.tools import register_tools
from src.service.memory_service import ActorContext

logger = logging.getLogger(__name__)

runtime = None
mcp_server: Server | None = None
sse_transport: SseServerTransport | None = None
streamable_session_manager: StreamableHTTPSessionManager | None = None
app_config = None


class ValidationError(TypedDict):
    field: str
    error: str
    expected: str


# Strict top-level JSON-RPC schema expected by the MCP HTTP endpoint.
REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["jsonrpc", "method"],
    "properties": {
        # JSON-RPC protocol version discriminator. MCP requests must use 2.0.
        "jsonrpc": {
            "type": "string",
            "enum": ["2.0"],
            "description": "JSON-RPC protocol version. Must be '2.0'.",
        },
        # Request id used for request/response correlation. Optional for notifications.
        "id": {
            "type": ["string", "integer", "null"],
            "description": "Optional request identifier for JSON-RPC correlation.",
        },
        # MCP method name, for example initialize, tools/list, tools/call.
        "method": {
            "type": "string",
            "description": "JSON-RPC method name to execute.",
        },
        # Structured arguments passed to the MCP method implementation.
        "params": {
            "type": "object",
            "description": "Optional method parameters object.",
        },
    },
}


def _expected_label(definition: dict[str, Any]) -> str:
    if "enum" in definition:
        return " | ".join(str(item) for item in definition["enum"])

    expected_type = definition.get("type")
    if isinstance(expected_type, list):
        return " | ".join(str(item) for item in expected_type)
    if expected_type is None:
        return "any"
    return str(expected_type)


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return False


def validateRequest(payload: Any) -> dict[str, Any]:
    errors: list[ValidationError] = []

    if not isinstance(payload, dict):
        return {
            "valid": False,
            "errors": [
                {
                    "field": "$",
                    "error": "must be a JSON object",
                    "expected": REQUEST_SCHEMA["type"],
                }
            ],
        }

    properties = REQUEST_SCHEMA["properties"]
    required_fields = REQUEST_SCHEMA["required"]

    for field_name in required_fields:
        if field_name not in payload:
            field_schema = properties[field_name]
            errors.append(
                {
                    "field": field_name,
                    "error": "is required",
                    "expected": _expected_label(field_schema),
                }
            )

    for field_name in payload.keys():
        if field_name not in properties:
            errors.append(
                {
                    "field": field_name,
                    "error": "unexpected field",
                    "expected": "no additional fields",
                }
            )

    for field_name, field_schema in properties.items():
        if field_name not in payload:
            continue

        value = payload[field_name]
        expected_type = field_schema.get("type")
        type_candidates = expected_type if isinstance(expected_type, list) else [expected_type]

        if expected_type is not None and not any(_matches_type(value, str(candidate)) for candidate in type_candidates):
            errors.append(
                {
                    "field": field_name,
                    "error": f"type mismatch: got {type(value).__name__}",
                    "expected": _expected_label(field_schema),
                }
            )
            continue

        if "enum" in field_schema and value not in field_schema["enum"]:
            errors.append(
                {
                    "field": field_name,
                    "error": f"invalid enum value: {value!r}",
                    "expected": _expected_label(field_schema),
                }
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def _validation_error_response(errors: list[ValidationError]) -> JSONResponse:
    return JSONResponse(
        {
            "success": False,
            "error": {
                "type": "validation_error",
                "message": "Request validation failed",
                "details": errors,
            },
        },
        status_code=400,
    )


def _admin_bad_request(message: str) -> JSONResponse:
    return JSONResponse(
        {
            "status": "error",
            "error": {
                "type": "bad_request",
                "message": message,
            },
        },
        status_code=400,
    )


def _admin_forbidden(message: str) -> JSONResponse:
    return JSONResponse(
        {
            "status": "error",
            "error": {
                "type": "forbidden",
                "message": message,
            },
        },
        status_code=403,
    )


def _parse_limit(raw_value: str | None, *, default: int = 100, minimum: int = 1, maximum: int = 500) -> int:
    if raw_value is None or not str(raw_value).strip():
        return default
    limit = int(str(raw_value).strip())
    if limit < minimum or limit > maximum:
        raise ValueError(f"limit must be between {minimum} and {maximum}")
    return limit


def _parse_optional_bool(raw_value: str | None) -> bool | None:
    if raw_value is None or not str(raw_value).strip():
        return None
    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("resolved must be a boolean value")


async def _parse_json_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise ValueError("request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _normalize_payload_bool(value: Any, *, field_name: str, default: bool | None = None) -> bool:
    if value is None:
        if default is None:
            raise ValueError(f"{field_name} must be a boolean")
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean")


def _normalize_payload_int(
    value: Any,
    *,
    field_name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None:
        return default
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if normalized < minimum or normalized > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return normalized


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _require_body_reason(value: Any, *, field_name: str = "reason") -> str:
    normalized = _normalize_optional_string(value)
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _build_admin_actor(payload: dict[str, Any]) -> ActorContext:
    if runtime is None:
        raise RuntimeError("Server not initialized")
    agent_id = _normalize_optional_string(payload.get("agent_id"))
    if not agent_id:
        raise ValueError("agent_id is required")
    return ActorContext(
        agent_id=agent_id,
        user_id=_normalize_optional_string(payload.get("user_id")),
        workspace_id=_normalize_optional_string(payload.get("workspace_id")) or runtime.config.default_workspace_id,
        project_id=_normalize_optional_string(payload.get("project_id")) or runtime.config.default_project_id,
    )


async def _read_request_body(receive: Receive) -> tuple[bytes, list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = []
    body_chunks: list[bytes] = []

    while True:
        message = await receive()
        messages.append(message)

        if message.get("type") != "http.request":
            break

        body_chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break

    return b"".join(body_chunks), messages


def _replay_receive(messages: list[dict[str, Any]]) -> Receive:
    queue = list(messages)

    async def _receive() -> dict[str, Any]:
        if queue:
            return queue.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    return _receive


def init_components() -> None:
    global runtime, mcp_server, sse_transport, streamable_session_manager, app_config

    config = get_config()
    app_config = config
    runtime = build_runtime(config)

    mcp_server = Server("llm-memory")
    register_tools(mcp_server, runtime.service)

    @mcp_server.list_resources()
    async def _list_resources():
        return []

    @mcp_server.list_resource_templates()
    async def _list_resource_templates():
        return []

    @mcp_server.list_prompts()
    async def _list_prompts():
        return []

    sse_transport = SseServerTransport("/messages/")
    streamable_session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        json_response=not config.mcp_sse_enabled,
        stateless=False,
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=config.mcp_allowed_hosts,
            allowed_origins=config.mcp_allowed_origins,
        ),
    )
    logger.info("LLM Memory HTTP runtime initialized", **config.startup_diagnostics())


@asynccontextmanager
async def lifespan(app: Starlette):
    init_components()
    if runtime is not None:
        await runtime.prewarm()
    assert streamable_session_manager is not None
    async with streamable_session_manager.run():
        yield


class StreamableHTTPASGIApp:
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope.get("type") == "http"
            and scope.get("path") == "/mcp"
            and scope.get("method") == "GET"
            and app_config is not None
            and not app_config.mcp_sse_enabled
        ):
            response = Response(
                "Method Not Allowed",
                status_code=405,
                headers={"Allow": "POST, DELETE"},
            )
            await response(scope, receive, send)
            return

        if scope.get("type") == "http" and scope.get("path") == "/mcp" and scope.get("method") == "POST":
            body_bytes, messages = await _read_request_body(receive)
            if not body_bytes:
                response = _validation_error_response(
                    [{"field": "$", "error": "request body is required", "expected": "JSON object"}]
                )
                await response(scope, receive, send)
                return

            try:
                payload = json.loads(body_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                response = _validation_error_response(
                    [{"field": "$", "error": "invalid JSON payload", "expected": "JSON object"}]
                )
                await response(scope, receive, send)
                return

            validation = validateRequest(payload)
            if not validation["valid"]:
                response = _validation_error_response(validation["errors"])
                await response(scope, receive, send)
                return

            receive = _replay_receive(messages)

        if streamable_session_manager is None:
            response = Response("Server not initialized", status_code=503)
            await response(scope, receive, send)
            return
        await streamable_session_manager.handle_request(scope, receive, send)


class LegacyMessagesASGIApp:
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if sse_transport is None:
            response = Response("Server not initialized", status_code=503)
            await response(scope, receive, send)
            return
        await sse_transport.handle_post_message(scope, receive, send)


streamable_http_app = StreamableHTTPASGIApp()
legacy_messages_app = LegacyMessagesASGIApp()


async def sse_legacy_endpoint(request: Request):
    if sse_transport is None or mcp_server is None:
        return Response("Server not initialized", status_code=503)

    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
        read_stream, write_stream = streams
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )

    return Response()


async def legacy_sse_method_not_allowed(request: Request):
    return Response(
        "Method Not Allowed",
        status_code=405,
        headers={"Allow": "GET"},
    )


async def health(request: Request):
    return JSONResponse(
        {
            "status": "ok",
            "server": "llm-memory",
            "api": "v2",
            "mcp_sse_enabled": bool(getattr(app_config, "mcp_sse_enabled", False)),
            "diagnostics": app_config.startup_diagnostics() if app_config is not None else None,
        }
    )


async def admin_summary(request: Request):
    if runtime is None:
        return JSONResponse({"status": "error", "message": "Server not initialized"}, status_code=503)
    return JSONResponse(
        {
            "status": "ok",
            "server": "llm-memory",
            "api": "v2",
            "summary": runtime.service.admin_summary(),
        }
    )


async def admin_audit(request: Request):
    if runtime is None:
        return JSONResponse({"status": "error", "message": "Server not initialized"}, status_code=503)
    try:
        limit = _parse_limit(request.query_params.get("limit"), default=100)
        payload = runtime.service.admin_list_audit(
            limit=limit,
            entry_id=request.query_params.get("entry_id"),
            action=request.query_params.get("action"),
            actor=request.query_params.get("actor"),
            reason=request.query_params.get("reason"),
            since=request.query_params.get("since"),
        )
    except ValueError as exc:
        return _admin_bad_request(str(exc))
    return JSONResponse(
        {
            "status": "ok",
            "server": "llm-memory",
            "api": "v2",
            "audit": payload,
        }
    )


async def admin_projects(request: Request):
    if runtime is None:
        return JSONResponse({"status": "error", "message": "Server not initialized"}, status_code=503)
    try:
        limit = _parse_limit(request.query_params.get("limit"), default=200)
        payload = runtime.service.admin_list_projects(
            workspace_id=request.query_params.get("workspace_id"),
            limit=limit,
        )
    except ValueError as exc:
        return _admin_bad_request(str(exc))
    return JSONResponse(
        {
            "status": "ok",
            "server": "llm-memory",
            "api": "v2",
            "projects": payload,
        }
    )


async def admin_fast_memory(request: Request):
    if runtime is None:
        return JSONResponse({"status": "error", "message": "Server not initialized"}, status_code=503)
    try:
        limit = _parse_limit(request.query_params.get("limit"), default=100)
        payload = runtime.service.admin_list_fast(
            workspace_id=request.query_params.get("workspace_id"),
            project_id=request.query_params.get("project_id"),
            agent_id=request.query_params.get("agent_id"),
            event_type=request.query_params.get("event_type"),
            resolved=_parse_optional_bool(request.query_params.get("resolved")),
            distillation_status=request.query_params.get("distillation_status"),
            limit=limit,
        )
    except ValueError as exc:
        return _admin_bad_request(str(exc))
    return JSONResponse(
        {
            "status": "ok",
            "server": "llm-memory",
            "api": "v2",
            "fast_memory": payload,
        }
    )


async def admin_fast_memory_candidates(request: Request):
    if runtime is None:
        return JSONResponse({"status": "error", "message": "Server not initialized"}, status_code=503)
    try:
        limit = _parse_limit(request.query_params.get("limit"), default=20, maximum=200)
        payload = runtime.service.admin_rank_fast_candidates(
            workspace_id=request.query_params.get("workspace_id"),
            project_id=request.query_params.get("project_id"),
            limit=limit,
            include_resolved=bool(_parse_optional_bool(request.query_params.get("include_resolved")) or False),
            distillation_status=request.query_params.get("distillation_status"),
        )
    except ValueError as exc:
        return _admin_bad_request(str(exc))
    return JSONResponse(
        {
            "status": "ok",
            "server": "llm-memory",
            "api": "v2",
            "candidates": payload,
        }
    )


async def admin_prepare_fast_distillation(request: Request):
    if runtime is None:
        return JSONResponse({"status": "error", "message": "Server not initialized"}, status_code=503)
    try:
        payload = await _parse_json_body(request)
        actor = _build_admin_actor(payload)
        prepared = runtime.service.prepare_fast_distillation(
            actor=actor,
            reason=_require_body_reason(payload.get("reason")),
            cluster_id=_normalize_optional_string(payload.get("cluster_id")),
            entry_id=_normalize_optional_string(payload.get("entry_id")),
            top_k=_normalize_payload_int(
                payload.get("top_k"),
                field_name="top_k",
                default=1,
                minimum=1,
                maximum=10,
            ),
            include_resolved=_normalize_payload_bool(
                payload.get("include_resolved"),
                field_name="include_resolved",
                default=False,
            ),
            distillation_status=_normalize_optional_string(payload.get("distillation_status")),
        )
    except PermissionError as exc:
        return _admin_forbidden(str(exc))
    except ValueError as exc:
        return _admin_bad_request(str(exc))
    return JSONResponse(
        {
            "status": "ok",
            "server": "llm-memory",
            "api": "v2",
            "distillation_prepare": prepared,
        }
    )


async def admin_apply_fast_distillation(request: Request):
    if runtime is None:
        return JSONResponse({"status": "error", "message": "Server not initialized"}, status_code=503)
    try:
        body = await _parse_json_body(request)
        actor = _build_admin_actor(body)
        payload = body.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object with a decisions array")
        result = await runtime.service.apply_fast_distillation(
            actor=actor,
            payload=payload,
            reason=_require_body_reason(body.get("reason")),
            dry_run=_normalize_payload_bool(body.get("dry_run"), field_name="dry_run", default=True),
        )
    except PermissionError as exc:
        return _admin_forbidden(str(exc))
    except ValueError as exc:
        return _admin_bad_request(str(exc))
    return JSONResponse(
        {
            "status": "ok",
            "server": "llm-memory",
            "api": "v2",
            "distillation_apply": result,
        }
    )


async def admin_fast_memory_entry(request: Request):
    if runtime is None:
        return JSONResponse({"status": "error", "message": "Server not initialized"}, status_code=503)
    entry_id = str(request.path_params.get("entry_id", "")).strip()
    if not entry_id:
        return _admin_bad_request("entry_id is required")
    payload = runtime.service.admin_get_fast(entry_id)
    if payload is None:
        return JSONResponse(
            {
                "status": "error",
                "error": {
                    "type": "not_found",
                    "message": f"Fast-memory entry '{entry_id}' was not found",
                },
            },
            status_code=404,
        )
    return JSONResponse(
        {
            "status": "ok",
            "server": "llm-memory",
            "api": "v2",
            "entry": payload,
        }
    )


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/admin/summary", admin_summary, methods=["GET"]),
    Route("/admin/audit", admin_audit, methods=["GET"]),
    Route("/admin/projects", admin_projects, methods=["GET"]),
    Route("/admin/fast-memory", admin_fast_memory, methods=["GET"]),
    Route("/admin/fast-memory/candidates", admin_fast_memory_candidates, methods=["GET"]),
    Route("/admin/fast-memory/distillation/prepare", admin_prepare_fast_distillation, methods=["POST"]),
    Route("/admin/fast-memory/distillation/apply", admin_apply_fast_distillation, methods=["POST"]),
    Route("/admin/fast-memory/{entry_id:str}", admin_fast_memory_entry, methods=["GET"]),
    Route("/mcp", endpoint=streamable_http_app, methods=["GET", "POST", "DELETE"]),
    Route("/sse", sse_legacy_endpoint, methods=["GET"]),
    Mount("/messages/", app=legacy_messages_app),
    Route("/sse", endpoint=legacy_sse_method_not_allowed, methods=["POST", "DELETE"]),
]

app = Starlette(routes=routes, lifespan=lifespan)


if __name__ == "__main__":
    run_config = get_config()
    uvicorn.run(app, host=run_config.mcp_host, port=run_config.mcp_port, log_level="info")
