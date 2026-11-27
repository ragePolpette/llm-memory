"""MCP HTTP server for llm-memory (streamable HTTP + legacy SSE), backed by the v2 store."""

from __future__ import annotations

import hmac
import ipaddress
import json
from contextlib import asynccontextmanager
from typing import Any, TypedDict

import structlog
import uvicorn
from mcp.server import Server
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from datetime import datetime, timezone

from src.config import Config, get_config
from src.logging_setup import configure_logging
from src.mcp_server.mcp_transport import LegacySseTransport, StreamableTransport
from src.mcp_server.tools_v2 import register_tools_v2
from src.store_v2.runtime import Runtime, build_runtime

configure_logging()
logger = structlog.get_logger()

API_TOKEN_ENV = "LLM_MEMORY_API_TOKEN"
PROTECTED_MCP_PREFIXES = ("/mcp", "/sse", "/messages")
ADMIN_PREFIX = "/admin"

runtime: Runtime | None = None
mcp_server: Server | None = None
sse_transport: LegacySseTransport | None = None
streamable_session_manager: StreamableTransport | None = None
app_config = None


class InsecureBindError(RuntimeError):
    """Raised when the HTTP transport would be exposed without authentication."""


def is_loopback_host(host: str) -> bool:
    normalized = str(host).strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_bind(config: Config) -> None:
    if is_loopback_host(config.mcp_host) or config.api_token:
        return
    raise InsecureBindError(
        f"Refusing to bind the HTTP transport to non-loopback host '{config.mcp_host}' "
        f"without authentication. Set {API_TOKEN_ENV} to a strong random token, or bind "
        "MCP_MEMORY_HOST to 127.0.0.1."
    )


def _bearer_token(scope: Scope) -> str | None:
    for name, value in scope.get("headers") or []:
        if name.lower() == b"authorization":
            scheme, _, token = value.decode("latin-1").partition(" ")
            if scheme.lower() == "bearer" and token.strip():
                return token.strip()
            return None
    return None


def is_authorized(scope: Scope, config: Config | None) -> bool:
    expected = getattr(config, "api_token", None)
    if not expected:
        return False
    provided = _bearer_token(scope)
    if provided is None:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _path_matches(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def _auth_error(status_code: int, error_type: str, message: str) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
    return JSONResponse(
        {"status": "error", "error": {"type": error_type, "message": message}},
        status_code=status_code,
        headers=headers,
    )


class BearerAuthMiddleware:
    """Bearer token gate: optional for MCP endpoints, mandatory for /admin."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        response: Response | None = None
        if _path_matches(path, ADMIN_PREFIX):
            if app_config is None:
                response = Response("Server not initialized", status_code=503)
            elif not app_config.api_token:
                response = _auth_error(
                    403,
                    "forbidden",
                    f"Admin endpoints are disabled until {API_TOKEN_ENV} is configured.",
                )
            elif not is_authorized(scope, app_config):
                response = _auth_error(401, "unauthorized", "Missing or invalid bearer token.")
        elif any(_path_matches(path, prefix) for prefix in PROTECTED_MCP_PREFIXES):
            if app_config is not None and app_config.api_token and not is_authorized(scope, app_config):
                response = _auth_error(401, "unauthorized", "Missing or invalid bearer token.")

        if response is not None:
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


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


def _parse_limit(raw_value: str | None, *, default: int = 100, minimum: int = 1, maximum: int = 500) -> int:
    if raw_value is None or not str(raw_value).strip():
        return default
    limit = int(str(raw_value).strip())
    if limit < minimum or limit > maximum:
        raise ValueError(f"limit must be between {minimum} and {maximum}")
    return limit


def _parse_after_id(raw_value: str | None) -> int | None:
    if raw_value is None or not str(raw_value).strip():
        return None
    try:
        value = int(str(raw_value).strip())
    except ValueError:
        raise ValueError("after_id must be a non-negative integer") from None
    if value < 0:
        raise ValueError("after_id must be a non-negative integer")
    return value


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
    validate_bind(config)
    app_config = config
    runtime = build_runtime()

    mcp_server = Server("llm-memory")
    register_tools_v2(mcp_server, runtime.service)

    @mcp_server.list_resources()
    async def _list_resources():
        return []

    @mcp_server.list_resource_templates()
    async def _list_resource_templates():
        return []

    @mcp_server.list_prompts()
    async def _list_prompts():
        return []

    sse_transport = LegacySseTransport(mcp_server, messages_endpoint="/messages/")
    streamable_session_manager = StreamableTransport(
        mcp_server,
        json_response=not config.mcp_sse_enabled,
        allowed_hosts=config.mcp_allowed_hosts,
        allowed_origins=config.mcp_allowed_origins,
    )
    logger.info("llm-memory HTTP runtime initialized", transport=config.startup_diagnostics()["transport"],
                **runtime.settings.diagnostics())
    if runtime.migration_report:
        logger.info("v1 database migrated", **runtime.migration_report)


@asynccontextmanager
async def lifespan(app: Starlette):
    init_components()
    assert runtime is not None and streamable_session_manager is not None
    runtime.start_background()
    try:
        async with streamable_session_manager.run():
            yield
    finally:
        await runtime.stop_background()
        runtime.close()


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


class LegacySseASGIApp:
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if sse_transport is None:
            response = Response("Server not initialized", status_code=503)
            await response(scope, receive, send)
            return
        await sse_transport.handle_sse(scope, receive, send)


streamable_http_app = StreamableHTTPASGIApp()
legacy_messages_app = LegacyMessagesASGIApp()
sse_legacy_endpoint = LegacySseASGIApp()


async def legacy_sse_method_not_allowed(request: Request):
    return Response(
        "Method Not Allowed",
        status_code=405,
        headers={"Allow": "GET"},
    )


async def health(request: Request):
    expose_diagnostics = app_config is not None and (
        not app_config.api_token or is_authorized(request.scope, app_config)
    )
    return JSONResponse(
        {
            "status": "ok",
            "server": "llm-memory",
            "api": "v2",
            "mcp_sse_enabled": bool(getattr(app_config, "mcp_sse_enabled", False)),
            "diagnostics": _diagnostics() if expose_diagnostics else None,
        }
    )


def _diagnostics() -> dict[str, Any]:
    payload: dict[str, Any] = {"transport": app_config.startup_diagnostics()["transport"]} if app_config else {}
    if runtime is not None:
        payload.update(runtime.settings.diagnostics())
        version = runtime.store.active_embedding_version()
        payload["active_embedding_version"] = version["version_id"] if version else None
    return payload


def _not_ready() -> JSONResponse:
    return JSONResponse({"status": "error", "message": "Server not initialized"}, status_code=503)


def _normalize_since(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("since must be a valid ISO 8601 timestamp") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


async def admin_audit(request: Request):
    """Audit log, newest first; with after_id, the rows after it oldest first (a cursor)."""

    if runtime is None:
        return _not_ready()
    params = request.query_params
    try:
        limit = _parse_limit(params.get("limit"), default=100)
        filters = {
            "entry_id": params.get("entry_id"),
            "action": params.get("action"),
            "actor": params.get("actor"),
            "reason": params.get("reason"),
            "since": _normalize_since(params.get("since")),
            "after_id": _parse_after_id(params.get("after_id")),
        }
    except ValueError as exc:
        return _admin_bad_request(str(exc))
    items = []
    for record in runtime.store.list_audit(limit=limit, **filters):
        preview = json.dumps(record.payload, ensure_ascii=True, sort_keys=True)
        items.append({
            "id": record.id,
            "entry_id": record.entry_id,
            "action": record.action,
            "actor": record.actor,
            "reason": record.reason,
            "payload": record.payload,
            "created_at": record.created_at,
            "payload_preview": preview if len(preview) <= 240 else preview[:237] + "...",
        })
    return JSONResponse({
        "status": "ok",
        "server": "llm-memory",
        "api": "v2",
        "audit": {"count": len(items), "limit": limit, "filters": filters, "items": items},
    })


async def admin_projects(request: Request):
    if runtime is None:
        return _not_ready()
    try:
        limit = _parse_limit(request.query_params.get("limit"), default=200)
    except ValueError as exc:
        return _admin_bad_request(str(exc))
    items = runtime.store.list_projects()[:limit]
    for project in items:
        project.update(runtime.store.count_items(project["project_id"]))
    return JSONResponse({
        "status": "ok",
        "server": "llm-memory",
        "api": "v2",
        "projects": {"count": len(items), "items": items},
    })


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/admin/audit", admin_audit, methods=["GET"]),
    Route("/admin/projects", admin_projects, methods=["GET"]),
    Route("/mcp", endpoint=streamable_http_app, methods=["GET", "POST", "DELETE"]),
    Route("/sse", sse_legacy_endpoint, methods=["GET"]),
    Mount("/messages/", app=legacy_messages_app),
    Route("/sse", endpoint=legacy_sse_method_not_allowed, methods=["POST", "DELETE"]),
]

app = Starlette(routes=routes, lifespan=lifespan, middleware=[Middleware(BearerAuthMiddleware)])


def main() -> None:
    run_config = get_config()
    try:
        validate_bind(run_config)
    except InsecureBindError as exc:
        raise SystemExit(f"error: {exc}") from exc
    uvicorn.run(app, host=run_config.mcp_host, port=run_config.mcp_port, log_level="info")


if __name__ == "__main__":
    main()
