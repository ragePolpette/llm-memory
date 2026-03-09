"""HTTP server MCP per LLM Memory v2 (streamable + SSE legacy)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

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

logger = logging.getLogger(__name__)

runtime = None
mcp_server: Server | None = None
sse_transport: SseServerTransport | None = None
streamable_session_manager: StreamableHTTPSessionManager | None = None
app_config = None


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
        }
    )


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/mcp", endpoint=streamable_http_app, methods=["GET", "POST", "DELETE"]),
    Route("/sse", sse_legacy_endpoint, methods=["GET"]),
    Mount("/messages/", app=legacy_messages_app),
    Route("/sse", endpoint=legacy_sse_method_not_allowed, methods=["POST", "DELETE"]),
]

app = Starlette(routes=routes, lifespan=lifespan)


if __name__ == "__main__":
    run_config = get_config()
    uvicorn.run(app, host=run_config.mcp_host, port=run_config.mcp_port, log_level="info")
