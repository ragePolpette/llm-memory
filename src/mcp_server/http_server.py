"""HTTP server MCP per LLM Memory v2 (streamable + SSE legacy)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
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


def init_components() -> None:
    global runtime, mcp_server, sse_transport, streamable_session_manager

    config = get_config()
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
        json_response=True,
        stateless=False,
    )


@asynccontextmanager
async def lifespan(app: Starlette):
    init_components()
    assert streamable_session_manager is not None
    async with streamable_session_manager.run():
        yield


class StreamableHTTPASGIApp:
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
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


async def health(request: Request):
    return JSONResponse({"status": "ok", "server": "llm-memory", "api": "v2"})


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/mcp", endpoint=streamable_http_app, methods=["GET", "POST", "DELETE"]),
    Route("/sse", sse_legacy_endpoint, methods=["GET"]),
    Mount("/messages/", app=legacy_messages_app),
    Route("/sse", endpoint=streamable_http_app, methods=["POST", "DELETE"]),
]

app = Starlette(routes=routes, lifespan=lifespan)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8767, log_level="info")
