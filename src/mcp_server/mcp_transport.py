"""Adapter around the `mcp` SDK HTTP transports.

Every use of `mcp` transport internals lives here, so an SDK upgrade only
needs to touch this module.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import Receive, Scope, Send


class StreamableTransport:
    """Streamable HTTP session manager bound to one MCP server."""

    def __init__(
        self,
        server: Server,
        *,
        json_response: bool,
        allowed_hosts: list[str],
        allowed_origins: list[str],
    ):
        self._manager = StreamableHTTPSessionManager(
            app=server,
            json_response=json_response,
            stateless=False,
            security_settings=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=list(allowed_hosts),
                allowed_origins=list(allowed_origins),
            ),
        )

    def run(self) -> AbstractAsyncContextManager[None]:
        return self._manager.run()

    async def handle_request(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._manager.handle_request(scope, receive, send)


class LegacySseTransport:
    """Legacy SSE transport (GET /sse + POST /messages/)."""

    def __init__(self, server: Server, *, messages_endpoint: str = "/messages/"):
        self._server = server
        self._transport = SseServerTransport(messages_endpoint)

    async def handle_sse(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with self._transport.connect_sse(scope, receive, send) as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

    async def handle_post_message(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._transport.handle_post_message(scope, receive, send)
