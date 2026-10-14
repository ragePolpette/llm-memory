from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest
from mcp.server import Server

from src.mcp_server.mcp_transport import LegacySseTransport, StreamableTransport


def _http_scope(method: str, path: str, headers: list[tuple[bytes, bytes]]) -> dict:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "http_version": "1.1",
        "scheme": "http",
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8767),
    }


@pytest.mark.asyncio
async def test_streamable_transport_serves_initialize_and_enforces_host_allowlist():
    transport = StreamableTransport(
        Server("adapter-test"),
        json_response=True,
        allowed_hosts=["127.0.0.1:*"],
        allowed_origins=["http://127.0.0.1:*"],
    )
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "adapter", "version": "0"},
            },
        }
    ).encode()

    async def call(host: bytes) -> list[dict]:
        sent: list[dict] = []
        messages = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        headers = [
            (b"host", host),
            (b"content-type", b"application/json"),
            (b"accept", b"application/json, text/event-stream"),
        ]
        await transport.handle_request(_http_scope("POST", "/mcp", headers), receive, send)
        return sent

    async with transport.run():
        allowed = await call(b"127.0.0.1:8767")
        rejected = await call(b"evil.example:80")

    allowed_start = next(m for m in allowed if m["type"] == "http.response.start")
    allowed_body = b"".join(m.get("body", b"") for m in allowed if m["type"] == "http.response.body")
    assert allowed_start["status"] == 200
    assert json.loads(allowed_body)["result"]["serverInfo"]["name"] == "adapter-test"

    rejected_start = next(m for m in rejected if m["type"] == "http.response.start")
    assert rejected_start["status"] == 421


@pytest.mark.asyncio
async def test_legacy_sse_transport_runs_server_with_raw_asgi_send():
    server = Server("adapter-test")
    transport = LegacySseTransport(server, messages_endpoint="/messages/")
    seen: dict = {}

    @asynccontextmanager
    async def fake_connect_sse(scope, receive, send):
        seen["send"] = send
        yield ("read-stream", "write-stream")

    async def fake_run(read_stream, write_stream, options):
        seen["streams"] = (read_stream, write_stream)
        seen["options"] = options

    transport._transport.connect_sse = fake_connect_sse  # type: ignore[method-assign]
    server.run = fake_run  # type: ignore[method-assign]

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        return None

    await transport.handle_sse(_http_scope("GET", "/sse", []), receive, send)

    assert seen["send"] is send
    assert seen["streams"] == ("read-stream", "write-stream")
    assert seen["options"].server_name == "adapter-test"


@pytest.mark.asyncio
async def test_legacy_sse_transport_rejects_unknown_session_message():
    transport = LegacySseTransport(Server("adapter-test"))
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = _http_scope("POST", "/messages/", [(b"content-type", b"application/json")])
    scope["query_string"] = b"session_id=00000000000000000000000000000000"
    await transport.handle_post_message(scope, receive, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] in {400, 404}
