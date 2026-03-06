from __future__ import annotations

import pytest

from src.mcp_server.http_server import (
    legacy_sse_method_not_allowed,
    routes,
    sse_legacy_endpoint,
    streamable_http_app,
)


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
