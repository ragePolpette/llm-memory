from __future__ import annotations

import logging

import pytest
from starlette.testclient import TestClient

import src.mcp_server.http_server as http_server
from src.config import Config
from src.mcp_server.http_server import InsecureBindError, is_loopback_host, validate_bind

TOKEN = "test-token-0123456789abcdef"
MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "http-test", "version": "0.0.1"},
    },
}


@pytest.fixture
def http_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_SQLITE_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_IMPORT_EXPORT_BASE_DIR", str(tmp_path / "exchange"))
    monkeypatch.setenv("MCP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models" / "hf"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "models" / "hf" / "transformers"))
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path / "models" / "hf" / "st"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash-local")
    monkeypatch.setenv("EMBEDDING_MODEL", "local-hash-http-test")
    monkeypatch.setenv("MCP_MEMORY_HOST", "127.0.0.1")
    monkeypatch.setenv("MEMORY_ENCRYPTION_ENABLED", "false")
    monkeypatch.delenv("LLM_MEMORY_API_TOKEN", raising=False)
    yield monkeypatch
    for name in ("runtime", "mcp_server", "sse_transport", "streamable_session_manager", "app_config"):
        monkeypatch.setattr(http_server, name, None)


def test_lifespan_starts_real_app_and_serves_health(http_env, caplog):
    # INFO must be enabled: the original crash only happened once the logger
    # actually formatted the startup diagnostics.
    caplog.set_level(logging.INFO)
    with TestClient(http_server.app, base_url="http://127.0.0.1:8767") as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["diagnostics"]["transport"]["auth_enabled"] is False

        mcp_response = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        assert mcp_response.status_code == 200
        assert mcp_response.json()["result"]["serverInfo"]["name"] == "llm-memory"


def test_admin_endpoints_are_disabled_without_token(http_env):
    with TestClient(http_server.app, base_url="http://127.0.0.1:8767") as client:
        response = client.get("/admin/summary")
        assert response.status_code == 403
        assert "LLM_MEMORY_API_TOKEN" in response.json()["error"]["message"]


def test_token_protects_mcp_and_admin_endpoints(http_env):
    http_env.setenv("LLM_MEMORY_API_TOKEN", TOKEN)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(http_server.app, base_url="http://127.0.0.1:8767") as client:
        assert client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS).status_code == 401
        wrong = {**MCP_HEADERS, "Authorization": "Bearer wrong-token"}
        assert client.post("/mcp", json=INITIALIZE, headers=wrong).status_code == 401
        ok = client.post("/mcp", json=INITIALIZE, headers={**MCP_HEADERS, **auth})
        assert ok.status_code == 200

        unauthorized = client.get("/admin/summary")
        assert unauthorized.status_code == 401
        assert unauthorized.headers["WWW-Authenticate"] == "Bearer"
        assert client.get("/admin/summary", headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
        summary = client.get("/admin/summary", headers=auth)
        assert summary.status_code == 200
        assert summary.json()["status"] == "ok"

        assert client.get("/sse").status_code == 401
        assert client.post("/messages/?session_id=x", json={}).status_code == 401

        public_health = client.get("/health").json()
        assert public_health["status"] == "ok"
        assert public_health["diagnostics"] is None
        private_health = client.get("/health", headers=auth).json()
        assert private_health["diagnostics"]["transport"]["auth_enabled"] is True
        assert TOKEN not in str(private_health)


def test_lifespan_refuses_non_loopback_bind_without_token(http_env):
    http_env.setenv("MCP_MEMORY_HOST", "0.0.0.0")
    with pytest.raises(InsecureBindError, match="LLM_MEMORY_API_TOKEN"):
        with TestClient(http_server.app):
            pass


def test_main_exits_with_clear_error_on_insecure_bind(http_env):
    http_env.setenv("MCP_MEMORY_HOST", "0.0.0.0")
    calls: list[tuple] = []
    http_env.setattr(http_server.uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    with pytest.raises(SystemExit, match="LLM_MEMORY_API_TOKEN"):
        http_server.main()
    assert calls == []

    http_env.setenv("LLM_MEMORY_API_TOKEN", TOKEN)
    http_server.main()
    assert calls and calls[0][1]["host"] == "0.0.0.0"


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.0.0.5", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.10", False),
        ("example.com", False),
    ],
)
def test_is_loopback_host(host, expected):
    assert is_loopback_host(host) is expected


def test_validate_bind_accepts_token_or_loopback(tmp_path):
    base = {"sqlite_db_path": tmp_path / "m.db", "import_export_base_dir": tmp_path / "x"}
    validate_bind(Config(mcp_host="127.0.0.1", api_token=None, **base))
    validate_bind(Config(mcp_host="0.0.0.0", api_token=TOKEN, **base))
    with pytest.raises(InsecureBindError):
        validate_bind(Config(mcp_host="0.0.0.0", api_token="   ", **base))
