from __future__ import annotations

import importlib
import importlib.util
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_serve():
    spec = importlib.util.spec_from_file_location("serve_script", PROJECT_ROOT / "scripts" / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_serve_applies_local_defaults_without_overriding_user_env():
    serve = _load_serve()
    env = {"EMBEDDING_MODEL": "custom-model"}

    serve.apply_environment(serve.parse_args(["--host", "127.0.0.1", "--port", "9001"]), env)

    assert env["EMBEDDING_MODEL"] == "custom-model"
    assert env["EMBEDDING_PROVIDER"] == "hash-local"
    assert env["MEMORY_ALLOW_OUTBOUND_NETWORK"] == "false"
    assert env["MCP_MEMORY_HOST"] == "127.0.0.1"
    assert env["MCP_MEMORY_PORT"] == "9001"


@pytest.mark.parametrize(("transport", "target"), [("http", "src.mcp_server.http_server"), ("stdio", "src.mcp_server.server")])
def test_serve_dispatches_to_selected_transport(monkeypatch, transport, target):
    serve = _load_serve()
    calls: list[str] = []
    module = importlib.import_module(target)
    monkeypatch.setattr(module, "main", lambda: calls.append(transport))
    monkeypatch.chdir(PROJECT_ROOT)
    for key in serve.LOCAL_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("MCP_MEMORY_HOST", raising=False)

    serve.main(["--transport", transport])

    assert calls == [transport]


def test_windows_batch_scripts_are_gone():
    assert not list(PROJECT_ROOT.rglob("*.bat"))


def test_unused_dependencies_are_not_declared():
    deps = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    names = {dep.split(">")[0].split("<")[0].split("=")[0].strip().lower() for dep in deps}
    assert "fastmcp" not in names
    assert "pyyaml" not in names


def test_mcp_transport_internals_are_isolated_in_adapter():
    offenders = []
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        if path.name == "mcp_transport.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "streamable_http_manager" in text or "mcp.server.sse" in text or "request._send" in text:
            offenders.append(path.name)
    assert offenders == []


def test_compose_publishes_loopback_only_and_requires_token():
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert '"127.0.0.1:8767:8767"' in compose
    assert "LLM_MEMORY_API_TOKEN=${LLM_MEMORY_API_TOKEN:?" in compose


def test_dockerfile_runs_as_non_root_user():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    user_lines = [line.split()[1] for line in dockerfile.splitlines() if line.startswith("USER ")]
    assert user_lines and user_lines[-1] not in {"root", "0"}
