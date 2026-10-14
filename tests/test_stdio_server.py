from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _server_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "MEMORY_SQLITE_PATH": str(tmp_path / "memory.db"),
            "MEMORY_IMPORT_EXPORT_BASE_DIR": str(tmp_path / "exchange"),
            "MCP_MODELS_DIR": str(tmp_path / "models"),
            "HF_HOME": str(tmp_path / "models" / "hf"),
            "TRANSFORMERS_CACHE": str(tmp_path / "models" / "hf" / "transformers"),
            "SENTENCE_TRANSFORMERS_HOME": str(tmp_path / "models" / "hf" / "st"),
            "EMBEDDING_PROVIDER": "hash-local",
            "EMBEDDING_MODEL": "local-hash-stdio-test",
            "MEMORY_ENCRYPTION_ENABLED": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def test_stdio_server_writes_only_json_rpc_to_stdout(tmp_path):
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "stdio-test", "version": "0.0.1"},
        },
    }
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    list_tools = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

    process = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        cwd=PROJECT_ROOT,
        env=_server_env(tmp_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    watchdog = threading.Timer(60, process.kill)
    watchdog.start()
    raw_lines: list[str] = []
    try:

        def send(message: dict) -> None:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()

        def read_message() -> dict:
            line = process.stdout.readline()
            raw_lines.append(line)
            return json.loads(line)

        send(initialize)
        first = read_message()
        send(initialized)
        send(list_tools)
        second = read_message()
        process.stdin.close()
        remaining = process.stdout.read()
        stderr = process.stderr.read()
        process.wait()
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
            process.wait()

    trailing = [line for line in remaining.splitlines() if line.strip()]
    raw_lines.extend(trailing)
    responses = [first, second] + [json.loads(line) for line in trailing]
    responses = [message for message in responses if "id" in message]

    assert len(responses) == 2, f"stdout={raw_lines!r} stderr={stderr!r}"
    assert all(response.get("jsonrpc") == "2.0" for response in responses)
    assert responses[0]["id"] == 1
    assert responses[0]["result"]["serverInfo"]["name"] == "llm-memory"
    assert responses[1]["id"] == 2
    assert any(tool["name"] == "memory.add" for tool in responses[1]["result"]["tools"])
    assert "LLM Memory runtime initialized" in stderr
