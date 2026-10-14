"""Cross-platform launcher for the llm-memory MCP server.

Usage:
    python scripts/serve.py                 # HTTP transport on 127.0.0.1:8767
    python scripts/serve.py --transport stdio
    python scripts/serve.py --host 127.0.0.1 --port 9000

Stop the server with Ctrl+C.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import MutableMapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_DEFAULTS = {
    "EMBEDDING_PROVIDER": "hash-local",
    "EMBEDDING_MODEL": "local-hash-v1",
    "MEMORY_STORAGE_BACKEND": "sqlite",
    "MEMORY_VECTOR_BACKEND": "sqlite",
    "MEMORY_SQLITE_PATH": "./data/memory.db",
    "MEMORY_ALLOW_OUTBOUND_NETWORK": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the llm-memory MCP server.")
    parser.add_argument("--transport", choices=("http", "stdio"), default="http")
    parser.add_argument("--host", help="HTTP bind host (overrides MCP_MEMORY_HOST)")
    parser.add_argument("--port", type=int, help="HTTP port (overrides MCP_MEMORY_PORT)")
    return parser.parse_args(argv)


def apply_environment(args: argparse.Namespace, env: MutableMapping[str, str]) -> None:
    for key, value in LOCAL_DEFAULTS.items():
        env.setdefault(key, value)
    if args.host:
        env["MCP_MEMORY_HOST"] = args.host
    if args.port is not None:
        env["MCP_MEMORY_PORT"] = str(args.port)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    apply_environment(args, os.environ)
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    if args.transport == "stdio":
        from src.mcp_server.server import main as run_stdio

        run_stdio()
    else:
        from src.mcp_server.http_server import main as run_http

        run_http()


if __name__ == "__main__":
    main()
