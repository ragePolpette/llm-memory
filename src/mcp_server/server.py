"""MCP stdio server for LLM Memory v2 (local-only)."""

from __future__ import annotations

import asyncio
import sys

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ..bootstrap import build_runtime
from ..config import get_config
from ..logging_setup import configure_logging
from .tools import register_tools

configure_logging()
logger = structlog.get_logger()


class MemoryServer:
    """Local MCP server backed by the v2 runtime."""

    def __init__(self):
        config = get_config()
        self.runtime = build_runtime(config)
        self.server = Server("llm-memory")
        register_tools(self.server, self.runtime.service)
        logger.info("LLM Memory runtime initialized", **config.startup_diagnostics())

    async def run(self):
        await self.runtime.prewarm()
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP server running on stdio")
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


def main():
    server = MemoryServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as exc:  # pragma: no cover
        logger.error("Server error", error=str(exc), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
