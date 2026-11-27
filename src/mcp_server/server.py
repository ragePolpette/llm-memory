"""MCP stdio server for llm-memory, backed by the v2 store."""

from __future__ import annotations

import asyncio
import sys

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ..logging_setup import configure_logging
from ..store_v2.runtime import build_runtime
from .tools_v2 import register_tools_v2

configure_logging()
logger = structlog.get_logger()


class MemoryServer:
    def __init__(self):
        self.runtime = build_runtime()
        self.server = Server("llm-memory")
        register_tools_v2(self.server, self.runtime.service)
        logger.info("llm-memory runtime initialized", **self.runtime.settings.diagnostics())

    async def run(self):
        self.runtime.start_background()
        try:
            async with stdio_server() as (read_stream, write_stream):
                logger.info("MCP server running on stdio")
                await self.server.run(read_stream, write_stream, self.server.create_initialization_options())
        finally:
            await self.runtime.stop_background()
            self.runtime.close()


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
