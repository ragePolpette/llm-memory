"""MCP Server stdio per LLM Memory v2 (local-only)."""

from __future__ import annotations

import asyncio
import logging
import sys

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ..bootstrap import build_runtime
from ..config import get_config
from .tools import register_tools

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
logger = structlog.get_logger()


class MemoryServer:
    """Server MCP locale con runtime v2."""

    def __init__(self):
        config = get_config()
        self.runtime = build_runtime(config)
        self.server = Server("llm-memory")
        register_tools(self.server, self.runtime.service)
        logger.info(
            "LLM Memory runtime initialized",
            sqlite_db=str(config.sqlite_db_path),
            embedding_provider=str(config.embedding_provider),
            embedding_model=config.embedding_model,
            allow_outbound_network=config.allow_outbound_network,
        )

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
