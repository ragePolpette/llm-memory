"""MCP Server per LLM Memory."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ..config import Config, get_config
from ..coordination.conflict_resolver import ConflictResolver
from ..coordination.scope_manager import ScopeManager
from ..embedding.embedding_service import get_embedding_provider
from ..indexing.indexer import MemoryIndexer
from ..storage.markdown_store import MarkdownStore
from ..vectordb.lance_store import LanceVectorStore
from .tools import register_tools

# Setup logging
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
    """Server MCP per gestione memoria condivisa."""
    
    def __init__(self, config: Config):
        self.config = config
        self.server = Server("llm-memory")
        
        # Inizializza componenti
        logger.info("Initializing LLM Memory server...")
        
        # Embedding provider (scarica modello in locale)
        logger.info(f"Loading embedding model: {config.embedding_model}")
        self.embedding_provider = get_embedding_provider(
            model_name=config.embedding_model,
            device=config.embedding_device
        )
        
        # Storage
        self.markdown_store = MarkdownStore(config.storage_dir)
        logger.info(f"Markdown storage: {config.storage_dir}")
        
        # Vector DB
        self.vector_store = LanceVectorStore(
            config.lancedb_dir,
            self.embedding_provider
        )
        logger.info(f"LanceDB storage: {config.lancedb_dir}")
        
        # Indexer
        self.indexer = MemoryIndexer(
            vector_store=self.vector_store,
            mode=config.indexing_mode,
            hybrid_threshold_bytes=config.hybrid_threshold_bytes,
            queue_max_size=config.queue_max_size,
        )
        logger.info(f"Indexing mode: {config.indexing_mode}")
        
        # Coordination
        self.scope_manager = ScopeManager()
        self.conflict_resolver = ConflictResolver(
            self.markdown_store,
            self.vector_store
        )
        
        # Registra tools MCP
        register_tools(
            server=self.server,
            markdown_store=self.markdown_store,
            vector_store=self.vector_store,
            indexer=self.indexer,
            scope_manager=self.scope_manager,
            conflict_resolver=self.conflict_resolver,
        )
        
        logger.info("LLM Memory server initialized successfully")
    
    async def run(self):
        """Avvia il server MCP."""
        # Avvia worker async se necessario
        if self.config.indexing_mode.value in ["async", "hybrid"]:
            await self.indexer.start_worker()
            logger.info("Async indexing worker started")
        
        try:
            async with stdio_server() as (read_stream, write_stream):
                logger.info("MCP server running on stdio")
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
        finally:
            # Cleanup
            if self.config.indexing_mode.value in ["async", "hybrid"]:
                await self.indexer.stop_worker()
                logger.info("Async indexing worker stopped")


def main():
    """Entry point del server."""
    config = get_config()
    server = MemoryServer(config)
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
