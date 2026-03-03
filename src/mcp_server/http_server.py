"""HTTP server MCP con supporto Streamable HTTP + SSE legacy.

Avvio:
  python -m src.mcp_server.http_server

Endpoint:
  - /mcp           Streamable HTTP (GET/POST/DELETE)
  - /sse           SSE legacy (GET)
  - /messages/     SSE legacy message POST endpoint
  - /sse (POST)    Alias streamable per backward compatibility con config esistenti
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Config, get_config
from src.coordination.conflict_resolver import ConflictResolver
from src.coordination.scope_manager import ScopeManager
from src.embedding.embedding_service import get_embedding_provider
from src.indexing.indexer import MemoryIndexer
from src.mcp_server.tools import register_tools
from src.storage.markdown_store import MarkdownStore
from src.vectordb.lance_store import LanceVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Stato globale server HTTP
config: Config | None = None
mcp_server: Server | None = None
indexer: MemoryIndexer | None = None
sse_transport: SseServerTransport | None = None
streamable_session_manager: StreamableHTTPSessionManager | None = None


def init_components() -> None:
    """Inizializza componenti dominio + server MCP low-level."""
    global config, mcp_server, indexer, sse_transport, streamable_session_manager

    logger.info("Initializing LLM Memory components...")
    config = get_config()

    logger.info(f"Loading embedding model: {config.embedding_model}")
    embedding_provider = get_embedding_provider(
        model_name=config.embedding_model,
        device=config.embedding_device,
    )

    markdown_store = MarkdownStore(config.storage_dir)
    vector_store = LanceVectorStore(config.lancedb_dir, embedding_provider)
    indexer = MemoryIndexer(
        vector_store=vector_store,
        mode=config.indexing_mode,
        hybrid_threshold_bytes=config.hybrid_threshold_bytes,
        queue_max_size=config.queue_max_size,
    )
    scope_manager = ScopeManager()
    conflict_resolver = ConflictResolver(markdown_store, vector_store)

    mcp_server = Server("llm-memory")
    register_tools(
        server=mcp_server,
        markdown_store=markdown_store,
        vector_store=vector_store,
        indexer=indexer,
        scope_manager=scope_manager,
        conflict_resolver=conflict_resolver,
    )

    @mcp_server.list_resources()
    async def _list_resources():
        return []

    @mcp_server.list_resource_templates()
    async def _list_resource_templates():
        return []

    @mcp_server.list_prompts()
    async def _list_prompts():
        return []
    # Legacy SSE: GET /sse -> endpoint event con /messages/
    sse_transport = SseServerTransport("/messages/")

    # Streamable HTTP moderno (session-based)
    streamable_session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        json_response=True,
        stateless=False,
    )

    logger.info("✓ Components initialized successfully")


@asynccontextmanager
async def lifespan(app: Starlette):
    """Lifecycle app: init componenti, worker indexer, session manager streamable."""
    init_components()

    assert config is not None
    assert indexer is not None
    assert streamable_session_manager is not None

    if config.indexing_mode.value in ["async", "hybrid"]:
        await indexer.start_worker()
        logger.info("Async indexing worker started")

    async with streamable_session_manager.run():
        yield

    if config.indexing_mode.value in ["async", "hybrid"]:
        await indexer.stop_worker()
        logger.info("Async indexing worker stopped")


class StreamableHTTPASGIApp:
    """ASGI adapter per StreamableHTTPSessionManager."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if streamable_session_manager is None:
            response = Response("Server not initialized", status_code=503)
            await response(scope, receive, send)
            return
        await streamable_session_manager.handle_request(scope, receive, send)


class LegacyMessagesASGIApp:
    """ASGI adapter per endpoint POST dei messaggi SSE legacy."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if sse_transport is None:
            response = Response("Server not initialized", status_code=503)
            await response(scope, receive, send)
            return
        await sse_transport.handle_post_message(scope, receive, send)


streamable_http_app = StreamableHTTPASGIApp()
legacy_messages_app = LegacyMessagesASGIApp()


async def sse_legacy_endpoint(request: Request):
    """GET /sse: avvia connessione SSE legacy con endpoint bootstrap /messages/."""
    if sse_transport is None or mcp_server is None:
        return Response("Server not initialized", status_code=503)

    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
        read_stream, write_stream = streams
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )

    # Necessario dopo disconnessione client per evitare NoneType error in Starlette
    return Response()


async def health(request: Request):
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "server": "llm-memory"})


async def wellknown_oauth(request: Request):
    """OAuth discovery (non implementato)."""
    return JSONResponse({}, status_code=404)


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/.well-known/oauth-authorization-server", wellknown_oauth, methods=["GET"]),

    # Streamable HTTP canonico
    Route("/mcp", endpoint=streamable_http_app, methods=["GET", "POST", "DELETE"]),

    # Legacy SSE canonico
    Route("/sse", sse_legacy_endpoint, methods=["GET"]),
    Mount("/messages/", app=legacy_messages_app),

    # Backward compatibility: config esistente punta a /sse con http-first
    Route("/sse", endpoint=streamable_http_app, methods=["POST", "DELETE"]),
]

app = Starlette(routes=routes, lifespan=lifespan)


if __name__ == "__main__":
    print("=" * 50)
    print("  LLM Memory HTTP Server")
    print("  http://127.0.0.1:8767")
    print("  Streamable: /mcp")
    print("  Legacy SSE: /sse + /messages/")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8767, log_level="info")

