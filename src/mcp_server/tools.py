"""Definizione tools MCP per LLM Memory."""

from __future__ import annotations

import logging
from typing import Literal, Optional
from uuid import UUID

from mcp.server import Server
from mcp.types import Tool, TextContent

from ..config import MemoryScope
from ..coordination.conflict_resolver import ConflictResolver
from ..coordination.scope_manager import ScopeManager
from ..indexing.indexer import MemoryIndexer
from ..models import Memory, MemorySummary, MemoryWriteResult
from ..storage.markdown_store import MarkdownStore
from ..vectordb.lance_store import LanceVectorStore

logger = logging.getLogger(__name__)


def register_tools(
    server: Server,
    markdown_store: MarkdownStore,
    vector_store: LanceVectorStore,
    indexer: MemoryIndexer,
    scope_manager: ScopeManager,
    conflict_resolver: ConflictResolver,
):
    """Registra tutti i tools MCP sul server."""
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Lista i tools disponibili."""
        return [
            Tool(
                name="memory_write",
                description="Salva una memoria nel sistema condiviso",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Contenuto della memoria"},
                        "context": {"type": "string", "description": "Contesto semantico"},
                        "agent_id": {"type": "string", "description": "ID dell'agente"},
                        "scope": {
                            "type": "string",
                            "enum": ["private", "shared", "global"],
                            "default": "shared",
                            "description": "Scope di visibilità"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                            "description": "Tag per categorizzazione"
                        },
                        "session_id": {"type": "string", "description": "ID sessione (opzionale)"},
                    },
                    "required": ["content", "context", "agent_id"],
                },
            ),
            Tool(
                name="memory_search",
                description="Ricerca semantica nelle memorie",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Query di ricerca"},
                        "agent_id": {"type": "string", "description": "ID dell'agente richiedente"},
                        "scope": {
                            "type": "string",
                            "enum": ["all", "private", "shared", "global"],
                            "default": "all",
                            "description": "Scope da cercare"
                        },
                        "limit": {"type": "integer", "default": 10, "description": "Numero massimo risultati"},
                    },
                    "required": ["query", "agent_id"],
                },
            ),
            Tool(
                name="memory_read",
                description="Legge una memoria specifica per ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "UUID della memoria"},
                        "agent_id": {"type": "string", "description": "ID dell'agente richiedente"},
                    },
                    "required": ["memory_id", "agent_id"],
                },
            ),
            Tool(
                name="memory_list",
                description="Lista memorie per scope/agente",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "ID dell'agente richiedente"},
                        "scope": {
                            "type": "string",
                            "enum": ["private", "shared", "global"],
                            "default": "shared",
                            "description": "Scope da listare"
                        },
                        "limit": {"type": "integer", "default": 50, "description": "Numero massimo risultati"},
                    },
                    "required": ["agent_id"],
                },
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Esegue un tool."""
        
        if name == "memory_write":
            return await _memory_write(
                arguments,
                markdown_store,
                indexer,
                scope_manager,
                conflict_resolver
            )
        
        elif name == "memory_search":
            return await _memory_search(
                arguments,
                vector_store,
                scope_manager
            )
        
        elif name == "memory_read":
            return await _memory_read(
                arguments,
                markdown_store,
                scope_manager
            )
        
        elif name == "memory_list":
            return await _memory_list(
                arguments,
                markdown_store,
                scope_manager
            )
        
        else:
            raise ValueError(f"Unknown tool: {name}")


async def _memory_write(
    args: dict,
    markdown_store: MarkdownStore,
    indexer: MemoryIndexer,
    scope_manager: ScopeManager,
    conflict_resolver: ConflictResolver,
) -> list[TextContent]:
    """Implementa memory_write."""
    
    scope = MemoryScope(args.get("scope", "shared"))
    agent_id = args["agent_id"]
    
    # Verifica permessi
    if not scope_manager.can_write(agent_id, scope):
        return [TextContent(
            type="text",
            text=f"Error: Agent {agent_id} cannot write to scope {scope.value}"
        )]
    
    # Crea memoria
    memory = Memory(
        content=args["content"],
        context=args["context"],
        agent_id=agent_id,
        scope=scope,
        tags=args.get("tags", []),
        session_id=args.get("session_id"),
    )
    
    # Check duplicati
    duplicate_id = await conflict_resolver.check_duplicate(memory)
    if duplicate_id:
        result = MemoryWriteResult(
            success=True,
            memory_id=UUID(duplicate_id),
            indexed=True,
            mode="duplicate",
            duplicate_of=UUID(duplicate_id),
            message="Content already exists, returning existing memory ID"
        )
        return [TextContent(type="text", text=result.model_dump_json(indent=2))]
    
    # Salva su filesystem
    await markdown_store.write(memory)
    
    # Indicizza
    index_result = await indexer.index(memory)
    
    result = MemoryWriteResult(
        success=True,
        memory_id=memory.id,
        indexed=index_result.indexed,
        mode=index_result.mode,
        message="Memory saved successfully"
    )
    
    return [TextContent(type="text", text=result.model_dump_json(indent=2))]


async def _memory_search(
    args: dict,
    vector_store: LanceVectorStore,
    scope_manager: ScopeManager,
) -> list[TextContent]:
    """Implementa memory_search."""
    
    query = args["query"]
    agent_id = args["agent_id"]
    scope = args.get("scope", "all")
    limit = args.get("limit", 10)
    
    # Costruisci filtro SQL per scope
    sql_filter = None
    if scope != "all":
        sql_filter = f"scope = '{scope}'"
    
    # Esegui ricerca
    results = await vector_store.search(query, limit=limit, filters=sql_filter)
    
    # Filtra per permessi
    accessible_results = []
    for result in results:
        memory = Memory(
            id=result.memory_id,
            content=result.content,
            context=result.context,
            agent_id=result.agent_id,
            scope=MemoryScope(result.scope),
            tags=result.tags,
            created_at=result.created_at,
        )
        if scope_manager.can_read(agent_id, memory):
            accessible_results.append(result)
    
    # Formatta risultati
    results_json = [r.model_dump() for r in accessible_results]
    
    return [TextContent(
        type="text",
        text=f"Found {len(accessible_results)} results:\n\n" + 
             "\n\n".join([f"**Score: {r['score']:.3f}**\n{r['content'][:200]}..." for r in results_json])
    )]


async def _memory_read(
    args: dict,
    markdown_store: MarkdownStore,
    scope_manager: ScopeManager,
) -> list[TextContent]:
    """Implementa memory_read."""
    
    memory_id = args["memory_id"]
    agent_id = args["agent_id"]
    
    # Leggi memoria
    memory = await markdown_store.read(memory_id)
    
    if not memory:
        return [TextContent(type="text", text=f"Memory {memory_id} not found")]
    
    # Verifica permessi
    if not scope_manager.can_read(agent_id, memory):
        return [TextContent(type="text", text=f"Access denied to memory {memory_id}")]
    
    return [TextContent(type="text", text=memory.model_dump_json(indent=2))]


async def _memory_list(
    args: dict,
    markdown_store: MarkdownStore,
    scope_manager: ScopeManager,
) -> list[TextContent]:
    """Implementa memory_list."""
    
    agent_id = args["agent_id"]
    scope = MemoryScope(args.get("scope", "shared"))
    limit = args.get("limit", 50)
    
    # Lista file
    paths = await markdown_store.list_memories(scope=scope, agent_id=agent_id, limit=limit)
    
    # Leggi e filtra
    summaries = []
    for path in paths:
        memory = await markdown_store.read_by_path(path)
        if memory and scope_manager.can_read(agent_id, memory):
            summary = MemorySummary(
                memory_id=memory.id,
                context=memory.context,
                agent_id=memory.agent_id,
                scope=memory.scope,
                tags=memory.tags,
                created_at=memory.created_at,
                content_preview=memory.content[:200]
            )
            summaries.append(summary)
    
    summaries_json = [s.model_dump() for s in summaries]
    
    return [TextContent(
        type="text",
        text=f"Found {len(summaries)} memories:\n\n" +
             "\n\n".join([f"**{s['context']}** ({s['created_at']})\n{s['content_preview']}..." for s in summaries_json])
    )]
