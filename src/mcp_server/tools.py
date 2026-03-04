"""Definizione tools MCP v2 + wrapper legacy v1."""

from __future__ import annotations

from pathlib import Path

from mcp.server import Server
from mcp.types import Tool, TextContent

from ..config import Tier
from ..service.memory_service import ActorContext, MemoryService


def _json_text(payload: dict | list) -> list[TextContent]:
    import json

    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=True, indent=2, default=str))]


def _actor_from_args(args: dict, service: MemoryService) -> ActorContext:
    raw_scope = args.get("scope")
    scope = raw_scope if isinstance(raw_scope, dict) else {}
    agent_id = args.get("agent_id") or scope.get("agent_id") or "unknown-agent"
    user_id = args.get("user_id") or scope.get("user_id")
    workspace_id = scope.get("workspace_id", service.config.default_workspace_id)
    project_id = scope.get("project_id", service.config.default_project_id)
    return ActorContext(
        agent_id=agent_id,
        user_id=user_id,
        workspace_id=workspace_id,
        project_id=project_id,
    )


def register_tools(server: Server, memory_service: MemoryService):
    """Registra tool MCP v2 e compatibilità v1."""

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="memory.about",
                description="Spiega scopo e confini: memorie operative persistenti (non retrieval di codice repository).",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="memory.add",
                description="Aggiunge memoria operativa persistente tiered (decisioni, fatti, regole operative).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "context": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "tier": {"type": "string", "enum": ["tier-1", "tier-2", "tier-3"], "default": "tier-1"},
                        "type": {
                            "type": "string",
                            "enum": ["fact", "assumption", "unknown", "decision", "invalidated"],
                            "default": "fact",
                        },
                        "visibility": {
                            "type": "string",
                            "enum": ["private", "shared", "global"],
                            "default": "shared",
                        },
                        "scope": {
                            "type": "object",
                            "properties": {
                                "workspace_id": {"type": "string"},
                                "project_id": {"type": "string"},
                                "user_id": {"type": "string"},
                                "agent_id": {"type": "string"},
                            },
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "sensitivity_tags": {"type": "array", "items": {"type": "string"}},
                        "links": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "target_id": {"type": "string"},
                                    "relation": {"type": "string"},
                                },
                                "required": ["target_id", "relation"],
                            },
                        },
                        "metadata": {"type": "object"},
                        "source": {"type": "string", "default": "mcp"},
                        "confidence": {"type": "number", "default": 0.5},
                    },
                    "required": ["content", "agent_id"],
                },
            ),
            Tool(
                name="memory.search",
                description="Ricerca semantica su memorie operative persistenti; non destinato a contesto codice repository.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                        "limit": {"type": "integer", "default": 10},
                        "include_invalidated": {"type": "boolean", "default": False},
                        "tier": {"type": "string", "enum": ["tier-1", "tier-2", "tier-3"]},
                    },
                    "required": ["query", "agent_id"],
                },
            ),
            Tool(
                name="memory.get",
                description="Recupera una memoria operativa per id (v2)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entry_id": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                    },
                    "required": ["entry_id", "agent_id"],
                },
            ),
            Tool(
                name="memory.invalidate",
                description="Invalida memorie obsolete/errate con motivo e audit trail (v2)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target_ids": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                    },
                    "required": ["target_ids", "reason", "agent_id"],
                },
            ),
            Tool(
                name="memory.promote",
                description="Promuove memorie operative verso tier superiore con consolidamento opzionale (v2)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entry_ids": {"type": "array", "items": {"type": "string"}},
                        "target_tier": {
                            "type": "string",
                            "enum": ["tier-1", "tier-2", "tier-3"],
                            "default": "tier-3",
                        },
                        "reason": {"type": "string"},
                        "merge": {"type": "boolean", "default": False},
                        "summary": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                    },
                    "required": ["entry_ids", "reason", "agent_id"],
                },
            ),
            Tool(
                name="memory.reembed",
                description="Reindex/reembed incrementale delle memorie operative (v2)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                        "model_id": {"type": "string"},
                        "dim": {"type": "integer"},
                        "activate": {"type": "boolean", "default": True},
                        "batch_size": {"type": "integer", "default": 64},
                    },
                    "required": ["agent_id"],
                },
            ),
            Tool(
                name="memory.export",
                description="Export locale memorie operative in jsonl / memory.md / sqlite (v2)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "format": {"type": "string", "enum": ["jsonl", "memory.md", "sqlite"]},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                    },
                    "required": ["path", "format", "agent_id"],
                },
            ),
            Tool(
                name="memory.import",
                description="Import locale memorie operative da jsonl / memory.md (v2)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "format": {"type": "string", "enum": ["jsonl", "memory.md"]},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                    },
                    "required": ["path", "format", "agent_id"],
                },
            ),
            # ---- wrapper legacy v1 ----
            Tool(
                name="memory_write",
                description="Compat legacy v1 -> memory.add (memoria operativa)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "context": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "scope": {"type": "string", "enum": ["private", "shared", "global"], "default": "shared"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "session_id": {"type": "string"},
                    },
                    "required": ["content", "context", "agent_id"],
                },
            ),
            Tool(
                name="memory_search",
                description="Compat legacy v1 -> memory.search (memoria operativa)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query", "agent_id"],
                },
            ),
            Tool(
                name="memory_read",
                description="Compat legacy v1 -> memory.get (memoria operativa)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "agent_id": {"type": "string"},
                    },
                    "required": ["memory_id", "agent_id"],
                },
            ),
            Tool(
                name="memory_list",
                description="Compat legacy v1 listing (memorie operative)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "limit": {"type": "integer", "default": 50},
                        "tier": {"type": "string", "enum": ["tier-1", "tier-2", "tier-3"]},
                    },
                    "required": ["agent_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        actor = _actor_from_args(arguments, memory_service)

        if name == "memory.about":
            return _json_text(
                {
                    "api_version": "v2",
                    "server": "llm-memory",
                    "purpose": "Memoria operativa persistente (decisioni, preferenze, regole, fatti di lavoro).",
                    "capabilities": [
                        "add/search/get/invalidate/promote/reembed/export/import"
                    ],
                    "boundaries": [
                        "Non e' un indicizzatore di codice repository",
                        "Per contesto codice/documenti usare llm-context",
                    ],
                }
            )

        if name == "memory.add":
            payload = await memory_service.add(arguments, actor)
            payload["api_version"] = "v2"
            return _json_text(payload)

        if name == "memory.search":
            bundles = await memory_service.search(
                query=arguments["query"],
                actor=actor,
                limit=int(arguments.get("limit", 10)),
                include_invalidated=bool(arguments.get("include_invalidated", False)),
                tier=arguments.get("tier"),
            )
            return _json_text({"api_version": "v2", "count": len(bundles), "bundles": [b.model_dump(mode="json") for b in bundles]})

        if name == "memory.get":
            entry = memory_service.get(arguments["entry_id"], actor)
            return _json_text({"api_version": "v2", "entry": entry.model_dump(mode="json") if entry else None})

        if name == "memory.invalidate":
            payload = memory_service.invalidate(
                target_ids=list(arguments.get("target_ids", [])),
                actor=actor,
                reason=arguments["reason"],
            )
            payload["api_version"] = "v2"
            return _json_text(payload)

        if name == "memory.promote":
            payload = memory_service.promote(
                entry_ids=list(arguments.get("entry_ids", [])),
                actor=actor,
                target_tier=Tier(arguments.get("target_tier", "tier-3")),
                reason=arguments.get("reason", "promotion"),
                merge=bool(arguments.get("merge", False)),
                summary=arguments.get("summary"),
            )
            payload["api_version"] = "v2"
            return _json_text(payload)

        if name == "memory.reembed":
            result = await memory_service.reembed(
                actor=actor,
                model_id=arguments.get("model_id"),
                dim=arguments.get("dim"),
                activate=bool(arguments.get("activate", True)),
                batch_size=int(arguments.get("batch_size", 64)),
            )
            return _json_text({"api_version": "v2", **result.model_dump(mode="json")})

        if name == "memory.export":
            result = memory_service.export_data(
                path=Path(arguments["path"]).expanduser(),
                fmt=arguments["format"],
                actor=actor,
            )
            return _json_text({"api_version": "v2", **result.model_dump(mode="json")})

        if name == "memory.import":
            result = await memory_service.import_data(
                path=Path(arguments["path"]).expanduser(),
                fmt=arguments["format"],
                actor=actor,
            )
            return _json_text({"api_version": "v2", **result.model_dump(mode="json")})

        # ----- legacy wrappers -----
        if name == "memory_write":
            payload = {
                "content": arguments["content"],
                "context": arguments.get("context", ""),
                "agent_id": arguments["agent_id"],
                "visibility": arguments.get("scope", "shared"),
                "tags": arguments.get("tags", []),
                "type": "fact",
                "tier": "tier-2",
                "metadata": {"session_id": arguments.get("session_id")},
            }
            result = await memory_service.add(payload, actor)
            return _json_text({"api_version": "v1", "result": result})

        if name == "memory_search":
            bundles = await memory_service.search(
                query=arguments["query"],
                actor=actor,
                limit=int(arguments.get("limit", 10)),
            )
            legacy = [
                {
                    "memory_id": bundle.entry_id,
                    "score": bundle.score,
                    "snippet": bundle.snippet,
                    "tier": bundle.tier.value,
                    "status": bundle.status.value,
                }
                for bundle in bundles
            ]
            return _json_text({"api_version": "v1", "count": len(legacy), "results": legacy})

        if name == "memory_read":
            entry = memory_service.get(arguments["memory_id"], actor)
            return _json_text({"api_version": "v1", "entry": entry.model_dump(mode="json") if entry else None})

        if name == "memory_list":
            entries = memory_service.list_entries(
                actor=actor,
                limit=int(arguments.get("limit", 50)),
                tier=arguments.get("tier"),
            )
            return _json_text(
                {
                    "api_version": "v1",
                    "count": len(entries),
                    "entries": [
                        {
                            "memory_id": entry.id,
                            "context": entry.context,
                            "scope": entry.visibility.value,
                            "tier": entry.tier.value,
                            "created_at": entry.created_at,
                            "content_preview": entry.content[:200],
                        }
                        for entry in entries
                    ],
                }
            )

        raise ValueError(f"Unknown tool: {name}")
