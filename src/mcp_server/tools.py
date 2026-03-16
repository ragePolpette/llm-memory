"""Definizione tools MCP v2."""

from __future__ import annotations

from pathlib import Path

from mcp.server import Server
from mcp.types import Tool, TextContent

from ..config import Tier
from ..service.memory_service import ActorContext, MemoryInputError, MemoryService


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
    writer_context = args.get("writer_context")
    writer_model = None
    writer_model_source = None
    if isinstance(writer_context, dict):
        writer_model = writer_context.get("model")
        writer_model_source = writer_context.get("source")
    return ActorContext(
        agent_id=agent_id,
        user_id=user_id,
        workspace_id=workspace_id,
        project_id=project_id,
        writer_model=writer_model,
        writer_model_source=writer_model_source,
    )


def register_tools(server: Server, memory_service: MemoryService):
    """Registra tool MCP v2."""

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="memory.about",
                description=(
                    "Scopo, confini e guida compilazione: salva solo memorie operative persistenti "
                    "(decisioni, fatti stabili, regole, assunzioni). Non salvare contesto temporaneo "
                    "di chat o retrieval di codice repository."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="memory.list_projects",
                description="Elenca i progetti registrati nel workspace corrente.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                    },
                    "required": ["agent_id"],
                },
            ),
            Tool(
                name="memory.get_project_info",
                description="Restituisce metadata e stato di un progetto registrato.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                    },
                    "required": ["project_id", "agent_id"],
                },
            ),
            Tool(
                name="memory.create_project",
                description="Crea esplicitamente un progetto nel workspace corrente se non esiste.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "display_name": {"type": "string"},
                        "description": {"type": "string"},
                        "metadata": {"type": "object"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                    },
                    "required": ["project_id", "agent_id"],
                },
            ),
            Tool(
                name="memory.add",
                description=(
                    "Aggiunge memoria operativa persistente tiered. Usare per decisioni/fatti/regole riusabili, "
                    "non per dump di contesto. Compilare sempre writer_model + context_fingerprint + importance "
                    "quando enforcement self-eval e attivo."
                ),
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
                        "writer_context": {
                            "type": "object",
                            "description": "Trusted writer identity injected by runtime/transport.",
                            "properties": {
                                "model": {"type": "string"},
                                "source": {"type": "string", "default": "runtime"},
                            },
                        },
                        "writer_model": {"type": "string"},
                        "scope_label": {
                            "type": "string",
                            "enum": ["shared", "project", "agent"],
                        },
                        "context_fingerprint": {
                            "type": "object",
                            "properties": {
                                "conversation_id": {"type": "string"},
                                "task_id": {"type": "string"},
                                "retrieved_ids": {"type": "array", "items": {"type": "string"}},
                                "tool_trace_fingerprint": {"type": ["string", "object", "array"]},
                                "prompt_fingerprint": {"type": ["string", "object", "array"]},
                            },
                        },
                        "importance": {
                            "type": "object",
                            "properties": {
                                "confidence": {"type": "number"},
                                "predictive_confidence": {"type": "number"},
                                "predictive_confidence_before": {"type": "number"},
                                "proxy_disagreement": {"type": "number"},
                                "disagreement_score": {"type": "number"},
                                "surprise_self_rating": {"type": "number"},
                                "self_rating": {"type": "number"},
                                "tool_steps": {"type": "integer", "minimum": 0},
                                "correction_count": {"type": "integer", "minimum": 0},
                                "inference_level": {"type": "integer", "minimum": 0, "maximum": 5},
                                "inference_steps": {"type": "integer", "minimum": 0},
                                "negative_impact": {"type": "number"},
                                "is_external": {"type": "boolean"},
                            },
                        },
                        "tool_steps": {"type": "integer", "minimum": 0},
                        "correction_count": {"type": "integer", "minimum": 0},
                        "inference_level": {"type": "integer", "minimum": 0, "maximum": 5},
                        "negative_impact": {"type": "number"},
                        "is_external": {"type": "boolean"},
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
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        actor = _actor_from_args(arguments, memory_service)

        if name == "memory.about":
            return _json_text(
                {
                    "api_version": "v2",
                    "server": "llm-memory",
                    "self_eval_enforced": bool(memory_service.config.self_eval_enforced),
                    "purpose": "Memoria operativa persistente (decisioni, preferenze, regole, fatti di lavoro).",
                    "capabilities": [
                        "list_projects/get_project_info/create_project/add/search/get/invalidate/promote/reembed/export/import"
                    ],
                    "boundaries": [
                        "Non salvare memorie di contesto temporaneo conversazionale",
                        "Non e' un indicizzatore di codice repository",
                        "Per contesto codice/documenti usare llm-context",
                        "La persistenza e' deny-by-default e richiede superamento della policy centrale",
                    ],
                    "what_to_store": [
                        "Decisioni operative durature",
                        "Fatti stabili di progetto e vincoli",
                        "Assunzioni/unknown da tracciare nel tempo",
                        "Regole di lavoro riusabili",
                    ],
                    "how_to_fill_memory_add": {
                        "always_recommended": [
                            "content",
                            "agent_id",
                            "context",
                            "type",
                            "tier",
                            "visibility",
                            "writer_model",
                            "context_fingerprint",
                            "importance",
                        ],
                        "importance_fields": [
                            "confidence|predictive_confidence|predictive_confidence_before OR proxy_disagreement|disagreement_score OR self_rating|surprise_self_rating",
                            "tool_steps/correction_count/inference_level",
                            "negative_impact (0..1) se evento critico",
                            "is_external true solo se dato esterno reale",
                        ],
                        "context_fingerprint_fields": [
                            "conversation_id",
                            "task_id",
                            "retrieved_ids (ordinati e deduplicati dal server)",
                            "tool_trace_fingerprint",
                            "prompt_fingerprint",
                        ],
                    },
                }
            )

        if name == "memory.list_projects":
            projects = memory_service.list_projects(actor)
            return _json_text(
                {
                    "api_version": "v2",
                    "count": len(projects),
                    "projects": [project.model_dump(mode="json") for project in projects],
                }
            )

        if name == "memory.get_project_info":
            project = memory_service.get_project_info(actor, arguments["project_id"])
            return _json_text(
                {
                    "api_version": "v2",
                    "project": project.model_dump(mode="json") if project else None,
                }
            )

        if name == "memory.create_project":
            project = memory_service.create_project(
                actor=actor,
                project_id=arguments["project_id"],
                display_name=arguments.get("display_name"),
                description=arguments.get("description", ""),
                metadata=arguments.get("metadata"),
            )
            return _json_text(
                {
                    "api_version": "v2",
                    "project": project.model_dump(mode="json"),
                }
            )

        if name == "memory.add":
            try:
                payload = await memory_service.add(arguments, actor, write_path="add")
            except MemoryInputError as exc:
                raise ValueError(exc.to_json()) from exc
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

        raise ValueError(f"Unknown tool: {name}")
