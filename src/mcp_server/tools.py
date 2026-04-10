"""Definizione tools MCP v2."""

from __future__ import annotations

from pathlib import Path

from mcp.server import Server
from mcp.types import Tool, TextContent

from ..config import MemoryScope, Tier
from ..models import EntryType
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


def _require_explicit_project_scope(name: str, arguments: dict, service: MemoryService) -> None:
    if not service.config.multi_project_enabled:
        return
    if name not in {
        "memory.add",
        "memory.log_fast",
        "memory.list_fast",
        "memory.get_fast",
        "memory.summarize_fast",
        "memory.discard_fast",
        "memory.promote_fast",
        "memory.search",
        "memory.get",
        "memory.invalidate",
        "memory.promote",
        "memory.reembed",
        "memory.export",
        "memory.import",
    }:
        return
    raw_scope = arguments.get("scope")
    scope = raw_scope if isinstance(raw_scope, dict) else {}
    scope_level = str(
        scope.get("scope_level") or scope.get("level") or arguments.get("scope_level") or "project"
    ).strip().lower()
    if scope_level in {"workspace", "global"}:
        return
    project_id = str(scope.get("project_id") or "").strip()
    if not project_id:
        raise ValueError(
            f"{name} requires explicit scope.project_id for project-scoped operations when "
            "MEMORY_MULTI_PROJECT_ENABLED=true. Pass scope.project_id explicitly or choose "
            "scope.scope_level=workspace|global."
        )


def _normalize_inference_scope(arguments: dict, service: MemoryService) -> tuple[str, dict]:
    raw_scope = str(arguments.get("scope", "project")).strip().lower()
    if raw_scope not in {"project", "global"}:
        raise ValueError("scope must be 'project' or 'global'")
    scope_payload = arguments.get("scope_ref")
    scope_payload = scope_payload if isinstance(scope_payload, dict) else {}
    project_id = str(arguments.get("project_id") or scope_payload.get("project_id") or "").strip()
    if raw_scope == "project" and service.config.multi_project_enabled and not project_id:
        raise ValueError(
            "capture/search inference memory with scope=project requires project_id when "
            "MEMORY_MULTI_PROJECT_ENABLED=true."
        )
    normalized_scope = {"scope_level": raw_scope}
    if project_id:
        normalized_scope["project_id"] = project_id
    if scope_payload.get("workspace_id"):
        normalized_scope["workspace_id"] = str(scope_payload["workspace_id"])
    if scope_payload.get("user_id"):
        normalized_scope["user_id"] = str(scope_payload["user_id"])
    if scope_payload.get("agent_id"):
        normalized_scope["agent_id"] = str(scope_payload["agent_id"])
    return raw_scope, normalized_scope


def _build_inference_capture_payload(arguments: dict, actor: ActorContext, service: MemoryService) -> dict:
    scope_mode, scope = _normalize_inference_scope(arguments, service)
    namespace = str(arguments["namespace"]).strip()
    phase = str(arguments["phase"]).strip().lower()
    product_target = str(arguments["product_target"]).strip()
    repo_target = str(arguments["repo_target"]).strip()
    tags = [str(tag).strip() for tag in arguments.get("tags", []) if str(tag).strip()]
    metadata = dict(arguments.get("metadata") or {})
    metadata.update(
        {
            "kind": "inference_memory",
            "namespace": namespace,
            "phase": phase,
            "ticket_key": str(arguments.get("ticket_key", "")).strip() or None,
            "product_target": product_target,
            "repo_target": repo_target,
            "scope_mode": scope_mode,
        }
    )
    visibility = MemoryScope.GLOBAL.value if scope_mode == "global" else MemoryScope.SHARED.value
    return {
        "content": arguments["content"],
        "context": arguments.get("context")
        or (
            f"Il progetto {product_target} richiede di conservare questa inferenza tecnica riusabile "
            f"per la fase {phase} nel repo {repo_target}."
        ),
        "agent_id": actor.agent_id,
        "user_id": actor.user_id,
        "scope": scope,
        "visibility": visibility,
        "type": "fact",
        "tier": "tier-1",
        "source": "harness_inference",
        "confidence": float(arguments.get("confidence", 0.7)),
        "tags": sorted(set(tags + [namespace, phase, product_target, repo_target, "inference-memory"])),
        "metadata": metadata,
    }


def _build_inference_search_plan(arguments: dict, service: MemoryService) -> tuple[str, dict]:
    scope_mode, scope = _normalize_inference_scope(arguments, service)
    product_target = str(arguments.get("product_target", "")).strip()
    query = str(arguments["query"]).strip()
    limit = int(arguments.get("limit", 10))
    search_args = {
        "query": query,
        "limit": limit,
        "include_invalidated": False,
        "tier": None,
        "include_project": scope_mode == "project",
        "include_workspace": False,
        "include_global": scope_mode == "global",
    }
    filters = {
        "namespace": str(arguments["namespace"]).strip(),
        "product_target": product_target,
        "scope_mode": scope_mode,
    }
    return scope_mode, {"search_args": search_args, "filters": filters}


def _actor_with_scope(actor: ActorContext, normalized_scope: dict) -> ActorContext:
    return ActorContext(
        agent_id=str(normalized_scope.get("agent_id") or actor.agent_id),
        user_id=normalized_scope.get("user_id", actor.user_id),
        workspace_id=str(normalized_scope.get("workspace_id") or actor.workspace_id),
        project_id=str(normalized_scope.get("project_id") or actor.project_id),
        writer_model=actor.writer_model,
        writer_model_source=actor.writer_model_source,
    )


def _map_inference_bundles(
    bundles: list, actor: ActorContext, memory_service: MemoryService, filters: dict
) -> list[dict]:
    items: list[dict] = []
    namespace = filters["namespace"]
    product_target = filters["product_target"]
    for bundle in bundles:
        entry = memory_service.get(bundle.entry_id, actor)
        if entry is None:
            continue
        metadata = dict(entry.metadata or {})
        if metadata.get("kind") != "inference_memory":
            continue
        if metadata.get("namespace") != namespace:
            continue
        if product_target and metadata.get("product_target") != product_target:
            continue
        items.append(
            {
                "memory_id": entry.id,
                "content": entry.content,
                "tags": entry.tags,
                "confidence": entry.confidence,
                "created_at": entry.created_at,
                "updated_at": entry.updated_at,
                "product_target": metadata.get("product_target"),
                "repo_target": metadata.get("repo_target"),
                "phase": metadata.get("phase"),
                "ticket_key": metadata.get("ticket_key"),
                "scope": metadata.get("scope_mode"),
            }
        )
    return items


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
                    "di chat o retrieval di codice repository. Espone il modello corrente di "
                    "single-project mode / multi-project mode e la scope hierarchy "
                    "project/workspace/global."
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
                description=(
                    "Crea esplicitamente un progetto nel workspace corrente se non esiste. In "
                    "multi-project mode e' il percorso amministrativo raccomandato prima delle "
                    "scritture project-scoped."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": (
                                "Identificatore stabile del progetto da creare o recuperare nel "
                                "catalogo progetti del workspace corrente."
                            ),
                        },
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
                name="memory.scope_overview",
                description=(
                    "Mostra i bucket operativi e i conteggi correnti per la scope hierarchy "
                    "project/workspace/global nel workspace attivo."
                ),
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
                name="memory.add",
                description=(
                    "Aggiunge memoria operativa persistente tiered. Usare per decisioni/fatti/regole riusabili, "
                    "non per dump di contesto. Compilare sempre writer_model + context_fingerprint + importance "
                    "quando enforcement self-eval e attivo. scope.scope_level definisce il bucket "
                    "intenzionale (project/workspace/global); in multi-project mode le scritture "
                    "project-scoped richiedono scope.project_id esplicito."
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
                            "description": (
                                "Scope esplicito della memoria. Usare scope_level per scegliere il "
                                "bucket target nella scope hierarchy."
                            ),
                            "properties": {
                                "scope_level": {
                                    "type": "string",
                                    "enum": ["project", "workspace", "global"],
                                    "description": (
                                        "Livello di scope intenzionale per la scrittura: project, "
                                        "workspace o global."
                                    ),
                                },
                                "workspace_id": {"type": "string"},
                                "project_id": {
                                    "type": "string",
                                    "description": (
                                        "Richiesto per scritture project-scoped quando "
                                        "MEMORY_MULTI_PROJECT_ENABLED=true."
                                    ),
                                },
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
                description=(
                    "Ricerca semantica su memorie operative persistenti; non destinato a contesto "
                    "codice repository. Supporta composizione esplicita degli scope "
                    "project/workspace/global tramite include_project/include_workspace/include_global."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {
                            "type": "object",
                            "description": (
                                "Scope base della ricerca. Se lo scope_level resta project e il "
                                "server e' in multi-project mode, usare scope.project_id esplicito."
                            ),
                            "properties": {
                                "scope_level": {
                                    "type": "string",
                                    "enum": ["project", "workspace", "global"],
                                    "description": (
                                        "Scope base della ricerca. Le flag include_* possono "
                                        "ampliare la composizione verso altri bucket."
                                    ),
                                },
                                "workspace_id": {"type": "string"},
                                "project_id": {
                                    "type": "string",
                                    "description": (
                                        "Project scope di partenza. Richiesto per ricerche "
                                        "project-scoped quando MEMORY_MULTI_PROJECT_ENABLED=true."
                                    ),
                                },
                                "user_id": {"type": "string"},
                                "agent_id": {"type": "string"},
                            },
                        },
                        "limit": {"type": "integer", "default": 10},
                        "include_invalidated": {"type": "boolean", "default": False},
                        "tier": {"type": "string", "enum": ["tier-1", "tier-2", "tier-3"]},
                        "include_project": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include i bucket project-scoped nella composizione della ricerca.",
                        },
                        "include_workspace": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include i bucket workspace-scoped nella composizione della ricerca.",
                        },
                        "include_global": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include i bucket global/shared nella composizione della ricerca.",
                        },
                    },
                    "required": ["query", "agent_id"],
                },
            ),
            Tool(
                name="memory.log_fast",
                description=(
                    "Scrive nella fast memory episodica del progetto corrente. Usare per tentativi, "
                    "incidenti, fix temporanei e note operative che non devono entrare subito nella memoria forte."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "context": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "event_type": {"type": "string", "default": "note"},
                        "scope": {
                            "type": "object",
                            "properties": {
                                "scope_level": {"type": "string", "enum": ["project", "workspace", "global"]},
                                "workspace_id": {"type": "string"},
                                "project_id": {"type": "string"},
                                "user_id": {"type": "string"},
                                "agent_id": {"type": "string"},
                            },
                        },
                        "visibility": {
                            "type": "string",
                            "enum": ["private", "shared", "global"],
                            "default": "shared",
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "metadata": {"type": "object"},
                        "source": {"type": "string", "default": "mcp"},
                        "resolved": {"type": "boolean", "default": False},
                    },
                    "required": ["content", "agent_id"],
                },
            ),
            Tool(
                name="memory.list_fast",
                description=(
                    "Elenca le fast-memory entries del progetto corrente con filtri base su event_type, "
                    "resolved e distillation_status."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {
                            "type": "object",
                            "properties": {
                                "scope_level": {"type": "string", "enum": ["project", "workspace", "global"]},
                                "workspace_id": {"type": "string"},
                                "project_id": {"type": "string"},
                                "user_id": {"type": "string"},
                                "agent_id": {"type": "string"},
                            },
                        },
                        "limit": {"type": "integer", "default": 100},
                        "event_type": {"type": "string"},
                        "resolved": {"type": "boolean"},
                        "distillation_status": {
                            "type": "string",
                            "enum": ["pending", "summarized", "promoted", "discarded"],
                        },
                    },
                    "required": ["agent_id"],
                },
            ),
            Tool(
                name="memory.get_fast",
                description="Recupera una fast-memory entry per id nel perimetro di scope corrente.",
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
                name="memory.summarize_fast",
                description=(
                    "Segna una fast-memory entry come sintetizzata e salva un summary di distillazione "
                    "con reason, cluster opzionale e stato resolved opzionale."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entry_id": {"type": "string"},
                        "summary": {"type": "string"},
                        "reason": {"type": "string"},
                        "cluster_id": {"type": "string"},
                        "resolved": {"type": "boolean"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                    },
                    "required": ["entry_id", "summary", "reason", "agent_id"],
                },
            ),
            Tool(
                name="memory.discard_fast",
                description=(
                    "Marca una fast-memory entry come scartata dal processo di distillazione, "
                    "mantenendo audit trail e stato finale opzionale."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entry_id": {"type": "string"},
                        "reason": {"type": "string"},
                        "resolved": {"type": "boolean"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                    },
                    "required": ["entry_id", "reason", "agent_id"],
                },
            ),
            Tool(
                name="memory.promote_fast",
                description=(
                    "Promuove una fast-memory entry nella memoria forte creando una knowledge entry "
                    "ricercabile e marcando la sorgente come promoted."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entry_id": {"type": "string"},
                        "reason": {"type": "string"},
                        "target_tier": {
                            "type": "string",
                            "enum": ["tier-1", "tier-2", "tier-3"],
                            "default": "tier-3",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["fact", "assumption", "unknown", "decision"],
                            "default": "fact",
                        },
                        "visibility": {
                            "type": "string",
                            "enum": ["private", "shared", "global"],
                            "default": "shared",
                        },
                        "summary": {"type": "string"},
                        "confidence": {"type": "number", "default": 0.7},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "scope": {"type": "object"},
                    },
                    "required": ["entry_id", "reason", "agent_id"],
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
                name="capture_inference_memory",
                description=(
                    "Wrapper ergonomico per insight tecnici dell'harness. Salva una inferenza "
                    "persistente con scope pubblico semplificato project|global; workspace resta interno."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string"},
                        "phase": {"type": "string", "enum": ["triage", "execution"]},
                        "ticket_key": {"type": "string"},
                        "product_target": {"type": "string"},
                        "repo_target": {"type": "string"},
                        "content": {"type": "string"},
                        "context": {"type": "string"},
                        "confidence": {"type": "number"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "scope": {"type": "string", "enum": ["project", "global"], "default": "project"},
                        "project_id": {"type": "string"},
                        "scope_ref": {"type": "object"},
                        "metadata": {"type": "object"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                    },
                    "required": [
                        "namespace",
                        "phase",
                        "product_target",
                        "repo_target",
                        "content",
                        "agent_id",
                    ],
                },
            ),
            Tool(
                name="search_inference_memory",
                description=(
                    "Ricerca insight tecnici persistiti dall'harness. Espone solo scope project|global "
                    "e restituisce risultati pronti da consumare senza reinterpretare memory.search."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string"},
                        "query": {"type": "string"},
                        "product_target": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                        "scope": {"type": "string", "enum": ["project", "global"], "default": "project"},
                        "project_id": {"type": "string"},
                        "scope_ref": {"type": "object"},
                        "agent_id": {"type": "string"},
                        "user_id": {"type": "string"},
                    },
                    "required": ["namespace", "query", "agent_id"],
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
        _require_explicit_project_scope(name, arguments, memory_service)
        actor = _actor_from_args(arguments, memory_service)

        if name == "memory.about":
            return _json_text(
                {
                    "api_version": "v2",
                    "server": "llm-memory",
                    "self_eval_enforced": bool(memory_service.config.self_eval_enforced),
                    "purpose": "Memoria operativa persistente (decisioni, preferenze, regole, fatti di lavoro).",
                    "multi_project_enabled": bool(memory_service.config.multi_project_enabled),
                    "scope_hierarchy": ["project", "workspace", "global"],
                    "capabilities": [
                        "list_projects/get_project_info/create_project/scope_overview/add/log_fast/list_fast/get_fast/summarize_fast/discard_fast/promote_fast/search/get/invalidate/promote/reembed/export/import"
                    ],
                    "tool_map": {
                        "generic": {
                            "memory.about": "Guida server, boundary e scope hierarchy.",
                            "memory.add": "Scrittura generica di memoria persistente.",
                            "memory.log_fast": "Scrittura esplicita nella fast memory episodica.",
                            "memory.list_fast": "Elenco fast-memory entries del progetto corrente.",
                            "memory.get_fast": "Recupera una fast-memory entry per id.",
                            "memory.summarize_fast": "Segna una fast-memory entry come sintetizzata.",
                            "memory.discard_fast": "Scarta una fast-memory entry nel processo di distillazione.",
                            "memory.promote_fast": "Promuove una fast-memory entry nella memoria forte.",
                            "memory.search": "Ricerca generica sulle memorie persistenti.",
                        },
                        "harness": {
                            "capture_inference_memory": (
                                "Salva insight tecnici dell'harness con contract ridotto e scope pubblico project|global."
                            ),
                            "search_inference_memory": (
                                "Ricerca insight tecnici dell'harness e restituisce item pronti da consumare."
                            ),
                        },
                    },
                    "harness_scope_guide": {
                        "project": "Usa per conoscenza specifica di prodotto/repo corrente.",
                        "global": "Usa per conoscenza trasversale non project-specific.",
                        "workspace": "Interno al server; non esposto dal layer harness.",
                    },
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

        if name == "memory.scope_overview":
            return _json_text(
                {
                    "api_version": "v2",
                    "multi_project_enabled": bool(memory_service.config.multi_project_enabled),
                    "overview": memory_service.scope_overview(actor),
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
                include_project=bool(arguments.get("include_project", True)),
                include_workspace=bool(arguments.get("include_workspace", True)),
                include_global=bool(arguments.get("include_global", True)),
            )
            return _json_text({"api_version": "v2", "count": len(bundles), "bundles": [b.model_dump(mode="json") for b in bundles]})

        if name == "memory.log_fast":
            try:
                payload = memory_service.log_fast(arguments, actor, write_path="log_fast")
            except MemoryInputError as exc:
                raise ValueError(exc.to_json()) from exc
            payload["api_version"] = "v2"
            return _json_text(payload)

        if name == "memory.list_fast":
            entries = memory_service.list_fast(
                actor=actor,
                limit=int(arguments.get("limit", 100)),
                event_type=arguments.get("event_type"),
                resolved=arguments.get("resolved"),
                distillation_status=arguments.get("distillation_status"),
            )
            return _json_text(
                {
                    "api_version": "v2",
                    "count": len(entries),
                    "entries": [entry.model_dump(mode="json") for entry in entries],
                }
            )

        if name == "memory.get_fast":
            entry = memory_service.get_fast(arguments["entry_id"], actor)
            return _json_text({"api_version": "v2", "entry": entry.model_dump(mode="json") if entry else None})

        if name == "memory.summarize_fast":
            payload = memory_service.summarize_fast(
                entry_id=arguments["entry_id"],
                actor=actor,
                summary=arguments["summary"],
                reason=arguments["reason"],
                cluster_id=arguments.get("cluster_id"),
                resolved=arguments.get("resolved"),
            )
            payload["api_version"] = "v2"
            return _json_text(payload)

        if name == "memory.discard_fast":
            payload = memory_service.discard_fast(
                entry_id=arguments["entry_id"],
                actor=actor,
                reason=arguments["reason"],
                resolved=arguments.get("resolved"),
            )
            payload["api_version"] = "v2"
            return _json_text(payload)

        if name == "memory.promote_fast":
            payload = await memory_service.promote_fast(
                entry_id=arguments["entry_id"],
                actor=actor,
                reason=arguments["reason"],
                target_tier=Tier(arguments["target_tier"]) if arguments.get("target_tier") else None,
                memory_type=EntryType(arguments.get("memory_type", EntryType.FACT.value)),
                visibility=MemoryScope(arguments.get("visibility", MemoryScope.SHARED.value)),
                summary=arguments.get("summary"),
                confidence=float(arguments.get("confidence", 0.7)),
            )
            payload["api_version"] = "v2"
            return _json_text(payload)

        if name == "capture_inference_memory":
            payload = _build_inference_capture_payload(arguments, actor, memory_service)
            try:
                result = await memory_service.add(payload, actor, write_path="capture_inference_memory")
            except MemoryInputError as exc:
                raise ValueError(exc.to_json()) from exc
            response = {
                "api_version": "v2",
                "stored": bool(result.get("success")),
                "memory_id": result.get("entry_id"),
            }
            if not response["stored"]:
                decision = result.get("decision") or {}
                response["reason"] = decision.get("summary") or decision.get("category") or "rejected"
            return _json_text(response)

        if name == "search_inference_memory":
            _, normalized_scope = _normalize_inference_scope(arguments, memory_service)
            _, plan = _build_inference_search_plan(arguments, memory_service)
            search_actor = _actor_with_scope(actor, normalized_scope)
            bundles = await memory_service.search(actor=search_actor, **plan["search_args"])
            items = _map_inference_bundles(bundles, search_actor, memory_service, plan["filters"])
            return _json_text({"api_version": "v2", "count": len(items), "items": items})

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
