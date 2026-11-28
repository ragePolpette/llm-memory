"""The seven MCP tools of llm-memory v2.

    memory.note       raw episode: what happened, cheap, expires unless a memory cites it
    memory.save       structured memory: decision, lesson, fact, constraint, preference
    memory.supersede  close a memory that is no longer true
    memory.recall     hybrid search over memories and episodes, within a character budget
    memory.get        one memory or episode by id
    memory.list       latest notes of a kind, or memories of a type, without a query
    memory.projects   list or create projects

Results are JSON. Errors come back as {"ok": false, "error": {code, message, details}}
with isError set, so the agent can read what to change and retry.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from mcp.server import Server
from mcp.types import CallToolResult, TextContent, Tool

from src.store_v2.service import Caller, MemoryServiceV2, ServiceError
from src.store_v2.store import MEMORY_TYPES

_PROJECT = {"type": "string", "description": "Project id, e.g. 'yesod'. '__global__' holds rules valid everywhere."}
_AGENT = {"type": "string", "description": "Name of the calling agent or harness, e.g. 'claude-code'."}
_TAGS = {"type": "array", "items": {"type": "string"}, "maxItems": 20}

TOOLS: list[Tool] = [
    Tool(
        name="memory.note",
        description=(
            "Record something that happened, as it happened: an error, a test result, an observation, a hint of a "
            "decision. Cheap and unfiltered; notes expire after about 60 days unless a saved memory cites them, so "
            "noting too much costs nothing durable. Never include secrets (keys, tokens, passwords): the write is "
            "rejected. Use memory.save instead for something that should guide future work."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": _PROJECT,
                "content": {"type": "string", "description": "What happened, in plain words (max 8000 chars)."},
                "event_type": {"type": "string", "description": "observation (default), error, test_run, "
                               "decision_hint, user_feedback, ..."},
                "context": {"type": "string", "description": "Where it happened: file, command, ticket."},
                "tags": _TAGS,
                "session_id": {"type": "string"},
                "metadata": {"type": "object", "description": "Structured fields for tools that read the note back."},
                "ttl_days": {"type": "integer", "minimum": 0, "maximum": 3650,
                             "description": "Days before the note expires (default 60). 0 keeps it for good: use it "
                             "only for records that must outlive the project's working memory."},
                "agent_id": _AGENT,
            },
            "required": ["project_id", "content"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory.save",
        description=(
            "Save a memory that should guide future work on the project. type: decision (a choice made), lesson "
            "(something learned the hard way), fact (how things are), constraint (a hard limit), preference (how "
            "the user wants things). statement is one self-contained sentence. why is required for decisions and "
            "lessons. Cite evidence (episode or memory ids, or external refs like a PR URL) when you have it. If "
            "this replaces an older memory, pass its id in supersedes. An identical active memory is not duplicated. "
            "Never include secrets: describe where a secret lives, not its value."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": _PROJECT,
                "type": {"type": "string", "enum": list(MEMORY_TYPES)},
                "statement": {"type": "string", "description": "One self-contained sentence (max 1000 chars)."},
                "why": {"type": "string", "description": "Reason or cause. Required for decision and lesson."},
                "how": {"type": "string", "description": "How to apply it, or how it was found."},
                "impact": {"type": "string", "enum": ["low", "medium", "high"]},
                "impact_note": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"kind": {"type": "string", "enum": ["episode", "memory", "external"]},
                                       "ref": {"type": "string"}},
                        "required": ["kind", "ref"],
                    },
                },
                "tags": _TAGS,
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "supersedes": {"type": "string", "description": "Id of the memory this one replaces."},
                "agent_id": _AGENT,
            },
            "required": ["project_id", "type", "statement"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory.supersede",
        description=(
            "Close a memory that is no longer true, with a reason. It stays readable as history "
            "(memory.recall with include_closed) but no longer appears in normal recall. To replace it with a new "
            "version, prefer memory.save with supersedes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "reason": {"type": "string"},
                "replacement_id": {"type": "string"},
                "agent_id": _AGENT,
            },
            "required": ["memory_id", "reason"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory.recall",
        description=(
            "Search the project's memories and notes (and global rules) by meaning and by exact terms. Ask a "
            "question or name the identifier you need (ticket, config key, error). Results are cut to budget_chars. "
            "Closed memories are hidden unless include_closed is true (useful for 'what did we use before?')."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": _PROJECT,
                "query": {"type": "string"},
                "kinds": {"type": "array", "items": {"type": "string", "enum": ["memory", "episode"]}},
                "include_global": {"type": "boolean", "default": True},
                "include_closed": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 8},
                "budget_chars": {"type": "integer", "minimum": 200, "default": 6000},
                "agent_id": _AGENT,
            },
            "required": ["project_id", "query"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory.get",
        description="Fetch one memory (with its sources and validity) or one episode by id.",
        inputSchema={
            "type": "object",
            "properties": {"id": {"type": "string"}, "agent_id": _AGENT},
            "required": ["id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory.list",
        description=(
            "List the latest notes of a project (kind=episode, optionally one event_type) or its memories "
            "(kind=memory, optionally one type), newest first, without a search query. Use memory.recall to search."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": _PROJECT,
                "kind": {"type": "string", "enum": ["episode", "memory"], "default": "episode"},
                "event_type": {"type": "string"},
                "type": {"type": "string", "enum": list(MEMORY_TYPES)},
                "since": {"type": "string", "description": "ISO 8601 timestamp, inclusive."},
                "include_closed": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
                "agent_id": _AGENT,
            },
            "required": ["project_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory.projects",
        description="List projects, or create one (action=create). Writes to an unknown project are rejected.",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "create"], "default": "list"},
                "project_id": {"type": "string", "description": "Lowercase, 2-64 chars: letters, digits, . _ -"},
                "display_name": {"type": "string"},
                "description": {"type": "string"},
                "agent_id": _AGENT,
            },
            "additionalProperties": False,
        },
    ),
]

_TOOL_NAMES = {tool.name for tool in TOOLS}


def _dispatch(service: MemoryServiceV2) -> dict[str, Callable[..., dict]]:
    return {
        "memory.note": service.note,
        "memory.save": service.save,
        "memory.supersede": service.supersede,
        "memory.recall": service.recall,
        "memory.get": service.get,
        "memory.list": service.list,
        "memory.projects": service.projects,
    }


def _result(payload: dict, *, error: bool = False) -> CallToolResult:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return CallToolResult(content=[TextContent(type="text", text=text)], structuredContent=payload, isError=error)


def call(service: MemoryServiceV2, name: str, arguments: dict[str, Any] | None) -> CallToolResult:
    """Run one tool call. Separate from the MCP server so it can be tested directly."""

    arguments = dict(arguments or {})
    if name not in _TOOL_NAMES:
        return _result({"ok": False, "error": {"code": "unknown_tool", "message": f"unknown tool {name!r}",
                                               "details": {"tools": sorted(_TOOL_NAMES)}}}, error=True)
    agent = str(arguments.pop("agent_id", "") or "agent").strip()[:64] or "agent"
    caller = Caller(actor=agent)
    try:
        payload = _dispatch(service)[name](caller, **arguments)
    except ServiceError as exc:
        return _result({"ok": False, "error": exc.to_dict()}, error=True)
    except TypeError as exc:  # unexpected or missing argument
        return _result({"ok": False, "error": {"code": "invalid_argument", "message": str(exc)}}, error=True)
    return _result({"ok": True, **payload})


def register_tools_v2(server: Server, service: MemoryServiceV2) -> None:
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict) -> CallToolResult:
        return call(service, name, arguments)
