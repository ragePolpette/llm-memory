"""Operations the MCP tools expose, on top of the v2 store.

Six operations: note (episode), save (structured memory), supersede, recall, get,
projects. Every write is validated, scanned for secrets and audited in the same
transaction as the data. A rejected write is audited too, without its content.

Audit actions and their payloads (what /admin/audit consumers such as Yesod read):

    note        {project_id, outcome: created|rejected, event_type, repeats}
    save        {project_id, outcome: created|duplicate|rejected, type, supersedes, sources}
    supersede   {project_id, replacement_id}            reason = the caller's reason
    create_project {project_id}
    expire_episodes {count}

Every payload of a write carries project_id at the top level. Rejections add
{reason_code, fields, kinds}; values never reach the audit log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from .retrieval import HybridRetriever
from .schema import GLOBAL_PROJECT
from .secrets import scan_fields
from .store import (
    MEMORY_TYPES,
    AuditRecord,
    ConflictError,
    Episode,
    Memory,
    MemoryStore,
    NotFoundError,
    StoreError,
)

EVENT_TYPE = re.compile(r"^[a-z][a-z0-9_.-]{1,39}$")
TAG = re.compile(r"^[\w.:/-]{1,64}$", re.UNICODE)
IMPACTS = ("low", "medium", "high")
WHY_REQUIRED = ("decision", "lesson")
SOURCE_KINDS = ("episode", "memory", "external")

LIMITS = {"statement": 1000, "content": 8000, "why": 2000, "how": 2000, "context": 4000, "impact_note": 1000,
          "reason": 500, "query": 2000, "display_name": 200, "description": 2000}


class ServiceError(Exception):
    """An error the caller can act on. `code` is stable, `message` says what to change."""

    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, **({"details": self.details} if self.details else {})}


@dataclass(frozen=True)
class Caller:
    """Who is writing. Phase 5 derives this from the client's token instead of the arguments."""

    actor: str
    source: str = "agent"  # agent | human | distilled


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _text(name: str, value: Any, *, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ServiceError("invalid_argument", f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise ServiceError("invalid_argument", f"{name} is required")
    limit = LIMITS.get(name)
    if limit and len(value) > limit:
        raise ServiceError("invalid_argument", f"{name} is longer than {limit} characters ({len(value)}): "
                           "store the gist and keep the detail in an episode or an external reference")
    return value


def _tags(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
        raise ServiceError("invalid_argument", "tags must be a list of strings")
    tags = list(dict.fromkeys(t.strip() for t in value if t.strip()))
    bad = [t for t in tags if not TAG.match(t)]
    if bad:
        raise ServiceError("invalid_argument", "tags may contain letters, digits and . _ : / - (max 64 chars)",
                           invalid=bad[:5])
    if len(tags) > 20:
        raise ServiceError("invalid_argument", "at most 20 tags")
    return tags


class MemoryServiceV2:
    def __init__(self, store: MemoryStore, retriever: HybridRetriever, *, episode_ttl_days: int = 60):
        self.store = store
        self.retriever = retriever
        self.episode_ttl_days = episode_ttl_days

    # -- helpers ------------------------------------------------------------------

    def _project(self, project_id: Any, *, allow_global: bool = True) -> str:
        if not isinstance(project_id, str) or not project_id.strip():
            raise ServiceError("invalid_argument", "project_id is required")
        try:
            project_id = self.store.validate_project_id(project_id.strip())
        except StoreError as exc:
            raise ServiceError("invalid_argument", str(exc)) from None
        if project_id == GLOBAL_PROJECT and not allow_global:
            raise ServiceError("invalid_argument", "this operation does not accept the global project")
        if self.store.get_project(project_id) is None:
            raise ServiceError("not_found", f"unknown project {project_id!r}: create it with memory.projects "
                               "(action=create) first", project_id=project_id)
        return project_id

    def _reject_secrets(self, action: str, project_id: str, caller: Caller, fields: Iterable[tuple[str, Any]]) -> None:
        findings = scan_fields(fields)
        if not findings:
            return
        names = sorted({f.field for f in findings})
        kinds = sorted({f.kind for f in findings})
        self.store.add_audit(AuditRecord(
            action=action, actor=caller.actor, reason="secret_detected",
            payload={"project_id": project_id, "outcome": "rejected", "reason_code": "secret_detected",
                     "fields": names, "kinds": kinds},
        ))
        raise ServiceError(
            "secret_rejected",
            "the write contains what looks like a secret and was not stored. Remove the value (describe where it "
            "lives instead, e.g. 'the key is in the FIREWORKS_API_KEY variable') and write again.",
            fields=names, kinds=kinds,
        )

    # -- writes -------------------------------------------------------------------

    def note(self, caller: Caller, *, project_id: str, content: str, event_type: str = "observation",
             context: str = "", tags: Optional[list] = None, session_id: Optional[str] = None,
             metadata: Optional[dict] = None) -> dict:
        project_id = self._project(project_id)
        content = _text("content", content, required=True)
        context = _text("context", context)
        event_type = _text("event_type", event_type or "observation")
        if not EVENT_TYPE.match(event_type):
            raise ServiceError("invalid_argument", "event_type must be a short lowercase word, e.g. observation, "
                               "error, test_run, decision_hint")
        tags = _tags(tags)
        if metadata is not None and not isinstance(metadata, dict):
            raise ServiceError("invalid_argument", "metadata must be an object")
        self._reject_secrets("note", project_id, caller, [("content", content), ("context", context), ("tags", tags),
                                                          ("metadata", metadata or {})])
        repeats = self.store.count_similar_episodes(project_id, event_type, content)
        expires_at = (_now() + timedelta(days=self.episode_ttl_days)).isoformat()
        episode = Episode(project_id=project_id, event_type=event_type, content=content, agent_id=caller.actor,
                          context=context, tags=tags, metadata=metadata or {}, session_id=session_id,
                          source=caller.source, expires_at=expires_at)
        episode_id = self.store.add_episode(episode, audit=AuditRecord(
            action="note", actor=caller.actor,
            payload={"project_id": project_id, "outcome": "created", "event_type": event_type, "repeats": repeats},
        ))
        return {"id": episode_id, "kind": "episode", "project_id": project_id, "expires_at": expires_at,
                "seen_before": repeats}

    def save(self, caller: Caller, *, project_id: str, type: str, statement: str, why: str = "", how: str = "",
             impact: Optional[str] = None, impact_note: str = "", evidence: Optional[list] = None,
             tags: Optional[list] = None, confidence: Optional[float] = None,
             supersedes: Optional[str] = None) -> dict:
        project_id = self._project(project_id)
        if type not in MEMORY_TYPES:
            raise ServiceError("invalid_argument", f"type must be one of {', '.join(MEMORY_TYPES)}")
        statement = _text("statement", statement, required=True)
        why = _text("why", why, required=type in WHY_REQUIRED)
        how = _text("how", how)
        impact_note = _text("impact_note", impact_note)
        if impact is not None and impact not in IMPACTS:
            raise ServiceError("invalid_argument", "impact must be low, medium or high")
        tags = _tags(tags)
        if confidence is None:
            confidence = 0.5
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ServiceError("invalid_argument", "confidence must be a number between 0 and 1")
        sources: list[tuple[str, str]] = []
        for index, item in enumerate(evidence or []):
            if not isinstance(item, dict) or item.get("kind") not in SOURCE_KINDS or not str(item.get("ref", "")).strip():
                raise ServiceError("invalid_argument", f"evidence[{index}] must be {{kind: episode|memory|external, "
                                   "ref: id or reference}")
            sources.append((item["kind"], str(item["ref"]).strip()))
        if caller.source == "distilled" and not sources:
            raise ServiceError("invalid_argument", "a distilled memory must cite the episodes or memories it comes from")
        self._reject_secrets("save", project_id, caller, [
            ("statement", statement), ("why", why), ("how", how), ("impact_note", impact_note), ("tags", tags),
            ("evidence", [ref for kind, ref in sources if kind == "external"]),
        ])
        memory = Memory(project_id=project_id, type=type, statement=statement, actor=caller.actor, source=caller.source,
                        why=why, how=how, impact=impact, impact_note=impact_note, tags=tags,
                        confidence=float(confidence))
        try:
            memory_id, created = self.store.add_memory(memory, sources=sources, supersedes=supersedes, audit=AuditRecord(
                action="save", actor=caller.actor,
                payload={"project_id": project_id, "outcome": "created", "type": type, "supersedes": supersedes,
                         "sources": len(sources)},
            ))
        except NotFoundError as exc:
            raise ServiceError("not_found", str(exc)) from None
        except ConflictError as exc:
            raise ServiceError("conflict", str(exc)) from None
        except StoreError as exc:
            raise ServiceError("invalid_argument", str(exc)) from None
        if not created:
            self.store.add_audit(AuditRecord(
                action="save", actor=caller.actor, entry_id=memory_id,
                payload={"project_id": project_id, "outcome": "duplicate", "type": type, "supersedes": supersedes,
                         "sources": len(sources)},
            ))
        return {"id": memory_id, "kind": "memory", "project_id": project_id, "created": created,
                "superseded": supersedes if created else None}

    def supersede(self, caller: Caller, *, memory_id: str, reason: str, replacement_id: Optional[str] = None) -> dict:
        reason = _text("reason", reason, required=True)
        memory = self.store.get_memory(_text("memory_id", memory_id, required=True))
        if memory is None:
            raise ServiceError("not_found", f"unknown memory: {memory_id}")
        self._reject_secrets("supersede", memory["project_id"], caller, [("reason", reason)])
        try:
            self.store.supersede(memory["id"], replacement_id=replacement_id, audit=AuditRecord(
                action="supersede", actor=caller.actor, reason=reason,
                payload={"project_id": memory["project_id"], "replacement_id": replacement_id},
            ))
        except ConflictError as exc:
            raise ServiceError("conflict", str(exc)) from None
        except NotFoundError as exc:
            raise ServiceError("not_found", str(exc)) from None
        except StoreError as exc:
            raise ServiceError("invalid_argument", str(exc)) from None
        return {"id": memory["id"], "project_id": memory["project_id"], "closed": True,
                "replacement_id": replacement_id}

    # -- reads --------------------------------------------------------------------

    def recall(self, caller: Caller, *, query: str, project_id: str, include_global: bool = True,
               kinds: Optional[list] = None, include_closed: bool = False, limit: int = 8,
               budget_chars: Optional[int] = 6000) -> dict:
        project_id = self._project(project_id)
        query = _text("query", query, required=True)
        kinds = kinds or ["memory", "episode"]
        if not isinstance(kinds, list) or not set(kinds) <= {"memory", "episode"} or not kinds:
            raise ServiceError("invalid_argument", "kinds must be a non-empty subset of [memory, episode]")
        if not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ServiceError("invalid_argument", "limit must be an integer between 1 and 50")
        if budget_chars is not None and (not isinstance(budget_chars, int) or budget_chars < 200):
            raise ServiceError("invalid_argument", "budget_chars must be an integer of at least 200")
        result = self.retriever.recall(query, [project_id], include_global=include_global, kinds=kinds,
                                       include_closed=include_closed, limit=limit, budget_chars=budget_chars)
        self.store.touch_memories(item.id for item in result.items if item.kind == "memory")
        items = []
        for item in result.items:
            entry = {"id": item.id, "kind": item.kind, "project_id": item.project_id, "score": item.score,
                     "matched_by": item.matched_by, "text": item.text}
            if item.kind == "memory":
                entry.update({"type": item.item["type"], "impact": item.item["impact"],
                              "confidence": item.item["confidence"], "created_at": item.item["created_at"],
                              "valid_to": item.item["valid_to"], "superseded_by": item.item["superseded_by"]})
            else:
                entry.update({"event_type": item.item["event_type"], "created_at": item.item["created_at"]})
            items.append(entry)
        return {"items": items, "mode": result.mode, "degraded": result.degraded, "truncated": result.truncated,
                "used_chars": result.used_chars}

    def get(self, caller: Caller, *, id: str) -> dict:
        item_id = _text("id", id, required=True)
        item = self.store.get_memory(item_id) or self.store.get_episode(item_id)
        if item is None:
            raise ServiceError("not_found", f"no memory or episode with id {item_id}")
        return {"item": item}

    # -- projects -----------------------------------------------------------------

    def projects(self, caller: Caller, *, action: str = "list", project_id: Optional[str] = None,
                 display_name: str = "", description: str = "") -> dict:
        if action == "list":
            return {"projects": self.store.list_projects()}
        if action != "create":
            raise ServiceError("invalid_argument", "action must be list or create")
        if not isinstance(project_id, str):
            raise ServiceError("invalid_argument", "project_id is required")
        try:
            project_id = self.store.validate_project_id(project_id.strip())
        except StoreError as exc:
            raise ServiceError("invalid_argument", str(exc)) from None
        display_name = _text("display_name", display_name)
        description = _text("description", description)
        self._reject_secrets("create_project", project_id, caller,
                             [("display_name", display_name), ("description", description)])
        created = self.store.ensure_project(project_id, display_name=display_name, description=description,
                                            actor=caller.actor)
        return {"project": self.store.get_project(project_id), "created": created}

    # -- maintenance --------------------------------------------------------------

    def expire_episodes(self) -> int:
        return self.store.expire_episodes(actor="system")
