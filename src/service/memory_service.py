"""Servizio applicativo memoria Tier-1/2/3 con governance e audit."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import Config, MemoryScope, ScopeLevel, Tier
from ..embedding.embedding_service import EmbeddingProvider, get_reembed_provider
from ..models import (
    AuditEvent,
    EmbeddingVersion,
    EntryLink,
    EntryStatus,
    EntryType,
    ExportResult,
    ImportResult,
    MemoryBundle,
    MemoryEntry,
    ProjectRecord,
    ReembedResult,
    ScopeRef,
    compute_content_hash,
)
from ..security.crypto import PayloadCipher, PayloadDecryptionError
from ..security.privacy import PrivacyPolicy
from ..storage.sqlite_store import SQLiteMemoryStore
from ..vectordb.sqlite_vector_store import SQLiteVectorStore
from ..interop.memory_markdown import parse_memory_markdown, render_memory_markdown
from .importance_scoring import (
    build_importance_metadata,
    has_inference_signal,
    has_surprise_signal,
)
from .persistence_policy import (
    POLICY_VERSION,
    PersistenceDecision,
    classify_internal_write,
    classify_persistence,
)


@dataclass
class ActorContext:
    """Identità chiamante per policy least-privilege."""

    agent_id: str
    user_id: Optional[str]
    workspace_id: str
    project_id: str
    writer_model: Optional[str] = None
    writer_model_source: Optional[str] = None


class MemoryInputError(ValueError):
    """Structured validation error for memory.add payloads."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        missing_fields: Optional[list[str]] = None,
        retryable: bool = True,
        details: Optional[dict[str, Any]] = None,
    ):
        self.code = code
        self.message = message
        self.missing_fields = missing_fields or []
        self.retryable = retryable
        self.details = details or {}
        super().__init__(self.to_json())

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "missing_fields": self.missing_fields,
            "retryable": self.retryable,
            "details": self.details,
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "error_type": "memory_input_error",
                **self.to_payload(),
            },
            ensure_ascii=True,
            sort_keys=True,
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryService:
    """Servizio principale memoria persistente locale."""

    _JSONL_IMPORT_ALLOWED_FIELDS = frozenset(MemoryEntry.model_fields.keys())
    _PROJECT_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{1,63}$"
    _WORKSPACE_BUCKET_PROJECT_ID = "__workspace__"
    _GLOBAL_BUCKET_WORKSPACE_ID = "__global__"
    _GLOBAL_BUCKET_PROJECT_ID = "__global__"

    def __init__(
        self,
        config: Config,
        store: SQLiteMemoryStore,
        vector_store: SQLiteVectorStore,
        embedding_provider: EmbeddingProvider,
        privacy_policy: PrivacyPolicy,
        cipher: PayloadCipher,
    ):
        self.config = config
        self.store = store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.privacy_policy = privacy_policy
        self.cipher = cipher
        self._ensure_default_project_record()

    def _ensure_default_project_record(self) -> None:
        default_project = ProjectRecord(
            workspace_id=self.config.default_workspace_id,
            project_id=self.config.default_project_id,
            display_name=self.config.default_project_id,
            metadata={"source": "bootstrap-default"},
        )
        self.store.upsert_project(default_project)

    def _validate_project_identifier(self, project_id: str) -> str:
        normalized = str(project_id).strip().lower()
        if not normalized:
            raise ValueError("project_id must be a non-empty string")
        import re

        if not re.match(self._PROJECT_ID_PATTERN, normalized):
            raise ValueError(
                "project_id must match ^[a-z0-9][a-z0-9._-]{1,63}$"
            )
        return normalized

    def list_projects(self, actor: ActorContext) -> list[ProjectRecord]:
        return self.store.list_projects(actor.workspace_id)

    def scope_overview(self, actor: ActorContext) -> dict[str, Any]:
        project_scope = ScopeRef(
            workspace_id=actor.workspace_id,
            project_id=actor.project_id,
            scope_level=ScopeLevel.PROJECT,
        )
        workspace_scope = ScopeRef(
            workspace_id=actor.workspace_id,
            project_id=self._WORKSPACE_BUCKET_PROJECT_ID,
            scope_level=ScopeLevel.WORKSPACE,
        )
        global_scope = ScopeRef(
            workspace_id=self._GLOBAL_BUCKET_WORKSPACE_ID,
            project_id=self._GLOBAL_BUCKET_PROJECT_ID,
            scope_level=ScopeLevel.GLOBAL,
        )
        return {
            "project": {
                "workspace_id": actor.workspace_id,
                "project_id": actor.project_id,
                "count": self.store.count_entries_for_scope(project_scope),
            },
            "workspace": {
                "workspace_id": actor.workspace_id,
                "count": self.store.count_entries_for_scope(workspace_scope),
            },
            "global": {
                "count": self.store.count_entries_for_scope(global_scope),
            },
        }

    def get_project_info(self, actor: ActorContext, project_id: str) -> Optional[ProjectRecord]:
        normalized = self._validate_project_identifier(project_id)
        return self.store.get_project(actor.workspace_id, normalized)

    def create_project(
        self,
        *,
        actor: ActorContext,
        project_id: str,
        display_name: Optional[str] = None,
        description: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProjectRecord:
        normalized = self._validate_project_identifier(project_id)
        existing = self.store.get_project(actor.workspace_id, normalized)
        if existing is not None:
            return existing
        now = utc_now_iso()
        project = ProjectRecord(
            workspace_id=actor.workspace_id,
            project_id=normalized,
            display_name=(display_name or normalized).strip() or normalized,
            description=description.strip(),
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        self.store.upsert_project(project)
        self.store.add_audit(
            AuditEvent(
                action="create_project",
                actor=actor.agent_id,
                reason="explicit_create_project",
                payload={
                    "workspace_id": actor.workspace_id,
                    "project_id": normalized,
                    "display_name": project.display_name,
                },
            )
        )
        return project

    @staticmethod
    def _normalize_since_filter(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None
        try:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("since must be a valid ISO 8601 timestamp") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    def admin_summary(self) -> dict[str, Any]:
        active_version = self.store.get_active_embedding_version()
        latest_audit = next((audit for audit in self.store.list_audit(limit=1)), None)
        return {
            "storage": {
                "sqlite_db_path": str(self.config.sqlite_db_path),
                "import_export_base_dir": str(self.config.import_export_base_dir),
            },
            "settings": {
                "encryption_enabled": bool(self.config.encryption_enabled),
                "multi_project_enabled": bool(self.config.multi_project_enabled),
                "allow_outbound_network": bool(self.config.allow_outbound_network),
            },
            "counts": {
                "entries_total": self.store.count_entries(),
                "active_entries": self.store.count_entries(exclude_invalidated=True),
                "invalidated_entries": self.store.count_entries(status=EntryStatus.INVALIDATED),
                "invalidation_markers": self.store.count_entries(entry_type=EntryType.INVALIDATED),
                "projects_total": len(self.store.list_projects()),
                "audit_events_total": self.store.count_audit(),
            },
            "scopes": {
                "project": self.store.count_entries(
                    scope_level=ScopeLevel.PROJECT.value,
                    exclude_invalidated=True,
                ),
                "workspace": self.store.count_entries(
                    scope_level=ScopeLevel.WORKSPACE.value,
                    exclude_invalidated=True,
                ),
                "global": self.store.count_entries(
                    scope_level=ScopeLevel.GLOBAL.value,
                    exclude_invalidated=True,
                ),
            },
            "embedding": {
                "active_version": active_version.model_dump(mode="json") if active_version else None,
            },
            "latest_audit_at": (
                latest_audit.created_at.isoformat()
                if latest_audit is not None and isinstance(latest_audit.created_at, datetime)
                else (latest_audit.created_at if latest_audit is not None else None)
            ),
        }

    def admin_list_audit(
        self,
        *,
        limit: int = 100,
        entry_id: Optional[str] = None,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        reason: Optional[str] = None,
        since: Optional[str] = None,
    ) -> dict[str, Any]:
        clamped_limit = max(1, min(int(limit), 500))
        normalized_since = self._normalize_since_filter(since)
        audits = self.store.list_audit(
            entry_id=entry_id,
            limit=clamped_limit,
            action=action,
            actor=actor,
            reason=reason,
            since=normalized_since,
        )
        items = []
        for audit in audits:
            payload = audit.model_dump(mode="json")
            payload["payload_preview"] = self._preview_text(
                json.dumps(audit.payload, ensure_ascii=True, sort_keys=True),
                limit=240,
            )
            items.append(payload)
        return {
            "count": len(items),
            "limit": clamped_limit,
            "filters": {
                "entry_id": entry_id,
                "action": action,
                "actor": actor,
                "reason": reason,
                "since": normalized_since,
            },
            "items": items,
        }

    def admin_list_projects(
        self,
        *,
        workspace_id: Optional[str] = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        clamped_limit = max(1, min(int(limit), 500))
        projects = self.store.list_projects(workspace_id=workspace_id)[:clamped_limit]
        items = []
        for project in projects:
            items.append(
                {
                    **project.model_dump(mode="json"),
                    "entry_count": self.store.count_entries(
                        workspace_id=project.workspace_id,
                        project_id=project.project_id,
                        scope_level=ScopeLevel.PROJECT.value,
                    ),
                    "active_entry_count": self.store.count_entries(
                        workspace_id=project.workspace_id,
                        project_id=project.project_id,
                        scope_level=ScopeLevel.PROJECT.value,
                        exclude_invalidated=True,
                    ),
                }
            )
        return {
            "count": len(items),
            "limit": clamped_limit,
            "workspace_id": workspace_id,
            "items": items,
        }

    def _validate_write_payload(self, payload: dict[str, Any]) -> None:
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise MemoryInputError(
                code="INVALID_CONTENT",
                message="content must be a non-empty string.",
                missing_fields=["content"],
                retryable=False,
            )

        agent_id = payload.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise MemoryInputError(
                code="INVALID_AGENT_ID",
                message="agent_id must be a non-empty string.",
                missing_fields=["agent_id"],
                retryable=False,
            )

        for key in ("context",):
            value = payload.get(key)
            if value is not None and not isinstance(value, str):
                raise MemoryInputError(
                    code="INVALID_FIELD_TYPE",
                    message=f"{key} must be a string when provided.",
                    missing_fields=[key],
                    retryable=False,
                )

    @staticmethod
    def _preview_text(value: Any, *, limit: int = 240) -> Any:
        if isinstance(value, str):
            compact = " ".join(value.split())
            return compact[:limit]
        return value

    def _log_activity(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server": "llm-memory",
            "event": event,
            **payload,
        }
        sys.stderr.write(f"[MCP_ACTIVITY] {json.dumps(record, ensure_ascii=True, default=str)}\n")

    @staticmethod
    def _decision_payload(decision: PersistenceDecision, *, write_path: str) -> dict[str, Any]:
        return decision.as_payload(write_path=write_path)

    def _audit_write_attempt(
        self,
        *,
        actor: ActorContext,
        write_path: str,
        decision: PersistenceDecision,
        scope: ScopeRef,
        entry_id: Optional[str] = None,
        outcome: str,
        duplicate_of: Optional[str] = None,
    ) -> None:
        payload = self._decision_payload(decision, write_path=write_path)
        payload.update(
            {
                "outcome": outcome,
                "scope": scope.model_dump(),
            }
        )
        if duplicate_of:
            payload["duplicate_of"] = duplicate_of

        self.store.add_audit(
            AuditEvent(
                entry_id=entry_id,
                action="write_attempt",
                actor=actor.agent_id,
                reason=decision.decision,
                payload=payload,
            )
        )

    def _audit_novelty_failure(
        self,
        *,
        actor: ActorContext,
        write_path: str,
        embedding_status: str,
        vector_query_status: str,
        error_type: str,
        memory_id: Optional[str] = None,
    ) -> None:
        self.store.add_audit(
            AuditEvent(
                entry_id=memory_id,
                action="novelty_computation_failed",
                actor=actor.agent_id,
                reason="degraded",
                payload={
                    "event": "novelty_computation_failed",
                    "error_type": error_type,
                    "embedding_status": embedding_status,
                    "vector_query_status": vector_query_status,
                    "write_path": write_path,
                    "memory_id": memory_id,
                },
            )
        )

    def _persist_internal_entry(
        self,
        *,
        actor: ActorContext,
        write_path: str,
        record_type: str,
        internal_reason: str,
        entry: MemoryEntry,
    ) -> PersistenceDecision:
        decision = classify_internal_write(
            record_type=record_type,
            content=entry.content,
            internal_reason=internal_reason,
            write_path=write_path,
        )
        if not decision.accepted:
            self._audit_write_attempt(
                actor=actor,
                write_path=write_path,
                decision=decision,
                scope=entry.scope,
                entry_id=entry.id,
                outcome="rejected_internal",
            )
            raise ValueError(
                f"Internal write rejected for {record_type}: {', '.join(decision.reason_codes)}"
            )

        metadata = dict(entry.metadata)
        metadata.update(
            {
                "persistence_decision": self._decision_payload(decision, write_path=write_path),
                "internal_reason": internal_reason,
                "generated_by": actor.agent_id,
            }
        )
        entry.metadata = metadata
        entry.source = "internal_governance"

        self.store.add_entry(entry)
        self._audit_write_attempt(
            actor=actor,
            write_path=write_path,
            decision=decision,
            scope=entry.scope,
            entry_id=entry.id,
            outcome="accepted_internal",
        )
        return decision

    def _resolve_exchange_path(self, path: Path, *, must_exist: bool) -> Path:
        base_dir = self.config.import_export_base_dir.resolve()
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate

        resolved = candidate.resolve(strict=must_exist)
        if not resolved.is_relative_to(base_dir):
            raise ValueError(
                f"Path '{path}' escapes configured import/export base directory '{base_dir}'."
            )
        return resolved

    def _sanitize_import_metadata(self, metadata: Any) -> dict[str, Any]:
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            raise ValueError("JSONL import requires metadata to be an object.")
        return {
            key: value
            for key, value in metadata.items()
            if key.lower() not in self.privacy_policy.drop_metadata_keys
        }

    def _parse_jsonl_import_row(self, row: Any, *, line_number: int) -> MemoryEntry:
        if not isinstance(row, dict):
            raise ValueError(f"JSONL import row {line_number} must be an object.")

        unexpected_fields = sorted(set(row.keys()) - self._JSONL_IMPORT_ALLOWED_FIELDS)
        if unexpected_fields:
            raise ValueError(
                f"JSONL import row {line_number} contains unsupported fields: {', '.join(unexpected_fields)}"
            )

        sanitized_row = dict(row)
        sanitized_row["metadata"] = self._sanitize_import_metadata(sanitized_row.get("metadata"))
        return MemoryEntry(**sanitized_row)

    def default_scope(self, *, agent_id: Optional[str] = None, user_id: Optional[str] = None) -> ScopeRef:
        return ScopeRef(
            workspace_id=self.config.default_workspace_id,
            project_id=self.config.default_project_id,
            scope_level=ScopeLevel.PROJECT,
            user_id=user_id,
            agent_id=agent_id,
        )

    def _current_embedding_version(self) -> EmbeddingVersion:
        fingerprint = self.embedding_provider.fingerprint()
        version_id = (
            f"{self.embedding_provider.provider_id()}::"
            f"{self.embedding_provider.model_id()}::"
            f"{fingerprint[:16]}"
        )

        version = EmbeddingVersion(
            version_id=version_id,
            provider_id=self.embedding_provider.provider_id(),
            embedding_model_id=self.embedding_provider.model_id(),
            dim=self.embedding_provider.dimension(),
            fingerprint=fingerprint,
            config={
                "provider": self.embedding_provider.provider_id(),
                "model": self.embedding_provider.model_id(),
                "dim": self.embedding_provider.dimension(),
            },
            active=True,
        )

        self.store.upsert_embedding_version(version, activate=True)
        return version

    def _scope_from_payload(self, payload: dict, actor: ActorContext) -> ScopeRef:
        scope_payload = payload.get("scope") or {}
        raw_scope_level = (
            scope_payload.get("scope_level")
            or scope_payload.get("level")
            or payload.get("scope_level")
            or "project"
        )
        scope_level = ScopeLevel(str(raw_scope_level).strip().lower())
        scope = ScopeRef(
            workspace_id=scope_payload.get("workspace_id", actor.workspace_id),
            project_id=scope_payload.get("project_id", actor.project_id),
            scope_level=scope_level,
            user_id=scope_payload.get("user_id", actor.user_id),
            agent_id=scope_payload.get("agent_id", actor.agent_id),
        )
        return self._normalize_scope(scope)

    def _normalize_scope(self, scope: ScopeRef) -> ScopeRef:
        if scope.scope_level == ScopeLevel.PROJECT:
            normalized_project_id = self._validate_project_identifier(scope.project_id)
            return ScopeRef(
                workspace_id=scope.workspace_id,
                project_id=normalized_project_id,
                scope_level=ScopeLevel.PROJECT,
                user_id=scope.user_id,
                agent_id=scope.agent_id,
            )
        if scope.scope_level == ScopeLevel.WORKSPACE:
            return ScopeRef(
                workspace_id=scope.workspace_id,
                project_id=self._WORKSPACE_BUCKET_PROJECT_ID,
                scope_level=ScopeLevel.WORKSPACE,
                user_id=scope.user_id,
                agent_id=scope.agent_id,
            )
        return ScopeRef(
            workspace_id=self._GLOBAL_BUCKET_WORKSPACE_ID,
            project_id=self._GLOBAL_BUCKET_PROJECT_ID,
            scope_level=ScopeLevel.GLOBAL,
            user_id=scope.user_id,
            agent_id=scope.agent_id,
        )

    def _scope_exists(self, scope: ScopeRef) -> bool:
        if scope.scope_level != ScopeLevel.PROJECT:
            return True
        return self.store.get_project(scope.workspace_id, scope.project_id) is not None

    def _search_scopes(
        self,
        actor: ActorContext,
        *,
        include_project: bool,
        include_workspace: bool,
        include_global: bool,
    ) -> list[ScopeRef]:
        scopes: list[ScopeRef] = []
        if include_project:
            scopes.append(
                ScopeRef(
                    workspace_id=actor.workspace_id,
                    project_id=actor.project_id,
                    scope_level=ScopeLevel.PROJECT,
                )
            )
        if include_workspace:
            scopes.append(
                ScopeRef(
                    workspace_id=actor.workspace_id,
                    project_id=self._WORKSPACE_BUCKET_PROJECT_ID,
                    scope_level=ScopeLevel.WORKSPACE,
                )
            )
        if include_global:
            scopes.append(
                ScopeRef(
                    workspace_id=self._GLOBAL_BUCKET_WORKSPACE_ID,
                    project_id=self._GLOBAL_BUCKET_PROJECT_ID,
                    scope_level=ScopeLevel.GLOBAL,
                )
            )
        return scopes

    @staticmethod
    def _has_usable_embedding(vector: Any) -> bool:
        if not isinstance(vector, list) or not vector:
            return False
        return any(float(component) != 0.0 for component in vector)

    def _can_read(self, actor: ActorContext, entry: MemoryEntry) -> bool:
        if entry.scope.scope_level == ScopeLevel.PROJECT:
            if entry.scope.workspace_id != actor.workspace_id:
                return False
            if entry.scope.project_id != actor.project_id:
                return False
        elif entry.scope.scope_level == ScopeLevel.WORKSPACE:
            if entry.scope.workspace_id != actor.workspace_id:
                return False
        elif entry.scope.scope_level == ScopeLevel.GLOBAL:
            pass
        else:
            return False

        if entry.visibility == MemoryScope.PRIVATE:
            # Legacy private entries without an owning agent should not become
            # readable just because user/workspace/project happen to match.
            if not entry.scope.agent_id:
                return False
            if entry.scope.user_id and entry.scope.user_id != actor.user_id:
                return False
            if entry.scope.agent_id != actor.agent_id:
                return False
        return True

    def _can_write(self, actor: ActorContext, visibility: MemoryScope, scope: ScopeRef) -> bool:
        if scope.scope_level == ScopeLevel.GLOBAL:
            return visibility == MemoryScope.GLOBAL
        if scope.scope_level == ScopeLevel.WORKSPACE:
            if scope.workspace_id != actor.workspace_id:
                return False
            if visibility == MemoryScope.GLOBAL:
                return False
        else:
            if scope.workspace_id != actor.workspace_id:
                return False
            if not self._scope_exists(scope):
                return False
            if visibility == MemoryScope.GLOBAL:
                return False
        if visibility == MemoryScope.PRIVATE and scope.agent_id and scope.agent_id != actor.agent_id:
            return False
        return True

    def _snippet_for(self, entry: MemoryEntry, max_chars: int = 220) -> str:
        content = entry.content
        if entry.encrypted:
            try:
                content = self.cipher.decrypt(entry.content)
            except PayloadDecryptionError:
                content = "[ENCRYPTED]"
        return content[:max_chars]

    def _validate_self_eval_payload(
        self,
        payload: dict[str, Any],
        *,
        writer_model: str,
    ) -> None:
        if not self.config.self_eval_enforced:
            return

        missing_fields: list[str] = []
        context_fingerprint = payload.get("context_fingerprint")
        if not isinstance(context_fingerprint, dict):
            missing_fields.append("context_fingerprint")

        importance = payload.get("importance")
        if not isinstance(importance, dict):
            missing_fields.append("importance")

        if not writer_model or writer_model == "unknown-model":
            missing_fields.append("writer_model")

        if missing_fields:
            raise MemoryInputError(
                code="MISSING_REQUIRED_FIELDS",
                message="Self-evaluation requires writer_model, context_fingerprint, and importance payload.",
                missing_fields=missing_fields,
            )

        if not has_surprise_signal(payload):
            raise MemoryInputError(
                code="MISSING_SURPRISE_SIGNAL",
                message="importance requires at least one surprise signal.",
                missing_fields=[
                    "importance.confidence|predictive_confidence|predictive_confidence_before|proxy_disagreement|disagreement_score|self_rating|surprise_self_rating"
                ],
            )

        if not has_inference_signal(payload):
            raise MemoryInputError(
                code="MISSING_INFERENCE_SIGNAL",
                message="importance requires tool_steps, correction_count, inference_level or inference_steps.",
                missing_fields=[
                    "importance.tool_steps|importance.correction_count|importance.inference_level|importance.inference_steps|tool_steps|correction_count|inference_level"
                ],
            )

    async def add(self, payload: dict, actor: ActorContext, *, write_path: str = "add") -> dict:
        scope = self._scope_from_payload(payload, actor)
        self._log_activity(
            "write_in",
            {
                "operation": write_path,
                "agent_id": actor.agent_id,
                "workspace_id": scope.workspace_id,
                "project_id": scope.project_id,
                "content": self._preview_text(payload.get("content")),
                "type": payload.get("type", "fact"),
                "tier": payload.get("tier", "tier-1"),
                "visibility": payload.get("visibility", payload.get("scope_visibility", "shared")),
            },
        )
        try:
            self._validate_write_payload(payload)
        except MemoryInputError as exc:
            invalid_decision = PersistenceDecision(
                accepted=False,
                category="noise",
                reason_codes=[exc.code],
                confidence=1.0,
                normalized_summary=str(payload.get("content", "")).strip()[:180],
            )
            self._audit_write_attempt(
                actor=actor,
                write_path=write_path,
                decision=invalid_decision,
                scope=scope,
                outcome="invalid",
            )
            raise

        visibility = MemoryScope(payload.get("visibility", payload.get("scope_visibility", "shared")))
        if not self._can_write(actor, visibility, scope):
            raise PermissionError("Write denied by scope policy")

        decision = classify_persistence(payload, actor, write_path=write_path)
        if not decision.accepted:
            self._audit_write_attempt(
                actor=actor,
                write_path=write_path,
                decision=decision,
                scope=scope,
                outcome="rejected",
            )
            result = {
                "success": False,
                "rejected": True,
                "reason": "policy-rejected",
                "decision": self._decision_payload(decision, write_path=write_path),
            }
            self._log_activity(
                "write_out",
                {
                    "operation": write_path,
                    "agent_id": actor.agent_id,
                    "success": result["success"],
                    "rejected": result["rejected"],
                    "reason": result["reason"],
                },
            )
            return result

        entry_type = EntryType(payload.get("type", "fact"))
        tier = Tier(payload.get("tier", "tier-1"))
        links = [EntryLink(**item) for item in payload.get("links", [])]
        tags = payload.get("tags", [])
        sensitivity_tags = payload.get("sensitivity_tags", [])
        raw_content = payload["content"]
        content_hash = compute_content_hash(raw_content)

        if self.config.dedup_hash_enabled:
            duplicated = self.store.find_by_hash(scope=scope, content_hash=content_hash)
            if duplicated is not None:
                self._audit_write_attempt(
                    actor=actor,
                    write_path=write_path,
                    decision=decision,
                    scope=scope,
                    entry_id=duplicated.id,
                    outcome="duplicate",
                    duplicate_of=duplicated.id,
                )
                result = {
                    "success": True,
                    "entry_id": duplicated.id,
                    "duplicate_of": duplicated.id,
                    "reason": "hash-duplicate",
                    "decision": self._decision_payload(decision, write_path=write_path),
                }
                self._log_activity(
                    "write_out",
                    {
                        "operation": write_path,
                        "agent_id": actor.agent_id,
                        "success": result["success"],
                        "entry_id": result["entry_id"],
                        "duplicate_of": result["duplicate_of"],
                        "reason": result["reason"],
                    },
                )
                return result

        version = self._current_embedding_version()
        query_vec: Optional[list[float]] = None
        novelty_computed = False
        novelty_status = "failed"
        top_similarities: list[float] = []
        embedding_status = "not_started"
        vector_query_status = "not_started"

        try:
            query_vec = (await self.embedding_provider.embed([raw_content]))[0]
            if self._has_usable_embedding(query_vec):
                embedding_status = "ok"
                novelty_candidates = self.vector_store.search(
                    query_vector=query_vec,
                    scopes=[scope],
                    version_id=version.version_id,
                    limit=5,
                    include_invalidated=False,
                )
                top_similarities = [similarity for _, similarity in novelty_candidates]
                novelty_computed = True
                novelty_status = "computed"
                vector_query_status = "ok"
            else:
                embedding_status = "empty_vector"
                vector_query_status = "skipped"
                self._audit_novelty_failure(
                    actor=actor,
                    write_path=write_path,
                    embedding_status=embedding_status,
                    vector_query_status=vector_query_status,
                    error_type="EmptyEmbeddingVector",
                )
                query_vec = None
        except Exception as exc:
            if embedding_status == "not_started":
                embedding_status = "failed"
                vector_query_status = "skipped"
            else:
                vector_query_status = "failed"
            self._audit_novelty_failure(
                actor=actor,
                write_path=write_path,
                embedding_status=embedding_status,
                vector_query_status=vector_query_status,
                error_type=type(exc).__name__,
            )
            top_similarities = []

        now_iso = utc_now_iso()
        metadata = build_importance_metadata(
            payload=payload,
            scope=scope,
            visibility=visibility,
            top_similarities=top_similarities,
            novelty_computed=novelty_computed,
            novelty_status=novelty_status,
            event_ts_utc=now_iso,
            actor_agent_id=actor.agent_id,
            runtime_writer_model=actor.writer_model,
        )
        self._validate_self_eval_payload(payload, writer_model=str(metadata.get("writer_model", "")).strip())
        metadata["persistence_decision"] = self._decision_payload(decision, write_path=write_path)

        privacy = self.privacy_policy.apply(
            content=raw_content,
            metadata=metadata,
            sensitivity_tags=sensitivity_tags,
        )

        stored_content = privacy.content
        encrypted = False
        if privacy.should_encrypt:
            cipher_result = self.cipher.encrypt(privacy.content)
            stored_content = cipher_result.payload
            encrypted = cipher_result.encrypted

        if self.config.dedup_semantic_enabled and query_vec is not None:
            try:
                semantic_matches = self.vector_store.similarity_search(
                    probe_vector=query_vec,
                    scopes=[scope],
                    version_id=version.version_id,
                    threshold=self.config.dedup_semantic_threshold,
                    limit=1,
                )
            except Exception as exc:
                self.store.add_audit(
                    AuditEvent(
                        action="semantic_dedup_failed",
                        actor=actor.agent_id,
                        reason="degraded",
                        payload={
                            "error_type": type(exc).__name__,
                            "write_path": write_path,
                            "embedding_status": embedding_status,
                            "vector_query_status": "semantic_dedup_failed",
                        },
                    )
                )
                semantic_matches = []

            if semantic_matches:
                duplicate_entry, similarity = semantic_matches[0]
                source_context_hash = str(metadata.get("context_hash", ""))
                existing_context_hash = str(duplicate_entry.metadata.get("context_hash", ""))
                same_context = bool(source_context_hash and source_context_hash == existing_context_hash)
                self._audit_write_attempt(
                    actor=actor,
                    write_path=write_path,
                    decision=decision,
                    scope=scope,
                    entry_id=duplicate_entry.id,
                    outcome="duplicate",
                    duplicate_of=duplicate_entry.id,
                )
                result = {
                    "success": True,
                    "entry_id": duplicate_entry.id,
                    "duplicate_of": duplicate_entry.id,
                    "reason": "semantic-duplicate",
                    "similarity": similarity,
                    "merge_policy": "local-variant-no-consolidate"
                    if same_context
                    else "transferable-candidate-consolidation",
                    "decision": self._decision_payload(decision, write_path=write_path),
                }
                self._log_activity(
                    "write_out",
                    {
                        "operation": write_path,
                        "agent_id": actor.agent_id,
                        "success": result["success"],
                        "entry_id": result["entry_id"],
                        "duplicate_of": result["duplicate_of"],
                        "reason": result["reason"],
                        "similarity": similarity,
                    },
                )
                return result

        now_value = utc_now_iso()
        entry_kwargs: dict[str, Any] = {}
        if write_path in {"import", "migration"} and payload.get("id"):
            entry_kwargs["id"] = payload["id"]

        entry = MemoryEntry(
            **entry_kwargs,
            tier=tier,
            scope=scope,
            visibility=visibility,
            source=payload.get("source", "mcp"),
            type=entry_type,
            status=EntryStatus(payload.get("status", EntryStatus.ACTIVE.value))
            if write_path in {"import", "migration"} and payload.get("status")
            else EntryStatus.ACTIVE,
            content=stored_content,
            context=payload.get("context", ""),
            tags=tags,
            sensitivity_tags=sensitivity_tags,
            metadata=privacy.metadata,
            links=links,
            confidence=float(payload.get("confidence", 0.5)),
            created_at=payload.get("created_at", now_value)
            if write_path in {"import", "migration"}
            else now_iso,
            updated_at=payload.get("updated_at", now_value)
            if write_path in {"import", "migration"}
            else now_iso,
            content_hash=content_hash,
            embedding_version_id=version.version_id,
            encrypted=encrypted,
            redacted=privacy.redacted,
        )

        self.store.add_entry(entry)
        if query_vec is not None:
            self.vector_store.upsert(
                entry_id=entry.id,
                version_id=version.version_id,
                vector=query_vec,
                created_at=now_iso,
            )

        self._audit_write_attempt(
            actor=actor,
            write_path=write_path,
            decision=decision,
            scope=entry.scope,
            entry_id=entry.id,
            outcome="accepted",
        )

        result = {
            "success": True,
            "entry_id": entry.id,
            "tier": entry.tier.value,
            "type": entry.type.value,
            "embedding_version_id": version.version_id,
            "redacted": entry.redacted,
            "encrypted": entry.encrypted,
            "writer_model": entry.metadata.get("writer_model"),
            "context_hash": entry.metadata.get("context_hash"),
            "importance_score": entry.metadata.get("importance_score"),
            "importance_class": entry.metadata.get("importance_class"),
            "novelty_score": entry.metadata.get("novelty_score"),
            "novelty_computed": entry.metadata.get("novelty_computed"),
            "decision": self._decision_payload(decision, write_path=write_path),
        }
        self._log_activity(
            "write_out",
            {
                "operation": write_path,
                "agent_id": actor.agent_id,
                "success": result["success"],
                "entry_id": result["entry_id"],
                "tier": result["tier"],
                "type": result["type"],
                "redacted": result["redacted"],
                "encrypted": result["encrypted"],
            },
        )
        return result

    def get(self, entry_id: str, actor: ActorContext) -> Optional[MemoryEntry]:
        entry = self.store.get_entry(entry_id)
        if entry is None:
            return None
        if not self._can_read(actor, entry):
            raise PermissionError("Read denied by scope policy")
        return entry

    def list_entries(
        self,
        actor: ActorContext,
        limit: int = 50,
        include_invalidated: bool = False,
        tier: Optional[str] = None,
    ) -> list[MemoryEntry]:
        scope = ScopeRef(
            workspace_id=actor.workspace_id,
            project_id=actor.project_id,
            scope_level=ScopeLevel.PROJECT,
        )
        requested_tier = Tier(tier) if tier else None
        entries = self.store.list_entries(
            scope=scope,
            include_invalidated=include_invalidated,
            limit=limit,
            tier=requested_tier,
        )
        return [entry for entry in entries if self._can_read(actor, entry)]

    async def search(
        self,
        query: str,
        actor: ActorContext,
        limit: int = 10,
        include_invalidated: bool = False,
        tier: Optional[str] = None,
        include_project: bool = True,
        include_workspace: bool = True,
        include_global: bool = True,
    ) -> list[MemoryBundle]:
        self._log_activity(
            "query_in",
            {
                "operation": "search",
                "agent_id": actor.agent_id,
                "workspace_id": actor.workspace_id,
                "project_id": actor.project_id,
                "query": self._preview_text(query),
                "limit": limit,
                "include_invalidated": include_invalidated,
                "tier": tier,
            },
        )
        scopes = self._search_scopes(
            actor,
            include_project=include_project,
            include_workspace=include_workspace,
            include_global=include_global,
        )
        version = self.store.get_active_embedding_version() or self._current_embedding_version()
        query_vector = (await self.embedding_provider.embed([query]))[0]

        candidates = self.vector_store.search(
            query_vector=query_vector,
            scopes=scopes,
            version_id=version.version_id,
            limit=max(limit * 4, 20),
            include_invalidated=include_invalidated,
        )

        bundles: list[MemoryBundle] = []
        for entry, similarity in candidates:
            if not self._can_read(actor, entry):
                continue
            if tier and entry.tier.value != tier:
                continue
            if (not include_invalidated) and (
                entry.status == EntryStatus.INVALIDATED or entry.type == EntryType.INVALIDATED
            ):
                continue

            score = self._rank(entry, similarity, actor=actor)
            bundles.append(
                MemoryBundle(
                    entry_id=entry.id,
                    type=entry.type,
                    tier=entry.tier,
                    status=entry.status,
                    scope=entry.scope,
                    visibility=entry.visibility,
                    snippet=self._snippet_for(entry),
                    confidence=max(0.0, min(1.0, score * entry.confidence)),
                    similarity=similarity,
                    score=score,
                    source=entry.source,
                    created_at=entry.created_at,
                    updated_at=entry.updated_at,
                    links=entry.links,
                )
            )

        bundles.sort(key=lambda b: b.score, reverse=True)
        result = bundles[:limit]
        self._log_activity(
            "query_out",
            {
                "operation": "search",
                "agent_id": actor.agent_id,
                "result_count": len(result),
                "has_results": bool(result),
            },
        )
        return result

    def _rank(self, entry: MemoryEntry, similarity: float, *, actor: ActorContext) -> float:
        now = datetime.now(timezone.utc)
        updated = entry.updated_at
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        age_days = max(0.0, (now - updated).total_seconds() / 86400.0)

        recency_score = 1.0 / (1.0 + (age_days / 30.0))
        tier_score = {
            Tier.TIER_1: 0.5,
            Tier.TIER_2: 0.8,
            Tier.TIER_3: 1.0,
        }[entry.tier]
        status_score = {
            EntryStatus.ACTIVE: 1.0,
            EntryStatus.SUPERSEDED: 0.35,
            EntryStatus.INVALIDATED: 0.0,
        }[entry.status]
        scope_score = {
            ScopeLevel.PROJECT: 1.0 if entry.scope.project_id == actor.project_id else 0.0,
            ScopeLevel.WORKSPACE: 0.85,
            ScopeLevel.GLOBAL: 0.7,
        }[entry.scope.scope_level]

        weighted = (
            similarity * self.config.ranking_similarity_weight
            + recency_score * self.config.ranking_recency_weight
            + tier_score * self.config.ranking_tier_weight
            + status_score * self.config.ranking_status_weight
        )
        return max(0.0, min(1.0, weighted * scope_score))

    def invalidate(
        self,
        target_ids: list[str],
        actor: ActorContext,
        reason: str,
        source: str = "mcp",
    ) -> dict:
        self._log_activity(
            "write_in",
            {
                "operation": "invalidate",
                "agent_id": actor.agent_id,
                "target_ids": target_ids,
                "reason": self._preview_text(reason),
                "source": source,
            },
        )
        invalidated: list[str] = []
        now_iso = utc_now_iso()

        for target_id in target_ids:
            entry = self.store.get_entry(target_id)
            if entry is None:
                continue
            if not self._can_read(actor, entry):
                continue

            entry.status = EntryStatus.INVALIDATED
            entry.updated_at = now_iso
            self.store.update_entry(entry)
            invalidated.append(entry.id)

            invalidation_entry = MemoryEntry(
                tier=Tier.TIER_3,
                scope=entry.scope,
                visibility=entry.visibility,
                source=source,
                type=EntryType.INVALIDATED,
                status=EntryStatus.ACTIVE,
                content=reason,
                context=f"Invalidation of {entry.id}",
                links=[EntryLink(target_id=entry.id, relation="invalidates")],
                confidence=1.0,
                created_at=now_iso,
                updated_at=now_iso,
            )
            internal_decision = self._persist_internal_entry(
                actor=actor,
                write_path="invalidate",
                record_type="invalidation_entry",
                internal_reason=reason,
                entry=invalidation_entry,
            )

            self.store.add_audit(
                AuditEvent(
                    entry_id=entry.id,
                    action="invalidate",
                    actor=actor.agent_id,
                    reason=reason,
                    payload={
                        "invalidation_entry_id": invalidation_entry.id,
                        "internal_write": self._decision_payload(
                            internal_decision,
                            write_path="invalidate",
                        ),
                    },
                )
            )

        result = {
            "success": True,
            "invalidated": invalidated,
            "count": len(invalidated),
            "reason": reason,
        }
        self._log_activity(
            "write_out",
            {
                "operation": "invalidate",
                "agent_id": actor.agent_id,
                "success": result["success"],
                "count": result["count"],
                "has_results": bool(invalidated),
            },
        )
        return result

    def promote(
        self,
        entry_ids: list[str],
        actor: ActorContext,
        target_tier: Tier,
        reason: str,
        merge: bool = False,
        summary: Optional[str] = None,
    ) -> dict:
        self._log_activity(
            "write_in",
            {
                "operation": "promote",
                "agent_id": actor.agent_id,
                "entry_ids": entry_ids,
                "target_tier": target_tier.value,
                "reason": self._preview_text(reason),
                "merge": merge,
                "summary": self._preview_text(summary),
            },
        )
        promoted: list[str] = []
        now_iso = utc_now_iso()

        for entry_id in entry_ids:
            entry = self.store.get_entry(entry_id)
            if entry is None:
                continue
            if not self._can_read(actor, entry):
                continue
            if entry.status == EntryStatus.INVALIDATED:
                continue

            entry.tier = target_tier
            entry.updated_at = now_iso
            self.store.update_entry(entry)
            promoted.append(entry.id)

            self.store.add_audit(
                AuditEvent(
                    entry_id=entry.id,
                    action="promote",
                    actor=actor.agent_id,
                    reason=reason,
                    payload={"target_tier": target_tier.value},
                )
            )

        merged_entry_id: Optional[str] = None
        if merge and promoted and summary:
            base_entry = self.store.get_entry(promoted[0])
            if base_entry is None:
                result = {
                    "success": True,
                    "promoted": promoted,
                    "count": len(promoted),
                    "target_tier": target_tier.value,
                    "merged_entry_id": None,
                }
                self._log_activity(
                    "write_out",
                    {
                        "operation": "promote",
                        "agent_id": actor.agent_id,
                        "success": result["success"],
                        "count": result["count"],
                        "merged_entry_id": result["merged_entry_id"],
                    },
                )
                return result
            base_scope = base_entry.scope
            merged_entry = MemoryEntry(
                tier=target_tier,
                scope=base_scope,
                visibility=MemoryScope.SHARED,
                source="consolidation",
                type=EntryType.FACT,
                status=EntryStatus.ACTIVE,
                content=summary,
                context="Merged summary",
                links=[EntryLink(target_id=entry_id, relation="consolidates") for entry_id in promoted],
                confidence=0.9,
                created_at=now_iso,
                updated_at=now_iso,
            )
            merge_reason = reason or "promotion-merge"
            internal_decision = self._persist_internal_entry(
                actor=actor,
                write_path="promote_merge",
                record_type="merge_summary",
                internal_reason=merge_reason,
                entry=merged_entry,
            )
            merged_entry_id = merged_entry.id
            self.store.add_audit(
                AuditEvent(
                    entry_id=merged_entry.id,
                    action="promote_merge",
                    actor=actor.agent_id,
                    reason=merge_reason,
                    payload={
                        "promoted_entry_ids": promoted,
                        "internal_write": self._decision_payload(
                            internal_decision,
                            write_path="promote_merge",
                        ),
                    },
                )
            )

        result = {
            "success": True,
            "promoted": promoted,
            "count": len(promoted),
            "target_tier": target_tier.value,
            "merged_entry_id": merged_entry_id,
        }
        self._log_activity(
            "write_out",
            {
                "operation": "promote",
                "agent_id": actor.agent_id,
                "success": result["success"],
                "count": result["count"],
                "merged_entry_id": result["merged_entry_id"],
            },
        )
        return result

    async def reembed(
        self,
        actor: ActorContext,
        model_id: Optional[str] = None,
        dim: Optional[int] = None,
        activate: bool = True,
        batch_size: int = 64,
    ) -> ReembedResult:
        reembed_provider = get_reembed_provider(
            self.config,
            self.embedding_provider,
            model_id=model_id,
            dim=dim,
        )
        await reembed_provider.prepare()

        base_model = reembed_provider.model_id()
        base_dim = reembed_provider.dimension()
        fingerprint = reembed_provider.fingerprint()
        version_id = f"{reembed_provider.provider_id()}::{base_model}::{fingerprint[:16]}"

        version = EmbeddingVersion(
            version_id=version_id,
            provider_id=reembed_provider.provider_id(),
            embedding_model_id=base_model,
            dim=base_dim,
            fingerprint=fingerprint,
            config={
                "requested_by": actor.agent_id,
                "requested_model_id": model_id,
                "requested_dim": dim,
            },
            active=activate,
        )
        self.store.upsert_embedding_version(version, activate=activate)

        processed = 0
        skipped = 0
        resumed = True
        failed_entry_ids: set[str] = set()

        while True:
            fetch_limit = batch_size + len(failed_entry_ids)
            pending_candidates = self.store.list_entries_missing_embedding(
                version_id=version_id,
                limit=fetch_limit,
            )
            pending = [entry for entry in pending_candidates if entry.id not in failed_entry_ids][:batch_size]
            if not pending:
                break

            plaintexts: list[str] = []
            for entry in pending:
                if entry.encrypted:
                    try:
                        plaintexts.append(self.cipher.decrypt(entry.content))
                    except PayloadDecryptionError:
                        plaintexts.append("")
                        skipped += 1
                        failed_entry_ids.add(entry.id)
                else:
                    plaintexts.append(entry.content)

            vectors = await reembed_provider.embed(plaintexts)
            now_iso = utc_now_iso()
            for index, entry in enumerate(pending):
                vector = vectors[index] if index < len(vectors) else []
                if not self._has_usable_embedding(vector):
                    skipped += 1
                    failed_entry_ids.add(entry.id)
                    continue
                self.vector_store.upsert(
                    entry_id=entry.id,
                    version_id=version_id,
                    vector=vector,
                    created_at=now_iso,
                )
                processed += 1

        remaining = self.store.count_entries_missing_embedding(version_id=version_id)
        self.store.add_audit(
            AuditEvent(
                action="reembed",
                actor=actor.agent_id,
                payload={
                    "version_id": version_id,
                    "processed": processed,
                    "skipped": skipped,
                    "failed_entries": len(failed_entry_ids),
                    "remaining": remaining,
                },
            )
        )

        return ReembedResult(
            version_id=version_id,
            processed=processed,
            skipped=skipped,
            remaining=remaining,
            resumed=resumed,
        )

    def export_data(
        self,
        path: Path,
        fmt: str,
        actor: ActorContext,
    ) -> ExportResult:
        fmt = fmt.lower()
        resolved_path = self._resolve_exchange_path(path, must_exist=False)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        entries = self.store.export_entries(
            scope=ScopeRef(
                workspace_id=actor.workspace_id,
                project_id=actor.project_id,
                scope_level=ScopeLevel.PROJECT,
            )
        )

        if fmt == "jsonl":
            with resolved_path.open("w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
                    f.write("\n")
            count = len(entries)

        elif fmt == "memory.md":
            markdown = render_memory_markdown(entries)
            resolved_path.write_text(markdown, encoding="utf-8")
            count = len(entries)

        elif fmt == "sqlite":
            self.store.backup_to(resolved_path)
            count = len(entries)

        else:
            raise ValueError(f"Unsupported export format: {fmt}")

        self.store.add_audit(
            AuditEvent(
                action="export",
                actor=actor.agent_id,
                payload={"path": str(resolved_path), "format": fmt, "count": count},
            )
        )
        return ExportResult(path=str(resolved_path), format=fmt, count=count)

    async def import_data(
        self,
        path: Path,
        fmt: str,
        actor: ActorContext,
        target_scope: Optional[ScopeRef] = None,
    ) -> ImportResult:
        fmt = fmt.lower()
        resolved_path = self._resolve_exchange_path(path, must_exist=True)
        imported = 0
        duplicates = 0
        rejected = 0

        scope = target_scope or ScopeRef(
            workspace_id=actor.workspace_id,
            project_id=actor.project_id,
            scope_level=ScopeLevel.PROJECT,
            user_id=actor.user_id,
            agent_id=actor.agent_id,
        )

        if fmt == "jsonl":
            rows = [
                json.loads(line)
                for line in resolved_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            entries = [
                self._parse_jsonl_import_row(row, line_number=index)
                for index, row in enumerate(rows, start=1)
            ]
        elif fmt == "memory.md":
            markdown = resolved_path.read_text(encoding="utf-8")
            entries = parse_memory_markdown(markdown, base_scope=scope)
        else:
            raise ValueError(f"Unsupported import format: {fmt}")

        for entry in entries:
            if entry.scope.workspace_id != scope.workspace_id or entry.scope.project_id != scope.project_id:
                entry.scope.workspace_id = scope.workspace_id
                entry.scope.project_id = scope.project_id

            import_payload = entry.model_dump(mode="json")
            if entry.encrypted:
                try:
                    import_payload["content"] = self.cipher.decrypt(entry.content)
                except PayloadDecryptionError:
                    rejected += 1
                    self.store.add_audit(
                        AuditEvent(
                            entry_id=entry.id,
                            action="write_attempt",
                            actor=actor.agent_id,
                            reason="rejected",
                            payload={
                                "decision": "rejected",
                                "accepted": False,
                                "category": "noise",
                                "reason_codes": ["IMPORT_DECRYPT_FAILED"],
                                "confidence": 1.0,
                                "policy_version": POLICY_VERSION,
                                "normalized_summary": "",
                                "write_path": "import",
                                "outcome": "invalid",
                                "scope": entry.scope.model_dump(),
                            },
                        )
                    )
                    continue
            import_payload["agent_id"] = actor.agent_id
            try:
                result = await self.add(import_payload, actor, write_path="import")
            except MemoryInputError:
                rejected += 1
                continue
            if result.get("duplicate_of"):
                duplicates += 1
                continue
            if not result.get("success"):
                rejected += 1
                continue
            imported += 1

        self.store.add_audit(
            AuditEvent(
                action="import",
                actor=actor.agent_id,
                payload={
                    "path": str(resolved_path),
                    "format": fmt,
                    "imported": imported,
                    "duplicates": duplicates,
                    "rejected": rejected,
                },
            )
        )

        return ImportResult(
            path=str(resolved_path),
            format=fmt,
            imported=imported,
            duplicates=duplicates,
            rejected=rejected,
        )
