"""Servizio applicativo memoria Tier-1/2/3 con governance e audit."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import Config, MemoryScope, Tier
from ..embedding.embedding_service import EmbeddingProvider
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
    ReembedResult,
    ScopeRef,
    compute_content_hash,
)
from ..security.crypto import PayloadCipher
from ..security.privacy import PrivacyPolicy
from ..storage.sqlite_store import SQLiteMemoryStore
from ..vectordb.sqlite_vector_store import SQLiteVectorStore
from ..interop.memory_markdown import parse_memory_markdown, render_memory_markdown
from .importance_scoring import (
    build_importance_metadata,
    has_inference_signal,
    has_surprise_signal,
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

    def default_scope(self, *, agent_id: Optional[str] = None, user_id: Optional[str] = None) -> ScopeRef:
        return ScopeRef(
            workspace_id=self.config.default_workspace_id,
            project_id=self.config.default_project_id,
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
        return ScopeRef(
            workspace_id=scope_payload.get("workspace_id", actor.workspace_id),
            project_id=scope_payload.get("project_id", actor.project_id),
            user_id=scope_payload.get("user_id", actor.user_id),
            agent_id=scope_payload.get("agent_id", actor.agent_id),
        )

    def _can_read(self, actor: ActorContext, entry: MemoryEntry) -> bool:
        if entry.scope.workspace_id != actor.workspace_id:
            return False
        if entry.scope.project_id != actor.project_id:
            return False

        if entry.visibility == MemoryScope.PRIVATE:
            if entry.scope.user_id != actor.user_id:
                return False
            if entry.scope.agent_id != actor.agent_id:
                return False
        return True

    def _can_write(self, actor: ActorContext, visibility: MemoryScope, scope: ScopeRef) -> bool:
        if visibility == MemoryScope.GLOBAL:
            return False
        if scope.workspace_id != actor.workspace_id:
            return False
        if scope.project_id != actor.project_id:
            return False
        if visibility == MemoryScope.PRIVATE and scope.agent_id and scope.agent_id != actor.agent_id:
            return False
        return True

    def _snippet_for(self, entry: MemoryEntry, max_chars: int = 220) -> str:
        content = entry.content
        if entry.encrypted:
            try:
                content = self.cipher.decrypt(entry.content)
            except Exception:
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

    async def add(self, payload: dict, actor: ActorContext) -> dict:
        scope = self._scope_from_payload(payload, actor)
        visibility = MemoryScope(payload.get("visibility", payload.get("scope_visibility", "shared")))
        if not self._can_write(actor, visibility, scope):
            raise PermissionError("Write denied by scope policy")

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
                return {
                    "success": True,
                    "entry_id": duplicated.id,
                    "duplicate_of": duplicated.id,
                    "reason": "hash-duplicate",
                }

        version = self._current_embedding_version()
        query_vec = (await self.embedding_provider.embed([raw_content]))[0]
        novelty_computed = True
        top_similarities: list[float] = []

        try:
            novelty_candidates = self.vector_store.search(
                query_vector=query_vec,
                scope=scope,
                version_id=version.version_id,
                limit=5,
                include_invalidated=False,
            )
            top_similarities = [similarity for _, similarity in novelty_candidates]
        except Exception:
            novelty_computed = False
            top_similarities = []

        now_iso = utc_now_iso()
        metadata = build_importance_metadata(
            payload=payload,
            scope=scope,
            visibility=visibility,
            top_similarities=top_similarities,
            novelty_computed=novelty_computed,
            event_ts_utc=now_iso,
            actor_agent_id=actor.agent_id,
            runtime_writer_model=actor.writer_model,
        )
        self._validate_self_eval_payload(payload, writer_model=str(metadata.get("writer_model", "")).strip())

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

        if self.config.dedup_semantic_enabled:
            semantic_matches = self.vector_store.similarity_search(
                probe_vector=query_vec,
                scope=scope,
                version_id=version.version_id,
                threshold=self.config.dedup_semantic_threshold,
                limit=1,
            )
            if semantic_matches:
                duplicate_entry, similarity = semantic_matches[0]
                source_context_hash = str(metadata.get("context_hash", ""))
                existing_context_hash = str(duplicate_entry.metadata.get("context_hash", ""))
                same_context = bool(source_context_hash and source_context_hash == existing_context_hash)
                return {
                    "success": True,
                    "entry_id": duplicate_entry.id,
                    "duplicate_of": duplicate_entry.id,
                    "reason": "semantic-duplicate",
                    "similarity": similarity,
                    "merge_policy": "local-variant-no-consolidate"
                    if same_context
                    else "transferable-candidate-consolidation",
                }

        entry = MemoryEntry(
            tier=tier,
            scope=scope,
            visibility=visibility,
            source=payload.get("source", "mcp"),
            type=entry_type,
            status=EntryStatus.ACTIVE,
            content=stored_content,
            context=payload.get("context", ""),
            tags=tags,
            sensitivity_tags=sensitivity_tags,
            metadata=privacy.metadata,
            links=links,
            confidence=float(payload.get("confidence", 0.5)),
            created_at=now_iso,
            updated_at=now_iso,
            content_hash=content_hash,
            embedding_version_id=version.version_id,
            encrypted=encrypted,
            redacted=privacy.redacted,
        )

        self.store.add_entry(entry)
        self.vector_store.upsert(
            entry_id=entry.id,
            version_id=version.version_id,
            vector=query_vec,
            created_at=now_iso,
        )

        self.store.add_audit(
            AuditEvent(
                entry_id=entry.id,
                action="add",
                actor=actor.agent_id,
                payload={
                    "tier": entry.tier.value,
                    "type": entry.type.value,
                    "visibility": entry.visibility.value,
                    "scope": entry.scope.model_dump(),
                },
            )
        )

        return {
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
        }

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
        scope = ScopeRef(workspace_id=actor.workspace_id, project_id=actor.project_id)
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
    ) -> list[MemoryBundle]:
        scope = ScopeRef(workspace_id=actor.workspace_id, project_id=actor.project_id)
        version = self.store.get_active_embedding_version() or self._current_embedding_version()
        query_vector = (await self.embedding_provider.embed([query]))[0]

        candidates = self.vector_store.search(
            query_vector=query_vector,
            scope=scope,
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

            score = self._rank(entry, similarity)
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
        return bundles[:limit]

    def _rank(self, entry: MemoryEntry, similarity: float) -> float:
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

        weighted = (
            similarity * self.config.ranking_similarity_weight
            + recency_score * self.config.ranking_recency_weight
            + tier_score * self.config.ranking_tier_weight
            + status_score * self.config.ranking_status_weight
        )
        return max(0.0, min(1.0, weighted))

    def invalidate(
        self,
        target_ids: list[str],
        actor: ActorContext,
        reason: str,
        source: str = "mcp",
    ) -> dict:
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
            self.store.add_entry(invalidation_entry)

            self.store.add_audit(
                AuditEvent(
                    entry_id=entry.id,
                    action="invalidate",
                    actor=actor.agent_id,
                    reason=reason,
                    payload={"invalidation_entry_id": invalidation_entry.id},
                )
            )

        return {
            "success": True,
            "invalidated": invalidated,
            "count": len(invalidated),
            "reason": reason,
        }

    def promote(
        self,
        entry_ids: list[str],
        actor: ActorContext,
        target_tier: Tier,
        reason: str,
        merge: bool = False,
        summary: Optional[str] = None,
    ) -> dict:
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
            base_scope = self.store.get_entry(promoted[0]).scope
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
            self.store.add_entry(merged_entry)
            merged_entry_id = merged_entry.id

        return {
            "success": True,
            "promoted": promoted,
            "count": len(promoted),
            "target_tier": target_tier.value,
            "merged_entry_id": merged_entry_id,
        }

    async def reembed(
        self,
        actor: ActorContext,
        model_id: Optional[str] = None,
        dim: Optional[int] = None,
        activate: bool = True,
        batch_size: int = 64,
    ) -> ReembedResult:
        # Versione embedding corrente provider + eventuale override model/dim a livello metadata.
        base_model = model_id or self.embedding_provider.model_id()
        base_dim = dim or self.embedding_provider.dimension()

        fingerprint = self.embedding_provider.fingerprint() + f"::{base_model}::{base_dim}"
        version_id = f"{self.embedding_provider.provider_id()}::{base_model}::{fingerprint[:16]}"

        version = EmbeddingVersion(
            version_id=version_id,
            provider_id=self.embedding_provider.provider_id(),
            embedding_model_id=base_model,
            dim=base_dim,
            fingerprint=fingerprint,
            config={"requested_by": actor.agent_id},
            active=activate,
        )
        self.store.upsert_embedding_version(version, activate=activate)

        processed = 0
        skipped = 0
        resumed = True

        while True:
            pending = self.store.list_entries_missing_embedding(version_id=version_id, limit=batch_size)
            if not pending:
                break

            plaintexts: list[str] = []
            for entry in pending:
                if entry.encrypted:
                    try:
                        plaintexts.append(self.cipher.decrypt(entry.content))
                    except Exception:
                        plaintexts.append("")
                        skipped += 1
                else:
                    plaintexts.append(entry.content)

            vectors = await self.embedding_provider.embed(plaintexts)
            now_iso = utc_now_iso()
            for entry, vector in zip(pending, vectors):
                if not vector:
                    skipped += 1
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
            scope=ScopeRef(workspace_id=actor.workspace_id, project_id=actor.project_id)
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
            shutil.copy2(self.store.db_path, resolved_path)
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

        scope = target_scope or ScopeRef(
            workspace_id=actor.workspace_id,
            project_id=actor.project_id,
            user_id=actor.user_id,
            agent_id=actor.agent_id,
        )

        if fmt == "jsonl":
            rows = [
                json.loads(line)
                for line in resolved_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            entries = [MemoryEntry(**row) for row in rows]
        elif fmt == "memory.md":
            markdown = resolved_path.read_text(encoding="utf-8")
            entries = parse_memory_markdown(markdown, base_scope=scope)
        else:
            raise ValueError(f"Unsupported import format: {fmt}")

        for entry in entries:
            if entry.scope.workspace_id != scope.workspace_id or entry.scope.project_id != scope.project_id:
                entry.scope.workspace_id = scope.workspace_id
                entry.scope.project_id = scope.project_id

            if self.store.find_by_hash(scope=entry.scope, content_hash=entry.content_hash):
                duplicates += 1
                continue

            self.store.add_entry(entry)
            imported += 1

            active = self.store.get_active_embedding_version() or self._current_embedding_version()
            content = entry.content
            if entry.encrypted:
                try:
                    content = self.cipher.decrypt(entry.content)
                except Exception:
                    content = ""
            vectors = await self.embedding_provider.embed([content])
            if vectors and vectors[0]:
                self.vector_store.upsert(
                    entry_id=entry.id,
                    version_id=active.version_id,
                    vector=vectors[0],
                    created_at=utc_now_iso(),
                )

        self.store.add_audit(
            AuditEvent(
                action="import",
                actor=actor.agent_id,
                payload={
                    "path": str(resolved_path),
                    "format": fmt,
                    "imported": imported,
                    "duplicates": duplicates,
                },
            )
        )

        return ImportResult(
            path=str(resolved_path),
            format=fmt,
            imported=imported,
            duplicates=duplicates,
        )
