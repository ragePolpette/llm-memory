"""Modelli dati condivisi per LLM Memory."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .config import MemoryScope, ScopeLevel, Tier


def utc_now() -> datetime:
    """Timestamp UTC standardizzato."""

    return datetime.now(timezone.utc)


def compute_content_hash(content: str) -> str:
    """Calcola SHA-256 del contenuto."""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class EntryType(str, Enum):
    """Tipologie canoniche per memory.md."""

    FACT = "fact"
    ASSUMPTION = "assumption"
    UNKNOWN = "unknown"
    DECISION = "decision"
    INVALIDATED = "invalidated"


class EntryStatus(str, Enum):
    """Stato di una entry."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"


class FastMemoryDistillationStatus(str, Enum):
    """Stato di distillazione di una fast-memory entry."""

    PENDING = "pending"
    SUMMARIZED = "summarized"
    PROMOTED = "promoted"
    DISCARDED = "discarded"


class FastMemoryDistillationRunStatus(str, Enum):
    """Stato del workflow di una distillation run."""

    PREPARED = "prepared"
    REVIEWED = "reviewed"
    APPLIED = "applied"


class ScopeRef(BaseModel):
    """Scope gerarchico: workspace/project/user/agent."""

    workspace_id: str = "default"
    project_id: str = "default"
    scope_level: ScopeLevel = ScopeLevel.PROJECT
    user_id: Optional[str] = None
    agent_id: Optional[str] = None


class ProjectRecord(BaseModel):
    """Metadata progetto per discovery e amministrazione scope."""

    workspace_id: str = "default"
    project_id: str
    display_name: str
    description: str = ""
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class EntryLink(BaseModel):
    """Link esplicito fra entry."""

    target_id: str
    relation: str


class EmbeddingVersion(BaseModel):
    """Metadati versione embedding."""

    version_id: str
    provider_id: str
    embedding_model_id: str
    dim: int
    fingerprint: str
    created_at: datetime = Field(default_factory=utc_now)
    config: dict = Field(default_factory=dict)
    active: bool = True


class MemoryEntry(BaseModel):
    """Entry di memoria persistente Tier-1/2/3."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    tier: Tier = Tier.TIER_1
    scope: ScopeRef = Field(default_factory=ScopeRef)
    visibility: MemoryScope = MemoryScope.SHARED
    source: str = "mcp"
    type: EntryType = EntryType.FACT
    status: EntryStatus = EntryStatus.ACTIVE
    content: str
    context: str = ""
    tags: list[str] = Field(default_factory=list)
    sensitivity_tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    links: list[EntryLink] = Field(default_factory=list)
    confidence: float = 0.5
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    content_hash: str = ""
    embedding_version_id: Optional[str] = None
    encrypted: bool = False
    redacted: bool = False

    def model_post_init(self, __context) -> None:
        if not self.content_hash:
            self.content_hash = compute_content_hash(self.content)

    @field_validator("tags", "sensitivity_tags", mode="before")
    @classmethod
    def _ensure_list(cls, value):
        if value is None:
            return []
        return value

    @field_validator("confidence")
    @classmethod
    def _bound_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, value))


class FastMemoryEntry(BaseModel):
    """Entry episodica separata dalla memoria forte."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = "default"
    project_id: str = "default"
    agent_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    event_type: str = "note"
    content: str
    context: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    source: str = "mcp"
    resolved: bool = False
    distillation_status: FastMemoryDistillationStatus = FastMemoryDistillationStatus.PENDING
    distilled_at: Optional[datetime] = None
    cluster_id: Optional[str] = None
    recurrence_count: int = 1
    first_seen_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)
    selection_score: Optional[float] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("tags", mode="before")
    @classmethod
    def _ensure_fast_tags(cls, value):
        if value is None:
            return []
        return value

    @field_validator("recurrence_count")
    @classmethod
    def _bound_recurrence_count(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator("selection_score")
    @classmethod
    def _bound_selection_score(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return max(0.0, min(1.0, float(value)))


class AuditEvent(BaseModel):
    """Evento audit trail locale."""

    id: Optional[int] = None
    entry_id: Optional[str] = None
    action: str
    actor: str
    reason: Optional[str] = None
    payload: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class FastMemoryDistillationRun(BaseModel):
    """Run tracciata del workflow prepare/review/apply della fast memory."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = "default"
    project_id: str = "default"
    agent_id: str
    user_id: Optional[str] = None
    status: FastMemoryDistillationRunStatus = FastMemoryDistillationRunStatus.PREPARED
    reason: str
    cluster_ids: list[str] = Field(default_factory=list)
    source_entry_ids: list[str] = Field(default_factory=list)
    prepared_payload: dict = Field(default_factory=dict)
    agent_output_payload: dict = Field(default_factory=dict)
    apply_result_payload: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    prepared_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None


class MemoryBundle(BaseModel):
    """Output modello-agnostico del retriever."""

    entry_id: str
    type: EntryType
    tier: Tier
    status: EntryStatus
    scope: ScopeRef
    visibility: MemoryScope
    snippet: str
    confidence: float
    similarity: float
    score: float
    source: str
    created_at: datetime
    updated_at: datetime
    links: list[EntryLink] = Field(default_factory=list)


class ReembedResult(BaseModel):
    """Risultato di una operazione reembed."""

    version_id: str
    processed: int
    skipped: int
    remaining: int
    resumed: bool


class ExportResult(BaseModel):
    """Risultato di export locale."""

    path: str
    format: str
    count: int


class ImportResult(BaseModel):
    """Risultato di import locale."""

    path: str
    format: str
    imported: int
    duplicates: int
    rejected: int = 0
