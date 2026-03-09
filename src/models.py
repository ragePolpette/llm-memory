"""Modelli dati condivisi per LLM Memory."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from .config import MemoryScope, Tier


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


class ScopeRef(BaseModel):
    """Scope gerarchico: workspace/project/user/agent."""

    workspace_id: str = "default"
    project_id: str = "default"
    user_id: Optional[str] = None
    agent_id: Optional[str] = None


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


class AuditEvent(BaseModel):
    """Evento audit trail locale."""

    id: Optional[int] = None
    entry_id: Optional[str] = None
    action: str
    actor: str
    reason: Optional[str] = None
    payload: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


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


# -------------------------
# Compatibilità API legacy
# -------------------------
class Memory(BaseModel):
    """Rappresenta una singola memoria nel sistema (v1 legacy)."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    context: str
    agent_id: str
    session_id: Optional[str] = None
    scope: MemoryScope = MemoryScope.SHARED
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    content_hash: str = ""

    def model_post_init(self, __context) -> None:
        if not self.content_hash:
            self.content_hash = compute_content_hash(self.content)

    @field_validator("tags", mode="before")
    @classmethod
    def ensure_list(cls, value):
        if value is None:
            return []
        return value


class MemoryWriteResult(BaseModel):
    """Risultato di una operazione di scrittura."""

    success: bool
    memory_id: UUID | str
    indexed: bool
    mode: str
    duplicate_of: Optional[UUID | str] = None
    message: Optional[str] = None


class SearchResult(BaseModel):
    """Risultato compatibile con la ricerca legacy."""

    memory_id: UUID | str
    content: str
    context: str
    agent_id: str
    scope: MemoryScope
    score: float
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    indexed: bool = True


class IndexResult(BaseModel):
    """Risultato di una operazione di indicizzazione."""

    indexed: bool
    mode: str
    queued: bool = False
    error: Optional[str] = None


class MemorySummary(BaseModel):
    """Sommario di una memoria per listing."""

    memory_id: UUID | str
    context: str
    agent_id: str
    scope: MemoryScope
    tags: list[str]
    created_at: datetime
    content_preview: str
