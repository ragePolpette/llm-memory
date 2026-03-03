"""Modelli dati condivisi per LLM Memory."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from .config import MemoryScope


class Memory(BaseModel):
    """Rappresenta una singola memoria nel sistema."""
    
    id: UUID = Field(default_factory=uuid4)
    content: str
    context: str
    agent_id: str
    session_id: Optional[str] = None
    scope: MemoryScope = MemoryScope.SHARED
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str = ""
    
    def model_post_init(self, __context) -> None:
        """Calcola hash del contenuto dopo inizializzazione."""
        if not self.content_hash:
            self.content_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Calcola SHA-256 del contenuto."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()
    
    @field_validator("tags", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        return v


class MemoryWriteResult(BaseModel):
    """Risultato di una operazione di scrittura."""
    
    success: bool
    memory_id: UUID
    indexed: bool
    mode: str  # sync | async
    duplicate_of: Optional[UUID] = None
    message: Optional[str] = None


class SearchResult(BaseModel):
    """Risultato di una ricerca semantica."""
    
    memory_id: UUID
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
    
    memory_id: UUID
    context: str
    agent_id: str
    scope: MemoryScope
    tags: list[str]
    created_at: datetime
    content_preview: str  # Primi 200 caratteri
