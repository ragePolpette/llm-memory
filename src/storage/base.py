"""Interfacce storage per backend swappabili."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..models import AuditEvent, EmbeddingVersion, FastMemoryDistillationStatus, FastMemoryEntry, MemoryEntry, ScopeRef


class MetadataStore(ABC):
    """Interfaccia astratta metadata store."""

    @abstractmethod
    def add_entry(self, entry: MemoryEntry) -> None:
        pass

    @abstractmethod
    def update_entry(self, entry: MemoryEntry) -> None:
        pass

    @abstractmethod
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        pass

    @abstractmethod
    def list_entries(self, scope: ScopeRef, include_invalidated: bool = False, limit: int = 50) -> list[MemoryEntry]:
        pass

    @abstractmethod
    def find_by_hash(self, scope: ScopeRef, content_hash: str) -> Optional[MemoryEntry]:
        pass

    @abstractmethod
    def upsert_embedding_version(self, version: EmbeddingVersion, activate: bool = False) -> None:
        pass

    @abstractmethod
    def set_active_embedding_version(self, version_id: str) -> None:
        pass

    @abstractmethod
    def get_active_embedding_version(self) -> Optional[EmbeddingVersion]:
        pass

    @abstractmethod
    def upsert_embedding(self, entry_id: str, version_id: str, vector: list[float], created_at: str) -> None:
        pass

    @abstractmethod
    def add_audit(self, event: AuditEvent) -> int:
        pass

    @abstractmethod
    def add_fast_entry(self, entry: FastMemoryEntry) -> None:
        pass

    @abstractmethod
    def update_fast_entry(self, entry: FastMemoryEntry) -> None:
        pass

    @abstractmethod
    def get_fast_entry(self, entry_id: str) -> Optional[FastMemoryEntry]:
        pass

    @abstractmethod
    def list_fast_entries(
        self,
        *,
        workspace_id: Optional[str] = None,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        event_type: Optional[str] = None,
        resolved: Optional[bool] = None,
        distillation_status: FastMemoryDistillationStatus | None = None,
        limit: int = 100,
    ) -> list[FastMemoryEntry]:
        pass

    @abstractmethod
    def count_fast_entries(
        self,
        *,
        workspace_id: Optional[str] = None,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        event_type: Optional[str] = None,
        resolved: Optional[bool] = None,
        distillation_status: FastMemoryDistillationStatus | None = None,
    ) -> int:
        pass
