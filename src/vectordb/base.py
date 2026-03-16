"""Interfacce vector store locali."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import MemoryEntry, ScopeRef


class VectorStore(ABC):
    """Interfaccia astratta per ricerca vettoriale locale."""

    @abstractmethod
    def upsert(self, entry_id: str, version_id: str, vector: list[float], created_at: str) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        scopes: list[ScopeRef],
        version_id: str,
        limit: int = 10,
        include_invalidated: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        pass

    @abstractmethod
    def similarity_search(
        self,
        probe_vector: list[float],
        scopes: list[ScopeRef],
        version_id: str,
        threshold: float,
        limit: int = 5,
    ) -> list[tuple[MemoryEntry, float]]:
        pass
