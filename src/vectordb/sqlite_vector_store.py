"""Vector store locale su SQLite (embeddings in tabella, ranking in-process)."""

from __future__ import annotations

import math

from ..models import MemoryEntry, ScopeRef
from ..storage.sqlite_store import SQLiteMemoryStore
from .base import VectorStore


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calcola similarità coseno in [0, 1]."""

    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    cosine = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


class SQLiteVectorStore(VectorStore):
    """Implementazione vector store che usa `SQLiteMemoryStore` come persistenza."""

    def __init__(self, store: SQLiteMemoryStore):
        self.store = store

    def upsert(self, entry_id: str, version_id: str, vector: list[float], created_at: str) -> None:
        self.store.upsert_embedding(entry_id=entry_id, version_id=version_id, vector=vector, created_at=created_at)

    def search(
        self,
        query_vector: list[float],
        scope: ScopeRef,
        version_id: str,
        limit: int = 10,
        include_invalidated: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        candidates = self.store.list_embeddings(
            version_id=version_id,
            scope=scope,
            include_invalidated=include_invalidated,
        )
        scored = [(entry, cosine_similarity(query_vector, vec)) for entry, vec in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, limit)]

    def similarity_search(
        self,
        probe_vector: list[float],
        scope: ScopeRef,
        version_id: str,
        threshold: float,
        limit: int = 5,
    ) -> list[tuple[MemoryEntry, float]]:
        matches = self.search(
            query_vector=probe_vector,
            scope=scope,
            version_id=version_id,
            limit=max(limit * 3, limit),
            include_invalidated=False,
        )
        filtered = [item for item in matches if item[1] >= threshold]
        return filtered[:limit]
