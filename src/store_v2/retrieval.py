"""Hybrid recall: BM25 and vector search fused with Reciprocal Rank Fusion.

BM25 catches exact identifiers (ticket ids, config keys, error codes); vectors catch
paraphrases and the other language. RRF merges the two rankings by rank only, so the
two score scales never need calibrating against each other.

Recall degrades instead of failing: with no provider, an unreachable provider, or a
query vector of the wrong version, it answers from text alone and says so.

The answer is cut to a character budget, so a caller can bound what lands in its
context window without knowing how long each item is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .embeddings import EmbeddingClient, EmbeddingError
from .schema import GLOBAL_PROJECT
from .store import MemoryStore

log = logging.getLogger(__name__)

RRF_K = 60  # the constant from Cormack et al. (2009); rankings this short are not sensitive to it


@dataclass
class RecallItem:
    kind: str
    id: str
    project_id: str
    score: float
    matched_by: list[str]
    text: str
    item: dict


@dataclass
class RecallResult:
    items: list[RecallItem] = field(default_factory=list)
    mode: str = "hybrid"  # "hybrid" | "text" | "vector"
    degraded: Optional[str] = None
    truncated: bool = False
    used_chars: int = 0


def render(item: dict) -> str:
    """The text a caller reads for one item."""

    if item["kind"] == "episode":
        parts = [item["content"], item.get("context") or ""]
    else:
        parts = [item["statement"]]
        if item.get("why"):
            parts.append(f"Why: {item['why']}")
        if item.get("how"):
            parts.append(f"How: {item['how']}")
    return "\n".join(part for part in parts if part)


class HybridRetriever:
    def __init__(self, store: MemoryStore, client: Optional[EmbeddingClient] = None, *,
                 candidates: int = 50, rrf_k: int = RRF_K):
        self.store = store
        self.client = client
        self.candidates = candidates
        self.rrf_k = rrf_k
        self._version_id: Optional[str] = None
        if client is not None:
            self._version_id = store.register_embedding_version(
                provider=client.provider, model=client.model, dim=client.dim, config=client.version_config(),
                activate=False,
            )

    def recall(self, query: str, project_ids: Iterable[str], *, include_global: bool = True,
               kinds: Iterable[str] = ("memory", "episode"), include_closed: bool = False,
               limit: int = 10, budget_chars: Optional[int] = None) -> RecallResult:
        projects = list(dict.fromkeys(project_ids))
        if include_global and GLOBAL_PROJECT not in projects:
            projects.append(GLOBAL_PROJECT)
        kinds = tuple(kinds)
        result = RecallResult()

        rankings: dict[str, list[tuple[str, str, str]]] = {}
        text_hits = self.store.search_text(query, projects, kinds=kinds, include_closed=include_closed,
                                           limit=self.candidates)
        rankings["text"] = [(h.kind, h.id, h.project_id) for h in text_hits]

        if self.client is None:
            result.degraded = "no embedding provider configured"
        else:
            try:
                vector = self.client.embed_query(query)
                vector_hits = self.store.search_vector(vector, projects, self._version_id, kinds=kinds,
                                                       include_closed=include_closed, limit=self.candidates)
                rankings["vector"] = [(h.kind, h.id, h.project_id) for h in vector_hits]
            except EmbeddingError as exc:
                result.degraded = f"vector search unavailable: {exc}"
                log.warning("recall fell back to text: %s", exc)
        result.mode = "hybrid" if len(rankings) == 2 else "text"

        fused: dict[tuple[str, str], dict] = {}
        for name, ranking in rankings.items():
            for rank, (kind, item_id, project_id) in enumerate(ranking, start=1):
                entry = fused.setdefault((kind, item_id), {"project_id": project_id, "score": 0.0, "matched_by": []})
                entry["score"] += 1.0 / (self.rrf_k + rank)
                entry["matched_by"].append(name)
        ordered = sorted(fused.items(), key=lambda kv: kv[1]["score"], reverse=True)

        for (kind, item_id), meta in ordered:
            if len(result.items) >= limit:
                result.truncated = True
                break
            item = self.store.get_memory(item_id) if kind == "memory" else self.store.get_episode(item_id)
            if item is None:  # removed between search and fetch
                continue
            text = render(item)
            if budget_chars is not None and result.items and result.used_chars + len(text) > budget_chars:
                result.truncated = True
                break
            result.items.append(RecallItem(kind, item_id, meta["project_id"], round(meta["score"], 6),
                                           meta["matched_by"], text, item))
            result.used_chars += len(text)
        return result
