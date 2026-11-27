"""Background embedding of new items.

Writes never wait for the provider: a note or memory is stored and searchable by text
at once, and gets its vector when the indexer reaches it. If the provider is down the
items stay queued and the next pass retries them; nothing is lost.

When the configured provider, model, dimension or query instruction changes, a new
embedding version becomes active and every item is queued again (the old vectors stay
until the new version is complete, then can be dropped).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from .embeddings import EmbeddingClient, EmbeddingError
from .store import MemoryStore

log = logging.getLogger(__name__)


@dataclass
class IndexPass:
    embedded: int = 0
    failed: int = 0
    remaining: int = 0
    error: Optional[str] = None


class EmbeddingIndexer:
    def __init__(self, store: MemoryStore, client: EmbeddingClient, *, batch_size: int = 64):
        self.store = store
        self.client = client
        self.batch_size = batch_size
        self.version_id = store.register_embedding_version(
            provider=client.provider, model=client.model, dim=client.dim, config=client.version_config(),
        )

    def run_once(self, *, max_items: Optional[int] = None) -> IndexPass:
        """Embed queued items until the queue is empty, max_items is reached or the provider fails."""

        result = IndexPass()
        while max_items is None or result.embedded < max_items:
            limit = self.batch_size if max_items is None else min(self.batch_size, max_items - result.embedded)
            batch = self.store.missing_embeddings(self.version_id, limit=limit)
            if not batch:
                break
            try:
                vectors = self.client.embed_documents([text for _, _, text in batch])
            except EmbeddingError as exc:
                result.failed += len(batch)
                result.error = str(exc)
                log.warning("embedding pass stopped: %s", exc)
                break
            for (kind, item_id, _), vector in zip(batch, vectors):
                try:
                    self.store.put_embedding(kind, item_id, self.version_id, vector)
                    result.embedded += 1
                except ValueError as exc:  # e.g. a zero vector: keep going, the item stays queued
                    result.failed += 1
                    result.error = str(exc)
            if len(batch) < limit or result.failed:
                break
        result.remaining = len(self.store.missing_embeddings(self.version_id, limit=1000))
        return result

    async def run_forever(self, *, interval_s: float = 5.0, idle_interval_s: float = 30.0) -> None:
        """Keep the index current. Backs off while the provider is failing."""

        backoff = interval_s
        while True:
            outcome = await asyncio.to_thread(self.run_once)
            if outcome.error:
                backoff = min(backoff * 2, 600.0)
            else:
                backoff = interval_s
            await asyncio.sleep(backoff if outcome.error or outcome.remaining else idle_interval_s)
