"""Wiring of the v2 store, embeddings, recall and service, from the environment.

    MEMORY_STORE_PATH        v2 database (default ./data/memory-v2.db)
    MEMORY_SQLITE_PATH       v1 database; migrated once into MEMORY_STORE_PATH when that file
                             does not exist yet (the v1 file is only read, never changed)
    MEMORY_EMBEDDING         on (default) | off
    MEMORY_EMBEDDING_*       provider settings, see embeddings.EmbeddingSettings
    MEMORY_EPISODE_TTL_DAYS  default 60
    MEMORY_SQLITE_VEC        auto (default) | on | off
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .embeddings import EmbeddingSettings, OpenAICompatibleEmbeddings
from .indexer import EmbeddingIndexer
from .migrate_v1 import migrate
from .retrieval import HybridRetriever
from .schema import GLOBAL_PROJECT
from .service import MemoryServiceV2
from .store import MemoryStore

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env: dict, key: str, default: str) -> Path:
    path = Path(env.get(key) or default).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass
class RuntimeSettings:
    store_path: Path
    v1_path: Optional[Path] = None
    embedding: Optional[EmbeddingSettings] = field(default_factory=EmbeddingSettings)
    episode_ttl_days: int = 60
    use_sqlite_vec: str | bool = "auto"

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "RuntimeSettings":
        env = dict(os.environ if env is None else env)
        vec = (env.get("MEMORY_SQLITE_VEC") or "auto").lower()
        embedding_on = (env.get("MEMORY_EMBEDDING") or "on").lower() not in ("off", "0", "false", "no")
        return cls(
            store_path=_path(env, "MEMORY_STORE_PATH", "./data/memory-v2.db"),
            v1_path=_path(env, "MEMORY_SQLITE_PATH", "./data/memory.db"),
            embedding=EmbeddingSettings.from_env(env) if embedding_on else None,
            episode_ttl_days=int(env.get("MEMORY_EPISODE_TTL_DAYS") or 60),
            use_sqlite_vec={"on": True, "true": True, "off": False, "false": False}.get(vec, "auto"),
        )

    def diagnostics(self) -> dict:
        return {
            "store_path": str(self.store_path),
            "embedding": None if self.embedding is None else {
                "provider": self.embedding.provider, "model": self.embedding.model, "dim": self.embedding.dim,
                "key_env": self.embedding.api_key_env,
                "key_present": bool(os.environ.get(self.embedding.api_key_env or "")),
            },
            "episode_ttl_days": self.episode_ttl_days,
        }


@dataclass
class Runtime:
    settings: RuntimeSettings
    store: MemoryStore
    retriever: HybridRetriever
    service: MemoryServiceV2
    indexer: Optional[EmbeddingIndexer] = None
    migration_report: Optional[dict] = None
    _tasks: list = field(default_factory=list)

    def start_background(self) -> None:
        """Start the embedding indexer and the hourly episode expiry on the running loop."""

        loop = asyncio.get_running_loop()
        if self.indexer is not None:
            self._tasks.append(loop.create_task(self.indexer.run_forever()))
        self._tasks.append(loop.create_task(self._expire_forever()))

    async def _expire_forever(self, interval_s: float = 3600.0) -> None:
        while True:
            try:
                removed = await asyncio.to_thread(self.service.expire_episodes)
                if removed:
                    log.info("expired %d episodes", removed)
            except Exception:  # pragma: no cover - keep the loop alive, the next run retries
                log.exception("episode expiry failed")
            await asyncio.sleep(interval_s)

    async def stop_background(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

    def close(self) -> None:
        self.store.close()


def build_runtime(settings: Optional[RuntimeSettings] = None) -> Runtime:
    settings = settings or RuntimeSettings.from_env()
    report = None
    if not settings.store_path.exists() and settings.v1_path is not None and settings.v1_path.exists():
        log.info("migrating v1 database %s into %s", settings.v1_path, settings.store_path)
        report = migrate(settings.v1_path, settings.store_path)
        log.info("migration done: %s", report)
    store = MemoryStore(settings.store_path, use_sqlite_vec=settings.use_sqlite_vec)
    store.ensure_project(GLOBAL_PROJECT, display_name="Global", description="Rules valid in every project")
    client = OpenAICompatibleEmbeddings(settings.embedding) if settings.embedding is not None else None
    indexer = EmbeddingIndexer(store, client) if client is not None else None
    retriever = HybridRetriever(store, client)
    service = MemoryServiceV2(store, retriever, episode_ttl_days=settings.episode_ttl_days)
    return Runtime(settings=settings, store=store, retriever=retriever, service=service, indexer=indexer,
                   migration_report=report)
