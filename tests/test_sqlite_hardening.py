from __future__ import annotations

import sqlite3

import pytest

import src.storage.sqlite_store as sqlite_store_module
from src.config import MemoryScope, Tier
from src.models import EntryType
from src.service.memory_service import ActorContext, EmbeddingUnavailableError
from src.storage.sqlite_store import SQLiteMemoryStore

ACTOR = ActorContext(agent_id="agent-sqlite", user_id="user-sqlite", workspace_id="ws-test", project_id="prj-test")
CONTENT = "Il progetto usa SQLite locale come backend persistente del runtime."


def test_every_connection_sets_timeout_busy_timeout_and_foreign_keys(tmp_path, monkeypatch):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    seen_timeouts: list[float] = []
    real_connect = sqlite3.connect

    def spy_connect(*args, **kwargs):
        seen_timeouts.append(kwargs.get("timeout"))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite_store_module.sqlite3, "connect", spy_connect)

    with store._conn() as conn:
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]

    assert seen_timeouts == [30.0]
    assert busy_timeout == 30000
    assert foreign_keys == 1


def test_foreign_keys_are_enforced_outside_init(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")

    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_embedding("missing-entry", "missing-version", [0.1, 0.2], "2026-01-01T00:00:00+00:00")


@pytest.mark.asyncio
async def test_add_is_atomic_when_vector_write_fails(service, monkeypatch):
    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(service.store, "_upsert_embedding_row", boom)

    with pytest.raises(sqlite3.OperationalError):
        await service.add(
            {"content": CONTENT, "context": "architecture", "agent_id": ACTOR.agent_id, "visibility": "shared"},
            ACTOR,
        )

    assert service.list_entries(ACTOR, limit=10) == []


@pytest.mark.asyncio
async def test_added_entry_always_has_a_vector(service):
    result = await service.add(
        {"content": CONTENT, "context": "architecture", "agent_id": ACTOR.agent_id, "visibility": "shared"},
        ACTOR,
    )

    assert result["success"] is True
    vector = service.store.get_embedding(result["entry_id"], result["embedding_version_id"])
    assert vector and any(vector)


@pytest.mark.asyncio
async def test_promote_fast_aborts_without_partial_entry_when_embedding_fails(service):
    created = service.log_fast(
        {
            "content": "Import incrementale fallisce dopo resume se il checkpoint non viene riallineato.",
            "agent_id": ACTOR.agent_id,
            "event_type": "incident",
        },
        ACTOR,
    )

    async def failing_embed(texts):
        raise RuntimeError("embed-fail")

    service.embedding_provider.embed = failing_embed  # type: ignore[method-assign]

    with pytest.raises(EmbeddingUnavailableError, match="Promotion aborted"):
        await service.promote_fast(
            entry_id=created["entry_id"],
            actor=ACTOR,
            reason="recurring import issue became reusable knowledge",
            target_tier=Tier.TIER_2,
            memory_type=EntryType.FACT,
            visibility=MemoryScope.SHARED,
        )

    assert service.list_entries(ACTOR, limit=10) == []
    fast_entry = service.get_fast(created["entry_id"], ACTOR)
    assert fast_entry is not None
    assert fast_entry.distillation_status.value == "pending"


@pytest.mark.asyncio
async def test_promote_fast_stores_entry_and_vector_together(service):
    created = service.log_fast(
        {
            "content": "Import incrementale fallisce dopo resume se il checkpoint non viene riallineato.",
            "agent_id": ACTOR.agent_id,
            "event_type": "incident",
        },
        ACTOR,
    )

    result = await service.promote_fast(
        entry_id=created["entry_id"],
        actor=ACTOR,
        reason="recurring import issue became reusable knowledge",
        target_tier=Tier.TIER_2,
        memory_type=EntryType.FACT,
        visibility=MemoryScope.SHARED,
    )

    entry = service.store.get_entry(result["promoted_entry_id"])
    assert entry is not None
    assert service.store.get_embedding(entry.id, entry.embedding_version_id)
