"""Unit test: storage + dedup + scope isolation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.service.memory_service import ActorContext
from src.storage.sqlite_store import SQLiteMemoryStore


@pytest.mark.asyncio
async def test_dedup_hash(service):
    actor = ActorContext(agent_id="agent-a", user_id="user-a", workspace_id="ws-test", project_id="prj-test")

    first = await service.add(
        {
            "content": "Il progetto usa SQLite locale per metadata.",
            "context": "arch",
            "agent_id": actor.agent_id,
            "visibility": "shared",
            "tier": "tier-2",
            "type": "fact",
        },
        actor,
    )

    second = await service.add(
        {
            "content": "Il progetto usa SQLite locale per metadata.",
            "context": "arch",
            "agent_id": actor.agent_id,
            "visibility": "shared",
            "tier": "tier-2",
            "type": "fact",
        },
        actor,
    )

    assert first["success"] is True
    assert second["duplicate_of"] == first["entry_id"]


@pytest.mark.asyncio
async def test_scope_isolation(service):
    actor_a = ActorContext(agent_id="agent-a", user_id="user-a", workspace_id="ws-test", project_id="project-a")
    actor_b = ActorContext(agent_id="agent-b", user_id="user-b", workspace_id="ws-test", project_id="project-b")
    service.create_project(actor=actor_a, project_id="project-a")
    service.create_project(actor=actor_b, project_id="project-b")

    await service.add(
        {
            "content": "Il progetto A richiede una memoria privata dedicata all agente proprietario.",
            "agent_id": actor_a.agent_id,
            "scope": {
                "workspace_id": "ws-test",
                "project_id": "project-a",
                "agent_id": "agent-a",
            },
            "visibility": "private",
            "tier": "tier-2",
            "type": "fact",
        },
        actor_a,
    )

    results_a = await service.search("Dato privato", actor_a, limit=5)
    results_b = await service.search("Dato privato", actor_b, limit=5)

    assert len(results_a) >= 1
    assert len(results_b) == 0


def test_sqlite_store_migrates_legacy_entries_without_scope_level(tmp_path: Path):
    db_path = tmp_path / "legacy-memory.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE entries (
            id TEXT PRIMARY KEY,
            tier TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            user_id TEXT,
            agent_id TEXT,
            visibility TEXT NOT NULL,
            source TEXT NOT NULL,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            content TEXT NOT NULL,
            context TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            sensitivity_tags_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            embedding_version_id TEXT,
            encrypted INTEGER NOT NULL DEFAULT 0,
            redacted INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()

    SQLiteMemoryStore(db_path)

    verify = sqlite3.connect(db_path)
    verify.row_factory = sqlite3.Row
    columns = {
        row["name"]: row
        for row in verify.execute("PRAGMA table_info(entries)").fetchall()
    }
    verify.close()

    assert "scope_level" in columns
    assert columns["scope_level"]["dflt_value"] == "'project'"
