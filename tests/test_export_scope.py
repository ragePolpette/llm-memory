from __future__ import annotations

import json
import sqlite3

import pytest

from src.service.memory_service import ActorContext

OWNER = ActorContext(agent_id="agent-owner", user_id="user-owner", workspace_id="ws-test", project_id="prj-test")
OTHER = ActorContext(agent_id="agent-other", user_id="user-other", workspace_id="ws-test", project_id="prj-test")


async def _seed(service) -> dict[str, str]:
    private = await service.add(
        {
            "content": "Il progetto mantiene audit trail locali per le operazioni di memoria.",
            "context": "private-note",
            "agent_id": OWNER.agent_id,
            "visibility": "private",
        },
        OWNER,
    )
    shared = await service.add(
        {
            "content": "Il progetto usa SQLite come backend locale condiviso tra gli agent.",
            "context": "architecture",
            "agent_id": OWNER.agent_id,
            "visibility": "shared",
        },
        OWNER,
    )
    other_private = await service.add(
        {
            "content": "Il progetto usa MemoryService come orchestratore della persistenza.",
            "context": "private-note",
            "agent_id": OTHER.agent_id,
            "visibility": "private",
        },
        OTHER,
    )
    service.log_fast(
        {"content": "Fast note che non deve finire in un export.", "agent_id": OWNER.agent_id},
        OWNER,
    )
    ids = {
        "private": private["entry_id"],
        "shared": shared["entry_id"],
        "other_private": other_private["entry_id"],
    }
    assert all(ids.values())
    return ids


@pytest.mark.asyncio
async def test_jsonl_export_excludes_private_entries_of_other_agents(service):
    ids = await _seed(service)

    result = service.export_data(service.config.import_export_base_dir / "other.jsonl", "jsonl", OTHER)

    lines = (service.config.import_export_base_dir / "other.jsonl").read_text(encoding="utf-8").splitlines()
    exported_ids = {json.loads(line)["id"] for line in lines}
    assert exported_ids == {ids["shared"], ids["other_private"]}
    assert result.count == 2


@pytest.mark.asyncio
async def test_memory_md_export_excludes_private_entries_of_other_agents(service):
    await _seed(service)

    service.export_data(service.config.import_export_base_dir / "other.md", "memory.md", OTHER)

    markdown = (service.config.import_export_base_dir / "other.md").read_text(encoding="utf-8")
    assert "audit trail locali" not in markdown
    assert "SQLite come backend locale" in markdown


@pytest.mark.asyncio
async def test_sqlite_export_contains_only_readable_rows(service):
    ids = await _seed(service)
    target = service.config.import_export_base_dir / "other.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"stale content")

    result = service.export_data(target, "sqlite", OTHER)

    assert result.count == 2
    conn = sqlite3.connect(target)
    try:
        entry_ids = {row[0] for row in conn.execute("SELECT id FROM entries")}
        embedding_ids = {row[0] for row in conn.execute("SELECT entry_id FROM embeddings")}
        audit_count = conn.execute("SELECT COUNT(1) FROM audit_log").fetchone()[0]
        fast_count = conn.execute("SELECT COUNT(1) FROM fast_memory_entries").fetchone()[0]
        projects = {row[0] for row in conn.execute("SELECT project_id FROM projects")}
        versions = conn.execute("SELECT COUNT(1) FROM embedding_versions").fetchone()[0]
        contents = " ".join(row[0] for row in conn.execute("SELECT content FROM entries"))
    finally:
        conn.close()

    assert entry_ids == {ids["shared"], ids["other_private"]}
    assert embedding_ids == entry_ids
    assert audit_count == 0
    assert fast_count == 0
    assert projects == {"prj-test"}
    assert versions == 1
    assert "audit trail locali" not in contents
    leftovers = [p.name for p in target.parent.iterdir() if p.name.startswith(".other.db.")]
    assert leftovers == []


@pytest.mark.asyncio
async def test_owner_export_includes_own_private_entries(service):
    ids = await _seed(service)

    service.export_data(service.config.import_export_base_dir / "owner.jsonl", "jsonl", OWNER)

    lines = (service.config.import_export_base_dir / "owner.jsonl").read_text(encoding="utf-8").splitlines()
    assert {json.loads(line)["id"] for line in lines} == {ids["private"], ids["shared"]}


@pytest.mark.asyncio
async def test_unsupported_export_format_is_rejected(service):
    with pytest.raises(ValueError, match="Unsupported export format"):
        service.export_data(service.config.import_export_base_dir / "x.csv", "csv", OWNER)
