"""Integration tests end-to-end v2."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.service.memory_service import ActorContext


@pytest.mark.asyncio
async def test_invalidation_precedence(service):
    actor = ActorContext(agent_id="agent-inv", user_id="user-inv", workspace_id="ws-test", project_id="prj-test")

    add_result = await service.add(
        {
            "content": "Assunzione iniziale: il servizio usa provider esterno.",
            "context": "assumption",
            "agent_id": actor.agent_id,
            "tier": "tier-2",
            "type": "assumption",
            "visibility": "shared",
        },
        actor,
    )
    entry_id = add_result["entry_id"]

    before = await service.search("provider esterno", actor, limit=5)
    assert any(bundle.entry_id == entry_id for bundle in before)

    inv = service.invalidate(target_ids=[entry_id], actor=actor, reason="Smentita: sistema locale-only")
    assert inv["count"] == 1

    after = await service.search("provider esterno", actor, limit=5, include_invalidated=False)
    assert all(bundle.entry_id != entry_id for bundle in after)


@pytest.mark.asyncio
async def test_import_export_memory_md_deterministic(service, tmp_path: Path):
    actor = ActorContext(agent_id="agent-io", user_id="user-io", workspace_id="ws-test", project_id="prj-test")

    await service.add(
        {
            "content": "Fatto stabile: la memoria e locale.",
            "context": "stable",
            "agent_id": actor.agent_id,
            "tier": "tier-3",
            "type": "fact",
            "visibility": "shared",
        },
        actor,
    )

    md_path = service.config.import_export_base_dir / "memory.md"
    export_result = service.export_data(md_path, "memory.md", actor)
    assert export_result.count >= 1

    exported_once = md_path.read_text(encoding="utf-8")
    export_result_2 = service.export_data(md_path, "memory.md", actor)
    assert export_result_2.count == export_result.count
    exported_twice = md_path.read_text(encoding="utf-8")

    assert exported_once == exported_twice


@pytest.mark.asyncio
async def test_reembed_query_consistency(service):
    actor = ActorContext(agent_id="agent-search", user_id="user-search", workspace_id="ws-test", project_id="prj-test")

    await service.add(
        {
            "content": "SQLite e il backend metadata predefinito.",
            "context": "architecture",
            "agent_id": actor.agent_id,
            "tier": "tier-2",
            "type": "fact",
            "visibility": "shared",
        },
        actor,
    )

    before = await service.search("backend metadata", actor, limit=3)
    assert len(before) >= 1

    await service.reembed(actor=actor, model_id="local-hash-v3", activate=True)
    after = await service.search("backend metadata", actor, limit=3)
    assert len(after) >= 1
    assert before[0].entry_id == after[0].entry_id


@pytest.mark.asyncio
async def test_import_jsonl(service, tmp_path: Path):
    actor = ActorContext(agent_id="agent-jsonl", user_id="user-jsonl", workspace_id="ws-test", project_id="prj-test")

    jsonl_path = service.config.import_export_base_dir / "memory.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        '{"id":"e-1","tier":"tier-2","scope":{"workspace_id":"ws-test","project_id":"prj-test","user_id":"user-jsonl","agent_id":"agent-jsonl"},"visibility":"shared","source":"test","type":"fact","status":"active","content":"JSONL import test","context":"import","tags":[],"sensitivity_tags":[],"metadata":{},"links":[],"confidence":0.7,"created_at":"2026-01-01T00:00:00+00:00","updated_at":"2026-01-01T00:00:00+00:00","content_hash":"abc","embedding_version_id":null,"encrypted":false,"redacted":false}\n',
        encoding="utf-8",
    )

    result = await service.import_data(jsonl_path, "jsonl", actor)
    assert result.imported == 1

    entries = service.list_entries(actor, limit=10)
    assert any(entry.id == "e-1" for entry in entries)


@pytest.mark.asyncio
async def test_import_jsonl_rejects_unsupported_top_level_fields(service):
    actor = ActorContext(agent_id="agent-jsonl", user_id="user-jsonl", workspace_id="ws-test", project_id="prj-test")

    jsonl_path = service.config.import_export_base_dir / "memory-extra.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        '{"id":"e-unsupported","content":"JSONL import test","context":"import","scope":{"workspace_id":"ws-test","project_id":"prj-test"},"visibility":"shared","extra_field":"boom"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported fields: extra_field"):
        await service.import_data(jsonl_path, "jsonl", actor)


@pytest.mark.asyncio
async def test_import_jsonl_sanitizes_metadata_keys(service):
    actor = ActorContext(agent_id="agent-jsonl", user_id="user-jsonl", workspace_id="ws-test", project_id="prj-test")

    jsonl_path = service.config.import_export_base_dir / "memory-metadata.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        '{"id":"e-2","tier":"tier-2","scope":{"workspace_id":"ws-test","project_id":"prj-test","user_id":"user-jsonl","agent_id":"agent-jsonl"},"visibility":"shared","source":"test","type":"fact","status":"active","content":"JSONL metadata sanitize test","context":"import","tags":[],"sensitivity_tags":[],"metadata":{"password":"secret","safe":"ok"},"links":[],"confidence":0.7,"created_at":"2026-01-01T00:00:00+00:00","updated_at":"2026-01-01T00:00:00+00:00","content_hash":"def","embedding_version_id":null,"encrypted":false,"redacted":false}\n',
        encoding="utf-8",
    )

    result = await service.import_data(jsonl_path, "jsonl", actor)
    assert result.imported == 1

    entry = service.get("e-2", actor)
    assert entry is not None
    assert entry.metadata == {"safe": "ok"}


@pytest.mark.asyncio
async def test_export_rejects_paths_outside_exchange_base(service, tmp_path: Path):
    actor = ActorContext(agent_id="agent-escape", user_id="user-escape", workspace_id="ws-test", project_id="prj-test")

    outside_path = tmp_path / ".." / "escape.md"

    with pytest.raises(ValueError, match="escapes configured import/export base directory"):
        service.export_data(outside_path, "memory.md", actor)


@pytest.mark.asyncio
async def test_import_rejects_paths_outside_exchange_base(service, tmp_path: Path):
    actor = ActorContext(agent_id="agent-escape", user_id="user-escape", workspace_id="ws-test", project_id="prj-test")

    outside_path = tmp_path / ".." / "escape.jsonl"
    outside_path.parent.mkdir(parents=True, exist_ok=True)
    outside_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes configured import/export base directory"):
        await service.import_data(outside_path, "jsonl", actor)
