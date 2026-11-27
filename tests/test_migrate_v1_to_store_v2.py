"""Migration of a real v1 database (written by the v1 service) to the v2 store."""

from __future__ import annotations

import pytest

from src.bootstrap import build_runtime
from src.config import Config
from src.service.memory_service import ActorContext
from src.store_v2.migrate_v1 import main, migrate
from src.store_v2.schema import GLOBAL_PROJECT
from src.store_v2.store import AuditRecord, Memory, MemoryStore

PROJECT = "prj-mig"


@pytest.fixture
def v1_db(tmp_path):
    config = Config(
        sqlite_db_path=tmp_path / "v1.db",
        import_export_base_dir=tmp_path / "exchange",
        embedding_provider="hash-local",
        embedding_model="local-hash-migration",
        embedding_dim=64,
        allow_outbound_network=False,
        default_workspace_id="default",
        default_project_id=PROJECT,
        self_eval_enforced=False,
        multi_project_enabled=True,
    )
    runtime = build_runtime(config)
    service = runtime.service
    actor = ActorContext(agent_id="agent-v1", user_id=None, workspace_id="default", project_id=PROJECT)
    service.create_project(actor=actor, project_id=PROJECT, display_name="Progetto migrato")
    return tmp_path / "v1.db", service, actor


async def _populate(service, actor):
    kept = await service.add(
        {"agent_id": "agent-v1", "content": "Il sistema usa PostgreSQL 16 per la fatturazione.", "type": "fact", "tags": ["db"]}, actor
    )
    decision = await service.add(
        {"agent_id": "agent-v1", "content": "Il progetto usa sempre ntfy per le notifiche urgenti.", "type": "decision"}, actor
    )
    assumption = await service.add(
        {"agent_id": "agent-v1", "content": "Il sistema richiede che il backup finisca entro le sei.", "type": "assumption"}, actor
    )
    gone = await service.add({"agent_id": "agent-v1", "content": "Il sistema usa Redis come cache delle sessioni.", "type": "fact"}, actor)
    service.invalidate([gone["entry_id"]], actor, reason="Redis rimosso")
    global_entry = await service.add(
        {
            "agent_id": "agent-v1",
            "content": "Il sistema deve usare dipendenze con almeno 45 giorni.",
            "type": "fact",
            "visibility": "global",
            "scope": {"scope_level": "global"},
        },
        actor,
    )
    fast = service.log_fast(
        {"agent_id": "agent-v1", "content": "Il job notturno ha avuto tre timeout di fila.", "event_type": "observation", "tags": ["run"]},
        actor,
    )
    return {
        "kept": kept["entry_id"],
        "decision": decision["entry_id"],
        "assumption": assumption["entry_id"],
        "gone": gone["entry_id"],
        "global": global_entry["entry_id"],
        "fast": fast["entry_id"],
    }


async def test_migration_preserves_ids_state_and_audit(v1_db, tmp_path):
    path, service, actor = v1_db
    ids = await _populate(service, actor)
    v1_audit = service.store.list_audit(limit=1000)
    v1_max_id = max(a.id for a in v1_audit)

    report = migrate(path, tmp_path / "v2.db")
    assert report["memories"] == 5
    assert report["episodes"] == 1
    assert report["invalidation_markers_skipped"] == 1
    assert report["audit_rows"] == len(v1_audit)
    assert report["max_audit_id"] == v1_max_id

    store = MemoryStore(tmp_path / "v2.db", use_sqlite_vec=False)
    try:
        kept = store.get_memory(ids["kept"])
        assert kept["project_id"] == PROJECT
        assert kept["statement"] == "Il sistema usa PostgreSQL 16 per la fatturazione."
        assert kept["source"] == "migrated" and kept["actor"] == "agent-v1"
        assert "db" in kept["tags"] and kept["valid_to"] is None
        assert store.get_memory(ids["decision"])["type"] == "decision"

        assumption = store.get_memory(ids["assumption"])
        assert assumption["type"] == "fact" and "v1-type:assumption" in assumption["tags"]

        gone = store.get_memory(ids["gone"])
        assert gone["valid_to"] is not None
        assert ids["gone"] not in {m["id"] for m in store.list_memories(PROJECT)}

        global_entry = store.get_memory(ids["global"])
        assert global_entry["project_id"] == GLOBAL_PROJECT and "v1-scope:global" in global_entry["tags"]

        episode = store.get_episode(ids["fast"])
        assert episode["project_id"] == PROJECT and episode["event_type"] == "observation"
        assert episode["expires_at"] is None and "v1_distillation_status" in episode["metadata"]

        v1_project = service.store.get_project("default", PROJECT)
        assert store.get_project(PROJECT)["display_name"] == v1_project.display_name

        # audit ids identical, and new rows continue after the old ones (Yesod's after_id cursor)
        v2_audit = {a.id: a for a in store.list_audit(limit=1000)}
        assert set(v2_audit) == {a.id for a in v1_audit}
        for row in v1_audit:
            assert v2_audit[row.id].action == row.action and v2_audit[row.id].payload == row.payload
        new_id = store.add_audit(AuditRecord(action="write_attempt", actor="v2"))
        assert new_id == v1_max_id + 1
        assert [a.id for a in store.list_audit(after_id=v1_max_id)] == [new_id]

        # migrated text is searchable; closed memories stay hidden by default
        assert store.search_text("PostgreSQL", [PROJECT])[0].id == ids["kept"]
        assert store.search_text("Redis", [PROJECT]) == []
        assert store.search_text("timeout", [PROJECT])[0].id == ids["fast"]

        # nothing has a vector yet: everything is queued for the v2 embedding provider
        version = store.register_embedding_version(provider="test", model="m", dim=4)
        assert len(store.missing_embeddings(version, limit=100)) == 6

        # a new identical memory is a duplicate of the migrated active one
        _, created = store.add_memory(
            Memory(project_id=PROJECT, type="fact", statement="Il sistema usa PostgreSQL 16 per la fatturazione.",
                   actor="v2"),
            audit=AuditRecord(action="write_attempt", actor="v2"),
        )
        assert created is False
    finally:
        store.close()


async def test_dry_run_writes_nothing(v1_db, tmp_path):
    path, service, actor = v1_db
    await _populate(service, actor)
    report = migrate(path, tmp_path / "v2.db", dry_run=True)
    assert report["dry_run"] and report["memories"] == 5
    assert not (tmp_path / "v2.db").exists()


async def test_refuses_non_empty_target_and_non_v1_source(v1_db, tmp_path):
    path, service, actor = v1_db
    await _populate(service, actor)
    migrate(path, tmp_path / "v2.db")
    with pytest.raises(ValueError, match="not empty"):
        migrate(path, tmp_path / "v2.db")
    with pytest.raises(ValueError, match="v1"):
        migrate(tmp_path / "v2.db", tmp_path / "other.db")
    with pytest.raises(FileNotFoundError):
        migrate(tmp_path / "missing.db", tmp_path / "other.db")


async def test_cli(v1_db, tmp_path, capsys):
    path, service, actor = v1_db
    await _populate(service, actor)
    assert main(["--from", str(path), "--dry-run"]) == 0
    assert '"memories": 5' in capsys.readouterr().out


async def test_runtime_migrates_v1_once_at_first_start(v1_db, tmp_path):
    from src.store_v2.runtime import RuntimeSettings, build_runtime

    path, service, actor = v1_db
    ids = await _populate(service, actor)
    settings = RuntimeSettings(store_path=tmp_path / "auto-v2.db", v1_path=path, embedding=None,
                               use_sqlite_vec=False)
    runtime = build_runtime(settings)
    try:
        assert runtime.migration_report["memories"] == 5
        assert runtime.store.get_memory(ids["kept"]) is not None
    finally:
        runtime.close()
    second = build_runtime(settings)  # the v2 file exists now: no second migration
    try:
        assert second.migration_report is None
    finally:
        second.close()
