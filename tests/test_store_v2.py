"""Tests for the v2 store (src/store_v2)."""

from __future__ import annotations

import math
import sqlite3

import pytest

from src.store_v2.schema import GLOBAL_PROJECT, LATEST_VERSION, apply_migrations
from src.store_v2.store import (
    AuditRecord,
    ConflictError,
    Episode,
    Memory,
    MemoryStore,
    NotFoundError,
    StoreError,
)

try:
    import sqlite_vec  # noqa: F401

    HAVE_VEC = True
except ImportError:  # pragma: no cover
    HAVE_VEC = False


def audit(action: str = "write_attempt", actor: str = "agent-a", **payload) -> AuditRecord:
    return AuditRecord(action=action, actor=actor, payload=payload)


@pytest.fixture(params=["scan", "vec"] if HAVE_VEC else ["scan"])
def store(request, tmp_path):
    s = MemoryStore(tmp_path / "v2.db", use_sqlite_vec=(request.param == "vec"))
    assert s.vec_enabled == (request.param == "vec")
    s.ensure_project("prj-a")
    s.ensure_project("prj-b")
    s.ensure_project(GLOBAL_PROJECT)
    yield s
    s.close()


def memory(project="prj-a", statement="Il sistema usa SQLite in WAL.", type="fact", **kw) -> Memory:
    return Memory(project_id=project, type=type, statement=statement, actor="agent-a", **kw)


# -- schema -------------------------------------------------------------------


def test_migrations_are_idempotent_and_versioned(tmp_path):
    conn = sqlite3.connect(tmp_path / "x.db", isolation_level=None)
    assert apply_migrations(conn) == LATEST_VERSION
    assert apply_migrations(conn) == LATEST_VERSION
    assert conn.execute("PRAGMA user_version").fetchone()[0] == LATEST_VERSION
    conn.close()


def test_refuses_a_newer_schema(tmp_path):
    conn = sqlite3.connect(tmp_path / "x.db", isolation_level=None)
    conn.execute(f"PRAGMA user_version = {LATEST_VERSION + 1}")
    with pytest.raises(RuntimeError):
        apply_migrations(conn)
    conn.close()


# -- projects -----------------------------------------------------------------


def test_ensure_project_is_idempotent_and_audited(store):
    before = store.count_audit()
    assert store.ensure_project("prj-c") is True
    assert store.ensure_project("prj-c") is False
    assert store.count_audit() == before + 1
    assert store.get_project("prj-c")["display_name"] == "prj-c"


@pytest.mark.parametrize("bad", ["", "A", "x", "has space", "../etc", "-lead"])
def test_rejects_invalid_project_ids(store, bad):
    with pytest.raises(StoreError):
        store.ensure_project(bad)


def test_writes_to_unknown_project_fail(store):
    with pytest.raises(NotFoundError):
        store.add_memory(memory(project="nope-prj"), audit=audit())


# -- atomicity of writes and audit ---------------------------------------------


def test_memory_and_audit_commit_together(store):
    memory_id, created = store.add_memory(memory(), audit=audit(outcome="accepted"))
    assert created
    rows = store.list_audit(entry_id=memory_id)
    assert len(rows) == 1 and rows[0].payload["outcome"] == "accepted"


def test_failed_write_leaves_no_audit_row(store):
    before = store.count_audit()
    with pytest.raises(NotFoundError):
        store.add_memory(memory(), audit=audit(), sources=[("episode", "missing-episode")])
    assert store.count_audit() == before
    assert store.list_memories("prj-a") == []


def test_audit_failure_rolls_back_the_write(store, monkeypatch):
    def boom(conn, record):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(MemoryStore, "_insert_audit", staticmethod(boom))
    with pytest.raises(sqlite3.OperationalError):
        store.add_memory(memory(), audit=audit())
    monkeypatch.undo()
    assert store.list_memories("prj-a") == []
    assert store.search_text("SQLite", ["prj-a"]) == []


# -- memories -------------------------------------------------------------------


def test_identical_active_memory_is_not_duplicated(store):
    first, created = store.add_memory(memory(), audit=audit())
    second, created_again = store.add_memory(memory(statement="  il sistema USA sqlite in wal. "), audit=audit())
    assert created and not created_again and first == second


def test_memory_types_are_enforced(store):
    with pytest.raises(StoreError):
        store.add_memory(memory(type="assumption"), audit=audit())


def test_supersede_closes_old_memory_and_links_replacement(store):
    old, _ = store.add_memory(memory(statement="Il provider è Charm Hyper.", type="decision"), audit=audit())
    new, _ = store.add_memory(
        memory(statement="Il provider è Fireworks.", type="decision"), audit=audit(), supersedes=old
    )
    closed = store.get_memory(old)
    assert closed["valid_to"] is not None and closed["superseded_by"] == new
    assert [m["id"] for m in store.list_memories("prj-a")] == [new]
    assert {m["id"] for m in store.list_memories("prj-a", active_only=False)} == {old, new}


def test_supersede_twice_conflicts(store):
    old, _ = store.add_memory(memory(), audit=audit())
    store.supersede(old, audit=audit("invalidate"))
    with pytest.raises(ConflictError):
        store.supersede(old, audit=audit("invalidate"))
    assert store.list_audit(action="invalidate", entry_id=old)


def test_closed_memory_does_not_block_a_new_identical_one(store):
    old, _ = store.add_memory(memory(), audit=audit())
    store.supersede(old, audit=audit("invalidate"))
    new, created = store.add_memory(memory(), audit=audit())
    assert created and new != old


def test_cannot_supersede_across_projects(store):
    other, _ = store.add_memory(memory(project="prj-b"), audit=audit())
    with pytest.raises(StoreError):
        store.add_memory(memory(statement="Nuova regola."), audit=audit(), supersedes=other)


def test_sources_must_exist_and_belong_to_project_or_global(store):
    episode_a = store.add_episode(Episode("prj-a", "observation", "Timeout sul job notturno.", "agent-a"), audit=audit())
    episode_b = store.add_episode(Episode("prj-b", "observation", "Altro progetto.", "agent-a"), audit=audit())
    episode_g = store.add_episode(Episode(GLOBAL_PROJECT, "note", "Regola globale.", "agent-a"), audit=audit())

    memory_id, _ = store.add_memory(
        memory(type="lesson", why="tre timeout"), audit=audit(),
        sources=[("episode", episode_a), ("episode", episode_g), ("external", "https://example.com/pr/1")],
    )
    assert {s["ref"] for s in store.get_memory(memory_id)["sources"]} == {episode_a, episode_g, "https://example.com/pr/1"}
    with pytest.raises(StoreError):
        store.add_memory(memory(statement="Altro."), audit=audit(), sources=[("episode", episode_b)])


# -- episodes ---------------------------------------------------------------------


def test_repeated_episodes_are_kept_and_counted(store):
    for _ in range(3):
        store.add_episode(Episode("prj-a", "error", "TimeoutError: page.goto", "jobbby"), audit=audit("fast_write"))
    assert store.count_similar_episodes("prj-a", "error", "timeouterror:  page.goto") == 3
    assert len(store.list_episodes("prj-a", event_type="error")) == 3


def test_expiry_removes_only_uncited_expired_episodes(store):
    keep = store.add_episode(
        Episode("prj-a", "note", "Citata da una memoria.", "a", expires_at="2026-01-01T00:00:00+00:00"), audit=audit()
    )
    drop = store.add_episode(
        Episode("prj-a", "note", "Nessuno la cita.", "a", expires_at="2026-01-01T00:00:00+00:00"), audit=audit()
    )
    future = store.add_episode(
        Episode("prj-a", "note", "Non ancora scaduta.", "a", expires_at="2099-01-01T00:00:00+00:00"), audit=audit()
    )
    store.add_memory(memory(type="lesson"), audit=audit(), sources=[("episode", keep)])

    assert store.expire_episodes(now="2026-06-01T00:00:00+00:00") == 1
    assert store.get_episode(drop) is None
    assert store.get_episode(keep) and store.get_episode(future)
    assert store.search_text("nessuno", ["prj-a"]) == []
    assert store.list_audit(action="expire_episodes")[0].payload["count"] == 1


# -- full-text search ---------------------------------------------------------------


def test_text_search_finds_exact_identifiers(store):
    hit_id = store.add_episode(
        Episode("prj-a", "error", "TS-4471: superato MaximumIterationsPerRequest=10 su 40 eventi.", "yesod"),
        audit=audit(),
    )
    store.add_episode(Episode("prj-a", "note", "Il deploy è andato bene.", "yesod"), audit=audit())
    for query in ("TS-4471", "MaximumIterationsPerRequest", "maximumiterationsperrequest"):
        hits = store.search_text(query, ["prj-a"])
        assert hits and hits[0].id == hit_id, query


def test_text_search_is_scoped_and_hides_closed_memories(store):
    a, _ = store.add_memory(memory(statement="Jobbby usa PostgreSQL."), audit=audit())
    store.add_memory(memory(project="prj-b", statement="Questo progetto usa PostgreSQL 16."), audit=audit())
    g, _ = store.add_memory(memory(project=GLOBAL_PROJECT, statement="Mai PostgreSQL senza backup."), audit=audit())

    ids = {hit.id for hit in store.search_text("PostgreSQL", ["prj-a", GLOBAL_PROJECT])}
    assert ids == {a, g}
    store.supersede(a, audit=audit("invalidate"))
    assert {hit.id for hit in store.search_text("PostgreSQL", ["prj-a"])} == set()
    assert {hit.id for hit in store.search_text("PostgreSQL", ["prj-a"], include_closed=True)} == {a}


@pytest.mark.parametrize("query", ['"', "AND OR NOT", "foo*", "a:b", "NEAR(", "' OR 1=1 --", ""])
def test_text_search_survives_fts_syntax(store, query):
    store.add_memory(memory(), audit=audit())
    store.search_text(query, ["prj-a"])  # must not raise


def test_text_search_ignores_accents(store):
    memory_id, _ = store.add_memory(memory(statement="La qualità è più importante della velocità."), audit=audit())
    assert store.search_text("qualita velocita", ["prj-a"])[0].id == memory_id


# -- vectors -------------------------------------------------------------------------


def unit(*values: float) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def test_vector_search_ranks_by_cosine_within_scope(store):
    version = store.register_embedding_version(provider="test", model="m", dim=3)
    near, _ = store.add_memory(memory(statement="vicino"), audit=audit())
    far, _ = store.add_memory(memory(statement="lontano"), audit=audit())
    other, _ = store.add_memory(memory(project="prj-b", statement="altro progetto"), audit=audit())
    store.put_embedding("memory", near, version, [1.0, 0.1, 0.0])
    store.put_embedding("memory", far, version, [0.0, 1.0, 0.0])
    store.put_embedding("memory", other, version, [1.0, 0.0, 0.0])

    hits = store.search_vector([1.0, 0.0, 0.0], ["prj-a"], version)
    assert [hit.id for hit in hits] == [near, far]
    assert hits[0].score == pytest.approx(unit(1.0, 0.1, 0.0)[0], abs=1e-5)


def test_vector_search_skips_closed_memories_and_filters_kinds(store):
    version = store.register_embedding_version(provider="test", model="m", dim=2)
    closed, _ = store.add_memory(memory(statement="vecchia"), audit=audit())
    episode = store.add_episode(Episode("prj-a", "note", "nota", "a"), audit=audit())
    store.put_embedding("memory", closed, version, [1.0, 0.0])
    store.put_embedding("episode", episode, version, [1.0, 0.0])
    store.supersede(closed, audit=audit("invalidate"))

    assert [hit.id for hit in store.search_vector([1.0, 0.0], ["prj-a"], version)] == [episode]
    assert store.search_vector([1.0, 0.0], ["prj-a"], version, kinds=["memory"]) == []


def test_put_embedding_validates_dimension_and_zero_vectors(store):
    version = store.register_embedding_version(provider="test", model="m", dim=2)
    memory_id, _ = store.add_memory(memory(), audit=audit())
    with pytest.raises(StoreError):
        store.put_embedding("memory", memory_id, version, [1.0, 0.0, 0.0])
    with pytest.raises(StoreError):
        store.put_embedding("memory", memory_id, version, [0.0, 0.0])


def test_missing_embeddings_and_version_switch(store):
    v1 = store.register_embedding_version(provider="test", model="a", dim=2)
    memory_id, _ = store.add_memory(memory(statement="Decisione", why="perché sì", type="decision"), audit=audit())
    episode = store.add_episode(Episode("prj-a", "note", "nota", "a"), audit=audit())
    assert {(k, i) for k, i, _ in store.missing_embeddings(v1)} == {("memory", memory_id), ("episode", episode)}
    assert "perché sì" in dict(((k, i), t) for k, i, t in store.missing_embeddings(v1))[("memory", memory_id)]

    store.put_embedding("memory", memory_id, v1, [1.0, 0.0])
    store.put_embedding("episode", episode, v1, [0.0, 1.0])
    assert store.missing_embeddings(v1) == []

    v2 = store.register_embedding_version(provider="test", model="b", dim=3)
    assert store.active_embedding_version()["version_id"] == v2
    assert len(store.missing_embeddings(v2)) == 2


def test_vector_index_can_be_rebuilt(store):
    version = store.register_embedding_version(provider="test", model="m", dim=2)
    memory_id, _ = store.add_memory(memory(), audit=audit())
    store.put_embedding("memory", memory_id, version, [1.0, 0.0])
    rebuilt = store.rebuild_vector_index(version)
    assert rebuilt == (1 if store.vec_enabled else 0)
    assert store.search_vector([1.0, 0.0], ["prj-a"], version)[0].id == memory_id


# -- audit ---------------------------------------------------------------------------


def test_audit_after_id_pages_forward_and_default_is_newest_first(store):
    ids = [store.add_audit(audit(n=i)) for i in range(5)]
    assert [r.id for r in store.list_audit(after_id=ids[1], limit=2)] == ids[2:4]
    newest = [r.id for r in store.list_audit(limit=500)]
    assert newest == sorted(newest, reverse=True)
