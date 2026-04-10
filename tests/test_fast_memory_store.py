from __future__ import annotations

from pathlib import Path

from src.models import FastMemoryDistillationStatus, FastMemoryEntry
from src.storage.sqlite_store import SQLiteMemoryStore


def test_fast_memory_store_roundtrip(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    entry = FastMemoryEntry(
        workspace_id="ws-test",
        project_id="project-alpha",
        agent_id="agent-fast",
        user_id="user-fast",
        session_id="session-1",
        event_type="debug_note",
        content="Il fix del parser richiede ancora una verifica sul payload reale.",
        context="debug attempt",
        tags=["debug", "parser"],
        metadata={"ticket": "MEM-101"},
        source="mcp",
    )

    store.add_fast_entry(entry)
    loaded = store.get_fast_entry(entry.id)

    assert loaded is not None
    assert loaded.id == entry.id
    assert loaded.workspace_id == "ws-test"
    assert loaded.project_id == "project-alpha"
    assert loaded.event_type == "debug_note"
    assert loaded.tags == ["debug", "parser"]
    assert loaded.metadata["ticket"] == "MEM-101"
    assert loaded.distillation_status == FastMemoryDistillationStatus.PENDING


def test_fast_memory_store_filters_and_counts(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.add_fast_entry(
        FastMemoryEntry(
            workspace_id="ws-test",
            project_id="project-alpha",
            agent_id="agent-a",
            event_type="incident",
            content="Errore ricorrente nella serializzazione.",
            resolved=False,
        )
    )
    store.add_fast_entry(
        FastMemoryEntry(
            workspace_id="ws-test",
            project_id="project-alpha",
            agent_id="agent-a",
            event_type="fix_attempt",
            content="Tentativo di fix con normalizzazione chiavi.",
            resolved=True,
            distillation_status=FastMemoryDistillationStatus.SUMMARIZED,
        )
    )
    store.add_fast_entry(
        FastMemoryEntry(
            workspace_id="ws-test",
            project_id="project-beta",
            agent_id="agent-b",
            event_type="incident",
            content="Timeout lato import batch.",
            resolved=False,
        )
    )

    incident_rows = store.list_fast_entries(
        workspace_id="ws-test",
        project_id="project-alpha",
        event_type="incident",
        limit=10,
    )
    resolved_count = store.count_fast_entries(
        workspace_id="ws-test",
        resolved=True,
    )
    summarized_count = store.count_fast_entries(
        workspace_id="ws-test",
        distillation_status=FastMemoryDistillationStatus.SUMMARIZED,
    )

    assert len(incident_rows) == 1
    assert incident_rows[0].project_id == "project-alpha"
    assert incident_rows[0].event_type == "incident"
    assert resolved_count == 1
    assert summarized_count == 1


def test_fast_memory_store_update_persists_changes(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    entry = FastMemoryEntry(
        workspace_id="ws-test",
        project_id="project-alpha",
        agent_id="agent-a",
        event_type="fix_attempt",
        content="Prima iterazione del fix.",
    )
    store.add_fast_entry(entry)

    entry.resolved = True
    entry.distillation_status = FastMemoryDistillationStatus.PROMOTED
    entry.recurrence_count = 3
    entry.selection_score = 0.74
    entry.cluster_id = "cluster-1"
    entry.metadata["promotion_candidate"] = True
    store.update_fast_entry(entry)

    updated = store.get_fast_entry(entry.id)

    assert updated is not None
    assert updated.resolved is True
    assert updated.distillation_status == FastMemoryDistillationStatus.PROMOTED
    assert updated.recurrence_count == 3
    assert updated.selection_score == 0.74
    assert updated.cluster_id == "cluster-1"
    assert updated.metadata["promotion_candidate"] is True
