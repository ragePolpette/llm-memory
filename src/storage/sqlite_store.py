"""Storage metadata locale basato su SQLite."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

from ..config import Tier
from ..models import (
    AuditEvent,
    EmbeddingVersion,
    EntryLink,
    EntryStatus,
    EntryType,
    FastMemoryDistillationRun,
    FastMemoryDistillationRunStatus,
    FastMemoryDistillationStatus,
    FastMemoryEntry,
    MemoryEntry,
    ProjectRecord,
    ScopeRef,
)


class SQLiteMemoryStore:
    """Store persistente locale per metadata, links, audit e embeddings."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def backup_to(self, target_path: Path) -> None:
        target = Path(target_path)
        if target.exists():
            target.unlink()

        source_conn = sqlite3.connect(self.db_path)
        target_conn = sqlite3.connect(target)
        try:
            source_conn.backup(target_conn)
            target_conn.commit()
        finally:
            target_conn.close()
            source_conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    scope_level TEXT NOT NULL DEFAULT 'project',
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
                );

                CREATE TABLE IF NOT EXISTS projects (
                    workspace_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(workspace_id, project_id)
                );

                CREATE INDEX IF NOT EXISTS idx_projects_workspace
                    ON projects(workspace_id, project_id);

                CREATE TABLE IF NOT EXISTS links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(entry_id) REFERENCES entries(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_links_entry ON links(entry_id);
                CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_id);

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_audit_entry ON audit_log(entry_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS fast_memory_entries (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    context TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    resolved INTEGER NOT NULL DEFAULT 0,
                    distillation_status TEXT NOT NULL,
                    distilled_at TEXT,
                    cluster_id TEXT,
                    recurrence_count INTEGER NOT NULL DEFAULT 1,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    selection_score REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_fast_memory_scope
                    ON fast_memory_entries(workspace_id, project_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_fast_memory_status
                    ON fast_memory_entries(distillation_status, resolved, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_fast_memory_event
                    ON fast_memory_entries(event_type, agent_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS fast_memory_distillation_runs (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    user_id TEXT,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    cluster_ids_json TEXT NOT NULL,
                    source_entry_ids_json TEXT NOT NULL,
                    prepared_payload_json TEXT NOT NULL,
                    agent_output_payload_json TEXT NOT NULL,
                    apply_result_payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    prepared_at TEXT,
                    reviewed_at TEXT,
                    applied_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_fast_memory_runs_scope
                    ON fast_memory_distillation_runs(workspace_id, project_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_fast_memory_runs_status
                    ON fast_memory_distillation_runs(status, created_at DESC);

                CREATE TABLE IF NOT EXISTS embedding_versions (
                    version_id TEXT PRIMARY KEY,
                    provider_id TEXT NOT NULL,
                    embedding_model_id TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    entry_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(entry_id, version_id),
                    FOREIGN KEY(entry_id) REFERENCES entries(id) ON DELETE CASCADE,
                    FOREIGN KEY(version_id) REFERENCES embedding_versions(version_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_embeddings_version ON embeddings(version_id);
                """
            )
            self._migrate_legacy_schema(conn)
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_entries_scope
                    ON entries(workspace_id, project_id, scope_level, status, tier, updated_at);

                CREATE INDEX IF NOT EXISTS idx_entries_hash_scope
                    ON entries(workspace_id, project_id, content_hash);
                """
            )

    @staticmethod
    def _table_has_column(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return any(str(row["name"]) == column_name for row in rows)

    def _migrate_legacy_schema(self, conn: sqlite3.Connection) -> None:
        if not self._table_has_column(conn, "entries", "scope_level"):
            conn.execute(
                "ALTER TABLE entries ADD COLUMN scope_level TEXT NOT NULL DEFAULT 'project'"
            )

    @staticmethod
    def _to_iso(value) -> str:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    def upsert_project(self, project: ProjectRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    workspace_id, project_id, display_name, description,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id, project_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    description=excluded.description,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    project.workspace_id,
                    project.project_id,
                    project.display_name,
                    project.description,
                    json.dumps(project.metadata, ensure_ascii=True, sort_keys=True),
                    self._to_iso(project.created_at),
                    self._to_iso(project.updated_at),
                ),
            )

    def get_project(self, workspace_id: str, project_id: str) -> Optional[ProjectRecord]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM projects
                WHERE workspace_id = ? AND project_id = ?
                """,
                (workspace_id, project_id),
            ).fetchone()
            if row is None:
                return None
            return self._project_from_row(row)

    def list_projects(self, workspace_id: Optional[str] = None) -> list[ProjectRecord]:
        with self._conn() as conn:
            if workspace_id is None:
                rows = conn.execute(
                    "SELECT * FROM projects ORDER BY workspace_id ASC, project_id ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM projects
                    WHERE workspace_id = ?
                    ORDER BY project_id ASC
                    """,
                    (workspace_id,),
                ).fetchall()
            return [self._project_from_row(row) for row in rows]

    def add_entry(self, entry: MemoryEntry) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO entries (
                    id, tier, workspace_id, project_id, scope_level, user_id, agent_id, visibility, source,
                    type, status, content, context, tags_json, sensitivity_tags_json,
                    metadata_json, confidence, content_hash, created_at, updated_at,
                    embedding_version_id, encrypted, redacted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.tier.value,
                    entry.scope.workspace_id,
                    entry.scope.project_id,
                    entry.scope.scope_level.value,
                    entry.scope.user_id,
                    entry.scope.agent_id,
                    entry.visibility.value,
                    entry.source,
                    entry.type.value,
                    entry.status.value,
                    entry.content,
                    entry.context,
                    json.dumps(entry.tags, ensure_ascii=True),
                    json.dumps(entry.sensitivity_tags, ensure_ascii=True),
                    json.dumps(entry.metadata, ensure_ascii=True, sort_keys=True),
                    float(entry.confidence),
                    entry.content_hash,
                    self._to_iso(entry.created_at),
                    self._to_iso(entry.updated_at),
                    entry.embedding_version_id,
                    int(entry.encrypted),
                    int(entry.redacted),
                ),
            )
            self._replace_links(conn, entry.id, entry.links)

    def update_entry(self, entry: MemoryEntry) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE entries
                SET tier=?, workspace_id=?, project_id=?, scope_level=?, user_id=?, agent_id=?, visibility=?,
                    source=?, type=?, status=?, content=?, context=?, tags_json=?,
                    sensitivity_tags_json=?, metadata_json=?, confidence=?, content_hash=?,
                    updated_at=?, embedding_version_id=?, encrypted=?, redacted=?
                WHERE id=?
                """,
                (
                    entry.tier.value,
                    entry.scope.workspace_id,
                    entry.scope.project_id,
                    entry.scope.scope_level.value,
                    entry.scope.user_id,
                    entry.scope.agent_id,
                    entry.visibility.value,
                    entry.source,
                    entry.type.value,
                    entry.status.value,
                    entry.content,
                    entry.context,
                    json.dumps(entry.tags, ensure_ascii=True),
                    json.dumps(entry.sensitivity_tags, ensure_ascii=True),
                    json.dumps(entry.metadata, ensure_ascii=True, sort_keys=True),
                    float(entry.confidence),
                    entry.content_hash,
                    self._to_iso(entry.updated_at),
                    entry.embedding_version_id,
                    int(entry.encrypted),
                    int(entry.redacted),
                    entry.id,
                ),
            )
            self._replace_links(conn, entry.id, entry.links)

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
            if row is None:
                return None
            return self._entry_from_row(conn, row)

    def list_entries(
        self,
        scope: ScopeRef,
        include_invalidated: bool = False,
        limit: int = 50,
        tier: Tier | None = None,
        entry_type: EntryType | None = None,
        visibility: str | None = None,
    ) -> list[MemoryEntry]:
        sql = [
            "SELECT * FROM entries WHERE workspace_id = ? AND project_id = ?",
        ]
        params: list = [scope.workspace_id, scope.project_id]

        if not include_invalidated:
            sql.append("AND status != ?")
            params.append(EntryStatus.INVALIDATED.value)

        if tier is not None:
            sql.append("AND tier = ?")
            params.append(tier.value)

        if entry_type is not None:
            sql.append("AND type = ?")
            params.append(entry_type.value)

        if visibility is not None:
            sql.append("AND visibility = ?")
            params.append(visibility)

        sql.append("ORDER BY updated_at DESC LIMIT ?")
        params.append(limit)

        query = " ".join(sql)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._entry_from_row(conn, row) for row in rows]

    def find_by_hash(self, scope: ScopeRef, content_hash: str) -> Optional[MemoryEntry]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM entries
                WHERE workspace_id = ? AND project_id = ? AND scope_level = ? AND content_hash = ?
                  AND status != ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (
                    scope.workspace_id,
                    scope.project_id,
                    scope.scope_level.value,
                    content_hash,
                    EntryStatus.INVALIDATED.value,
                ),
            ).fetchone()
            if row is None:
                return None
            return self._entry_from_row(conn, row)

    def upsert_embedding_version(self, version: EmbeddingVersion, activate: bool = False) -> None:
        with self._conn() as conn:
            if activate:
                conn.execute("UPDATE embedding_versions SET active = 0")

            conn.execute(
                """
                INSERT INTO embedding_versions (
                    version_id, provider_id, embedding_model_id, dim, fingerprint,
                    config_json, created_at, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(version_id) DO UPDATE SET
                    provider_id=excluded.provider_id,
                    embedding_model_id=excluded.embedding_model_id,
                    dim=excluded.dim,
                    fingerprint=excluded.fingerprint,
                    config_json=excluded.config_json,
                    active=excluded.active
                """,
                (
                    version.version_id,
                    version.provider_id,
                    version.embedding_model_id,
                    version.dim,
                    version.fingerprint,
                    json.dumps(version.config, ensure_ascii=True, sort_keys=True),
                    version.created_at.isoformat(),
                    int(version.active or activate),
                ),
            )

    def set_active_embedding_version(self, version_id: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE embedding_versions SET active = 0")
            conn.execute(
                "UPDATE embedding_versions SET active = 1 WHERE version_id = ?",
                (version_id,),
            )

    def get_embedding_version(self, version_id: str) -> Optional[EmbeddingVersion]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM embedding_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            if row is None:
                return None
            return self._embedding_version_from_row(row)

    def get_active_embedding_version(self) -> Optional[EmbeddingVersion]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM embedding_versions WHERE active = 1 ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return self._embedding_version_from_row(row)

    def list_embedding_versions(self) -> list[EmbeddingVersion]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM embedding_versions ORDER BY created_at DESC"
            ).fetchall()
            return [self._embedding_version_from_row(row) for row in rows]

    def upsert_embedding(self, entry_id: str, version_id: str, vector: list[float], created_at: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO embeddings (entry_id, version_id, vector_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entry_id, version_id)
                DO UPDATE SET vector_json = excluded.vector_json, created_at = excluded.created_at
                """,
                (
                    entry_id,
                    version_id,
                    json.dumps(vector, ensure_ascii=True),
                    created_at,
                ),
            )
            conn.execute(
                "UPDATE entries SET embedding_version_id = ?, updated_at = ? WHERE id = ?",
                (version_id, created_at, entry_id),
            )

    def get_embedding(self, entry_id: str, version_id: str) -> Optional[list[float]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT vector_json FROM embeddings WHERE entry_id = ? AND version_id = ?",
                (entry_id, version_id),
            ).fetchone()
            if row is None:
                return None
            return [float(v) for v in json.loads(row["vector_json"])]

    def list_embeddings(
        self,
        version_id: str,
        scopes: list[ScopeRef] | None = None,
        include_invalidated: bool = False,
        *,
        scope: ScopeRef | None = None,
    ) -> list[tuple[MemoryEntry, list[float]]]:
        if scope is not None:
            scopes = [scope]
        if not scopes:
            return []
        sql = [
            """
            SELECT e.*, em.vector_json
            FROM entries e
            INNER JOIN embeddings em ON em.entry_id = e.id
            WHERE em.version_id = ?
            """,
        ]
        params: list = [version_id]
        scope_predicates: list[str] = []
        for scope in scopes:
            scope_predicates.append(
                "(e.workspace_id = ? AND e.project_id = ? AND e.scope_level = ?)"
            )
            params.extend(
                [
                    scope.workspace_id,
                    scope.project_id,
                    scope.scope_level.value,
                ]
            )
        sql.append("AND (" + " OR ".join(scope_predicates) + ")")

        if not include_invalidated:
            sql.append("AND e.status != ?")
            params.append(EntryStatus.INVALIDATED.value)
            sql.append("AND e.type != ?")
            params.append(EntryType.INVALIDATED.value)

        sql.append("ORDER BY e.updated_at DESC")
        query = " ".join(sql)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            result: list[tuple[MemoryEntry, list[float]]] = []
            for row in rows:
                entry = self._entry_from_row(conn, row)
                vector = [float(v) for v in json.loads(row["vector_json"])]
                result.append((entry, vector))
            return result

    def list_entries_missing_embedding(self, version_id: str, limit: int = 1000) -> list[MemoryEntry]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT e.* FROM entries e
                LEFT JOIN embeddings em
                  ON em.entry_id = e.id AND em.version_id = ?
                WHERE em.entry_id IS NULL
                  AND e.status != ?
                ORDER BY e.updated_at ASC
                LIMIT ?
                """,
                (version_id, EntryStatus.INVALIDATED.value, limit),
            ).fetchall()
            return [self._entry_from_row(conn, row) for row in rows]

    def count_entries_missing_embedding(self, version_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(1) AS c FROM entries e
                LEFT JOIN embeddings em
                  ON em.entry_id = e.id AND em.version_id = ?
                WHERE em.entry_id IS NULL
                  AND e.status != ?
                """,
                (version_id, EntryStatus.INVALIDATED.value),
            ).fetchone()
            return int(row["c"] if row else 0)

    def add_audit(self, event: AuditEvent) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO audit_log (entry_id, action, actor, reason, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.entry_id,
                    event.action,
                    event.actor,
                    event.reason,
                    json.dumps(event.payload, ensure_ascii=True, sort_keys=True),
                    event.created_at.isoformat(),
                ),
            )
            return int(cur.lastrowid)

    def add_fast_entry(self, entry: FastMemoryEntry) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO fast_memory_entries (
                    id, workspace_id, project_id, agent_id, user_id, session_id,
                    event_type, content, context, tags_json, metadata_json, source,
                    resolved, distillation_status, distilled_at, cluster_id,
                    recurrence_count, first_seen_at, last_seen_at,
                    selection_score, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.workspace_id,
                    entry.project_id,
                    entry.agent_id,
                    entry.user_id,
                    entry.session_id,
                    entry.event_type,
                    entry.content,
                    entry.context,
                    json.dumps(entry.tags, ensure_ascii=True),
                    json.dumps(entry.metadata, ensure_ascii=True, sort_keys=True),
                    entry.source,
                    int(entry.resolved),
                    entry.distillation_status.value,
                    self._to_iso(entry.distilled_at) if entry.distilled_at is not None else None,
                    entry.cluster_id,
                    int(entry.recurrence_count),
                    self._to_iso(entry.first_seen_at),
                    self._to_iso(entry.last_seen_at),
                    entry.selection_score,
                    self._to_iso(entry.created_at),
                    self._to_iso(entry.updated_at),
                ),
            )

    def update_fast_entry(self, entry: FastMemoryEntry) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE fast_memory_entries
                SET workspace_id=?, project_id=?, agent_id=?, user_id=?, session_id=?,
                    event_type=?, content=?, context=?, tags_json=?, metadata_json=?, source=?,
                    resolved=?, distillation_status=?, distilled_at=?, cluster_id=?,
                    recurrence_count=?, first_seen_at=?, last_seen_at=?,
                    selection_score=?, updated_at=?
                WHERE id=?
                """,
                (
                    entry.workspace_id,
                    entry.project_id,
                    entry.agent_id,
                    entry.user_id,
                    entry.session_id,
                    entry.event_type,
                    entry.content,
                    entry.context,
                    json.dumps(entry.tags, ensure_ascii=True),
                    json.dumps(entry.metadata, ensure_ascii=True, sort_keys=True),
                    entry.source,
                    int(entry.resolved),
                    entry.distillation_status.value,
                    self._to_iso(entry.distilled_at) if entry.distilled_at is not None else None,
                    entry.cluster_id,
                    int(entry.recurrence_count),
                    self._to_iso(entry.first_seen_at),
                    self._to_iso(entry.last_seen_at),
                    entry.selection_score,
                    self._to_iso(entry.updated_at),
                    entry.id,
                ),
            )

    def get_fast_entry(self, entry_id: str) -> Optional[FastMemoryEntry]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM fast_memory_entries WHERE id = ?",
                (entry_id,),
            ).fetchone()
            if row is None:
                return None
            return self._fast_entry_from_row(row)

    def list_fast_entries(
        self,
        *,
        workspace_id: Optional[str] = None,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        event_type: Optional[str] = None,
        resolved: Optional[bool] = None,
        distillation_status: FastMemoryDistillationStatus | None = None,
        limit: int = 100,
    ) -> list[FastMemoryEntry]:
        sql = ["SELECT * FROM fast_memory_entries WHERE 1 = 1"]
        params: list[object] = []

        if workspace_id is not None:
            sql.append("AND workspace_id = ?")
            params.append(workspace_id)

        if project_id is not None:
            sql.append("AND project_id = ?")
            params.append(project_id)

        if agent_id is not None:
            sql.append("AND agent_id = ?")
            params.append(agent_id)

        if event_type is not None:
            sql.append("AND event_type = ?")
            params.append(event_type)

        if resolved is not None:
            sql.append("AND resolved = ?")
            params.append(int(resolved))

        if distillation_status is not None:
            sql.append("AND distillation_status = ?")
            params.append(distillation_status.value)

        sql.append("ORDER BY created_at DESC, id DESC LIMIT ?")
        params.append(max(1, int(limit)))

        with self._conn() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
            return [self._fast_entry_from_row(row) for row in rows]

    def count_fast_entries(
        self,
        *,
        workspace_id: Optional[str] = None,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        event_type: Optional[str] = None,
        resolved: Optional[bool] = None,
        distillation_status: FastMemoryDistillationStatus | None = None,
    ) -> int:
        sql = ["SELECT COUNT(1) AS c FROM fast_memory_entries WHERE 1 = 1"]
        params: list[object] = []

        if workspace_id is not None:
            sql.append("AND workspace_id = ?")
            params.append(workspace_id)

        if project_id is not None:
            sql.append("AND project_id = ?")
            params.append(project_id)

        if agent_id is not None:
            sql.append("AND agent_id = ?")
            params.append(agent_id)

        if event_type is not None:
            sql.append("AND event_type = ?")
            params.append(event_type)

        if resolved is not None:
            sql.append("AND resolved = ?")
            params.append(int(resolved))

        if distillation_status is not None:
            sql.append("AND distillation_status = ?")
            params.append(distillation_status.value)

        with self._conn() as conn:
            row = conn.execute(" ".join(sql), params).fetchone()
            return int(row["c"] if row else 0)

    def add_fast_distillation_run(self, run: FastMemoryDistillationRun) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO fast_memory_distillation_runs (
                    id, workspace_id, project_id, agent_id, user_id, status, reason,
                    cluster_ids_json, source_entry_ids_json, prepared_payload_json,
                    agent_output_payload_json, apply_result_payload_json,
                    created_at, updated_at, prepared_at, reviewed_at, applied_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.workspace_id,
                    run.project_id,
                    run.agent_id,
                    run.user_id,
                    run.status.value,
                    run.reason,
                    json.dumps(run.cluster_ids, ensure_ascii=True),
                    json.dumps(run.source_entry_ids, ensure_ascii=True),
                    json.dumps(run.prepared_payload, ensure_ascii=True, sort_keys=True),
                    json.dumps(run.agent_output_payload, ensure_ascii=True, sort_keys=True),
                    json.dumps(run.apply_result_payload, ensure_ascii=True, sort_keys=True),
                    self._to_iso(run.created_at),
                    self._to_iso(run.updated_at),
                    self._to_iso(run.prepared_at) if run.prepared_at is not None else None,
                    self._to_iso(run.reviewed_at) if run.reviewed_at is not None else None,
                    self._to_iso(run.applied_at) if run.applied_at is not None else None,
                ),
            )

    def update_fast_distillation_run(self, run: FastMemoryDistillationRun) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE fast_memory_distillation_runs
                SET workspace_id=?, project_id=?, agent_id=?, user_id=?, status=?, reason=?,
                    cluster_ids_json=?, source_entry_ids_json=?, prepared_payload_json=?,
                    agent_output_payload_json=?, apply_result_payload_json=?, updated_at=?,
                    prepared_at=?, reviewed_at=?, applied_at=?
                WHERE id=?
                """,
                (
                    run.workspace_id,
                    run.project_id,
                    run.agent_id,
                    run.user_id,
                    run.status.value,
                    run.reason,
                    json.dumps(run.cluster_ids, ensure_ascii=True),
                    json.dumps(run.source_entry_ids, ensure_ascii=True),
                    json.dumps(run.prepared_payload, ensure_ascii=True, sort_keys=True),
                    json.dumps(run.agent_output_payload, ensure_ascii=True, sort_keys=True),
                    json.dumps(run.apply_result_payload, ensure_ascii=True, sort_keys=True),
                    self._to_iso(run.updated_at),
                    self._to_iso(run.prepared_at) if run.prepared_at is not None else None,
                    self._to_iso(run.reviewed_at) if run.reviewed_at is not None else None,
                    self._to_iso(run.applied_at) if run.applied_at is not None else None,
                    run.id,
                ),
            )

    def get_fast_distillation_run(self, run_id: str) -> Optional[FastMemoryDistillationRun]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM fast_memory_distillation_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return self._fast_distillation_run_from_row(row)

    def list_fast_distillation_runs(
        self,
        *,
        workspace_id: Optional[str] = None,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        status: FastMemoryDistillationRunStatus | None = None,
        limit: int = 50,
    ) -> list[FastMemoryDistillationRun]:
        sql = ["SELECT * FROM fast_memory_distillation_runs WHERE 1 = 1"]
        params: list[object] = []

        if workspace_id is not None:
            sql.append("AND workspace_id = ?")
            params.append(workspace_id)

        if project_id is not None:
            sql.append("AND project_id = ?")
            params.append(project_id)

        if agent_id is not None:
            sql.append("AND agent_id = ?")
            params.append(agent_id)

        if status is not None:
            sql.append("AND status = ?")
            params.append(status.value)

        sql.append("ORDER BY created_at DESC, id DESC LIMIT ?")
        params.append(max(1, int(limit)))

        with self._conn() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
            return [self._fast_distillation_run_from_row(row) for row in rows]

    def count_entries(
        self,
        *,
        status: EntryStatus | None = None,
        entry_type: EntryType | None = None,
        scope_level: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        exclude_invalidated: bool = False,
    ) -> int:
        sql = ["SELECT COUNT(1) AS c FROM entries WHERE 1 = 1"]
        params: list[object] = []

        if workspace_id is not None:
            sql.append("AND workspace_id = ?")
            params.append(workspace_id)

        if project_id is not None:
            sql.append("AND project_id = ?")
            params.append(project_id)

        if scope_level is not None:
            sql.append("AND scope_level = ?")
            params.append(scope_level)

        if status is not None:
            sql.append("AND status = ?")
            params.append(status.value)

        if entry_type is not None:
            sql.append("AND type = ?")
            params.append(entry_type.value)

        if exclude_invalidated:
            sql.append("AND status != ?")
            params.append(EntryStatus.INVALIDATED.value)
            sql.append("AND type != ?")
            params.append(EntryType.INVALIDATED.value)

        with self._conn() as conn:
            row = conn.execute(" ".join(sql), params).fetchone()
            return int(row["c"] if row else 0)

    def count_audit(
        self,
        *,
        entry_id: Optional[str] = None,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        reason: Optional[str] = None,
        since: Optional[str] = None,
    ) -> int:
        sql = ["SELECT COUNT(1) AS c FROM audit_log WHERE 1 = 1"]
        params: list[object] = []

        if entry_id:
            sql.append("AND entry_id = ?")
            params.append(entry_id)

        if action:
            sql.append("AND action = ?")
            params.append(action)

        if actor:
            sql.append("AND actor = ?")
            params.append(actor)

        if reason:
            sql.append("AND reason = ?")
            params.append(reason)

        if since:
            sql.append("AND created_at >= ?")
            params.append(since)

        with self._conn() as conn:
            row = conn.execute(" ".join(sql), params).fetchone()
            return int(row["c"] if row else 0)

    def list_audit(
        self,
        entry_id: Optional[str] = None,
        limit: int = 200,
        *,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        reason: Optional[str] = None,
        since: Optional[str] = None,
    ) -> list[AuditEvent]:
        sql = ["SELECT * FROM audit_log WHERE 1 = 1"]
        params: list[object] = []

        if entry_id:
            sql.append("AND entry_id = ?")
            params.append(entry_id)

        if action:
            sql.append("AND action = ?")
            params.append(action)

        if actor:
            sql.append("AND actor = ?")
            params.append(actor)

        if reason:
            sql.append("AND reason = ?")
            params.append(reason)

        if since:
            sql.append("AND created_at >= ?")
            params.append(since)

        sql.append("ORDER BY created_at DESC LIMIT ?")
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()

            return [
                AuditEvent(
                    id=int(row["id"]),
                    entry_id=row["entry_id"],
                    action=row["action"],
                    actor=row["actor"],
                    reason=row["reason"],
                    payload=json.loads(row["payload_json"]),
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def export_entries(self, scope: Optional[ScopeRef] = None) -> list[MemoryEntry]:
        with self._conn() as conn:
            if scope:
                rows = conn.execute(
                    """
                    SELECT * FROM entries
                    WHERE workspace_id = ? AND project_id = ?
                    ORDER BY created_at ASC, id ASC
                    """,
                    (scope.workspace_id, scope.project_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM entries ORDER BY created_at ASC, id ASC"
                ).fetchall()
            return [self._entry_from_row(conn, row) for row in rows]

    def count_entries_for_scope(self, scope: ScopeRef, *, include_invalidated: bool = False) -> int:
        sql = [
            """
            SELECT COUNT(1) AS c FROM entries
            WHERE workspace_id = ? AND project_id = ? AND scope_level = ?
            """
        ]
        params: list = [scope.workspace_id, scope.project_id, scope.scope_level.value]
        if not include_invalidated:
            sql.append("AND status != ?")
            params.append(EntryStatus.INVALIDATED.value)
            sql.append("AND type != ?")
            params.append(EntryType.INVALIDATED.value)
        with self._conn() as conn:
            row = conn.execute(" ".join(sql), params).fetchone()
            return int(row["c"] if row else 0)

    def _replace_links(self, conn: sqlite3.Connection, entry_id: str, links: Iterable[EntryLink]) -> None:
        conn.execute("DELETE FROM links WHERE entry_id = ?", (entry_id,))
        for link in links:
            conn.execute(
                """
                INSERT INTO links(entry_id, target_id, relation, created_at)
                VALUES (?, ?, ?, datetime('now'))
                """,
                (entry_id, link.target_id, link.relation),
            )

    def _entry_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> MemoryEntry:
        link_rows = conn.execute(
            "SELECT target_id, relation FROM links WHERE entry_id = ? ORDER BY id ASC",
            (row["id"],),
        ).fetchall()

        links = [
            EntryLink(target_id=link_row["target_id"], relation=link_row["relation"])
            for link_row in link_rows
        ]

        return MemoryEntry(
            id=row["id"],
            tier=Tier(row["tier"]),
            scope=ScopeRef(
                workspace_id=row["workspace_id"],
                project_id=row["project_id"],
                scope_level=row["scope_level"],
                user_id=row["user_id"],
                agent_id=row["agent_id"],
            ),
            visibility=row["visibility"],
            source=row["source"],
            type=EntryType(row["type"]),
            status=EntryStatus(row["status"]),
            content=row["content"],
            context=row["context"],
            tags=json.loads(row["tags_json"]),
            sensitivity_tags=json.loads(row["sensitivity_tags_json"]),
            metadata=json.loads(row["metadata_json"]),
            links=links,
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            content_hash=row["content_hash"],
            embedding_version_id=row["embedding_version_id"],
            encrypted=bool(row["encrypted"]),
            redacted=bool(row["redacted"]),
        )

    @staticmethod
    def _fast_entry_from_row(row: sqlite3.Row) -> FastMemoryEntry:
        return FastMemoryEntry(
            id=row["id"],
            workspace_id=row["workspace_id"],
            project_id=row["project_id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            event_type=row["event_type"],
            content=row["content"],
            context=row["context"],
            tags=json.loads(row["tags_json"]),
            metadata=json.loads(row["metadata_json"]),
            source=row["source"],
            resolved=bool(row["resolved"]),
            distillation_status=FastMemoryDistillationStatus(row["distillation_status"]),
            distilled_at=row["distilled_at"],
            cluster_id=row["cluster_id"],
            recurrence_count=int(row["recurrence_count"]),
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            selection_score=float(row["selection_score"]) if row["selection_score"] is not None else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _fast_distillation_run_from_row(row: sqlite3.Row) -> FastMemoryDistillationRun:
        return FastMemoryDistillationRun(
            id=row["id"],
            workspace_id=row["workspace_id"],
            project_id=row["project_id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            status=FastMemoryDistillationRunStatus(row["status"]),
            reason=row["reason"],
            cluster_ids=json.loads(row["cluster_ids_json"]),
            source_entry_ids=json.loads(row["source_entry_ids_json"]),
            prepared_payload=json.loads(row["prepared_payload_json"]),
            agent_output_payload=json.loads(row["agent_output_payload_json"]),
            apply_result_payload=json.loads(row["apply_result_payload_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            prepared_at=row["prepared_at"],
            reviewed_at=row["reviewed_at"],
            applied_at=row["applied_at"],
        )

    @staticmethod
    def _embedding_version_from_row(row: sqlite3.Row) -> EmbeddingVersion:
        return EmbeddingVersion(
            version_id=row["version_id"],
            provider_id=row["provider_id"],
            embedding_model_id=row["embedding_model_id"],
            dim=int(row["dim"]),
            fingerprint=row["fingerprint"],
            config=json.loads(row["config_json"]),
            created_at=row["created_at"],
            active=bool(row["active"]),
        )

    @staticmethod
    def _project_from_row(row: sqlite3.Row) -> ProjectRecord:
        return ProjectRecord(
            workspace_id=row["workspace_id"],
            project_id=row["project_id"],
            display_name=row["display_name"],
            description=row["description"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
