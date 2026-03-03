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
    MemoryEntry,
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

                CREATE INDEX IF NOT EXISTS idx_entries_scope
                    ON entries(workspace_id, project_id, status, tier, updated_at);

                CREATE INDEX IF NOT EXISTS idx_entries_hash_scope
                    ON entries(workspace_id, project_id, content_hash);

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

    @staticmethod
    def _to_iso(value) -> str:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    def add_entry(self, entry: MemoryEntry) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO entries (
                    id, tier, workspace_id, project_id, user_id, agent_id, visibility, source,
                    type, status, content, context, tags_json, sensitivity_tags_json,
                    metadata_json, confidence, content_hash, created_at, updated_at,
                    embedding_version_id, encrypted, redacted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.tier.value,
                    entry.scope.workspace_id,
                    entry.scope.project_id,
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
                SET tier=?, workspace_id=?, project_id=?, user_id=?, agent_id=?, visibility=?,
                    source=?, type=?, status=?, content=?, context=?, tags_json=?,
                    sensitivity_tags_json=?, metadata_json=?, confidence=?, content_hash=?,
                    updated_at=?, embedding_version_id=?, encrypted=?, redacted=?
                WHERE id=?
                """,
                (
                    entry.tier.value,
                    entry.scope.workspace_id,
                    entry.scope.project_id,
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
                WHERE workspace_id = ? AND project_id = ? AND content_hash = ?
                  AND status != ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (
                    scope.workspace_id,
                    scope.project_id,
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
        scope: ScopeRef,
        include_invalidated: bool = False,
    ) -> list[tuple[MemoryEntry, list[float]]]:
        sql = [
            """
            SELECT e.*, em.vector_json
            FROM entries e
            INNER JOIN embeddings em ON em.entry_id = e.id
            WHERE em.version_id = ?
              AND e.workspace_id = ?
              AND e.project_id = ?
            """,
        ]
        params: list = [version_id, scope.workspace_id, scope.project_id]

        if not include_invalidated:
            sql.append("AND e.status != ?")
            params.append(EntryStatus.INVALIDATED.value)

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

    def list_audit(self, entry_id: Optional[str] = None, limit: int = 200) -> list[AuditEvent]:
        with self._conn() as conn:
            if entry_id:
                rows = conn.execute(
                    """
                    SELECT * FROM audit_log
                    WHERE entry_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (entry_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

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
