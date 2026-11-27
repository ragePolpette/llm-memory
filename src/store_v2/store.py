"""SQLite store for llm-memory v2: projects, episodes, structured memories, audit.

Design rules:
- Every write and its audit row commit in the same transaction.
- Episodes are immutable; memories are closed (valid_to) rather than deleted.
- Full-text search uses FTS5. Vector search uses sqlite-vec when the extension can be
  loaded, otherwise a brute-force scan over float32 blobs. Both read the same canonical
  `embeddings` table, so the index is an optimisation and never the source of truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional
from uuid import uuid4

from .schema import GLOBAL_PROJECT, apply_migrations

MEMORY_TYPES = ("decision", "lesson", "fact", "constraint", "preference")
_PROJECT_ID = re.compile(r"^(?:__global__|[a-z0-9][a-z0-9._-]{1,63})$")
_FTS_TOKEN = re.compile(r"[\w\-]+", re.UNICODE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(*parts: str) -> str:
    normalized = "\x1f".join(" ".join(part.split()).lower() for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def pack_vector(vector: Iterable[float]) -> bytes:
    values = [float(v) for v in vector]
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class StoreError(Exception):
    """Base error of the v2 store."""


class NotFoundError(StoreError):
    pass


class ConflictError(StoreError):
    pass


@dataclass
class Episode:
    project_id: str
    event_type: str
    content: str
    agent_id: str
    context: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    session_id: Optional[str] = None
    source: str = "agent"
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)
    expires_at: Optional[str] = None


@dataclass
class Memory:
    project_id: str
    type: str
    statement: str
    actor: str
    source: str = "agent"
    why: str = ""
    how: str = ""
    impact: Optional[str] = None
    impact_note: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    confidence: float = 0.5
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)
    updated_at: str = ""
    valid_from: str = ""
    valid_to: Optional[str] = None
    superseded_by: Optional[str] = None
    last_used_at: Optional[str] = None

    def __post_init__(self) -> None:
        self.updated_at = self.updated_at or self.created_at
        self.valid_from = self.valid_from or self.created_at


@dataclass
class AuditRecord:
    action: str
    actor: str
    entry_id: Optional[str] = None
    reason: Optional[str] = None
    payload: dict = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    id: Optional[int] = None


@dataclass
class SearchHit:
    kind: str  # "episode" | "memory"
    id: str
    project_id: str
    score: float


class MemoryStore:
    """Thread-safe access to one v2 database file."""

    BUSY_TIMEOUT_MS = 30_000

    def __init__(self, db_path: Path | str, *, use_sqlite_vec: str | bool = "auto"):
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(db_path), isolation_level=None, check_same_thread=False, timeout=self.BUSY_TIMEOUT_MS / 1000
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA busy_timeout={self.BUSY_TIMEOUT_MS}")
        self._conn.execute("PRAGMA foreign_keys=ON")
        if str(db_path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        apply_migrations(self._conn)
        self.vec_enabled = self._try_load_sqlite_vec(use_sqlite_vec)
        self._vec_tables: set[int] = set()

    # -- infrastructure -------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _try_load_sqlite_vec(self, mode: str | bool) -> bool:
        if mode is False:
            return False
        try:
            import sqlite_vec  # type: ignore[import-not-found]

            self._conn.enable_load_extension(True)
            try:
                sqlite_vec.load(self._conn)
            finally:
                self._conn.enable_load_extension(False)
            return True
        except Exception:
            if mode is True:
                raise
            return False

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def _read(self, sql: str, params: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    @staticmethod
    def _insert_audit(conn: sqlite3.Connection, record: AuditRecord) -> int:
        cursor = conn.execute(
            "INSERT INTO audit_log(entry_id, action, actor, reason, payload_json, created_at) VALUES (?,?,?,?,?,?)",
            (
                record.entry_id,
                record.action,
                record.actor,
                record.reason,
                json.dumps(record.payload, ensure_ascii=False, sort_keys=True),
                record.created_at,
            ),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def validate_project_id(project_id: str) -> str:
        value = str(project_id or "").strip().lower()
        if not _PROJECT_ID.match(value):
            raise StoreError(f"invalid project_id: {project_id!r}")
        return value

    # -- projects -------------------------------------------------------------

    def ensure_project(self, project_id: str, *, display_name: str = "", description: str = "",
                       actor: str = "system", metadata: Optional[dict] = None) -> bool:
        """Create the project if missing. Returns True when it was created."""

        project_id = self.validate_project_id(project_id)
        now = utc_now()
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO projects(project_id, display_name, description, metadata_json, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?)",
                (project_id, display_name or project_id, description, json.dumps(metadata or {}), now, now),
            )
            created = cursor.rowcount == 1
            if created:
                self._insert_audit(conn, AuditRecord(action="create_project", actor=actor, payload={"project_id": project_id}))
        return created

    def get_project(self, project_id: str) -> Optional[dict]:
        rows = self._read("SELECT * FROM projects WHERE project_id = ?", (project_id,))
        return self._project_row(rows[0]) if rows else None

    def list_projects(self) -> list[dict]:
        return [self._project_row(row) for row in self._read("SELECT * FROM projects ORDER BY project_id")]

    @staticmethod
    def _project_row(row: sqlite3.Row) -> dict:
        return {
            "project_id": row["project_id"],
            "display_name": row["display_name"],
            "description": row["description"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _require_project(self, conn: sqlite3.Connection, project_id: str) -> None:
        if not conn.execute("SELECT 1 FROM projects WHERE project_id = ?", (project_id,)).fetchone():
            raise NotFoundError(f"unknown project: {project_id}")

    # -- episodes ---------------------------------------------------------------

    def add_episode(self, episode: Episode, *, audit: AuditRecord) -> str:
        """Insert an episode and its audit row atomically. Returns the episode id.

        Identical episodes are stored again on purpose: a repeated event is a signal.
        `count_similar_episodes` exposes the recurrence.
        """

        episode.project_id = self.validate_project_id(episode.project_id)
        digest = content_hash(episode.event_type, episode.content)
        with self.transaction() as conn:
            self._require_project(conn, episode.project_id)
            conn.execute(
                "INSERT INTO episodes(id, project_id, event_type, content, context, tags_json, metadata_json, agent_id,"
                " session_id, source, content_hash, created_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    episode.id, episode.project_id, episode.event_type, episode.content, episode.context,
                    json.dumps(episode.tags, ensure_ascii=False), json.dumps(episode.metadata, ensure_ascii=False),
                    episode.agent_id, episode.session_id, episode.source, digest, episode.created_at, episode.expires_at,
                ),
            )
            self._index_episode(conn, episode)
            audit.entry_id = audit.entry_id or episode.id
            self._insert_audit(conn, audit)
        return episode.id

    @staticmethod
    def _index_episode(conn: sqlite3.Connection, episode: Episode) -> None:
        cursor = conn.execute("INSERT INTO episodes_fts_map(episode_id) VALUES (?)", (episode.id,))
        conn.execute(
            "INSERT INTO episodes_fts(rowid, content, context, tags) VALUES (?,?,?,?)",
            (cursor.lastrowid, episode.content, episode.context, " ".join(episode.tags)),
        )

    def count_similar_episodes(self, project_id: str, event_type: str, content: str) -> int:
        digest = content_hash(event_type, content)
        return int(self._read(
            "SELECT COUNT(1) AS c FROM episodes WHERE project_id = ? AND content_hash = ?", (project_id, digest)
        )[0]["c"])

    def get_episode(self, episode_id: str) -> Optional[dict]:
        rows = self._read("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        return self._episode_row(rows[0]) if rows else None

    def list_episodes(self, project_id: str, *, event_type: Optional[str] = None, since: Optional[str] = None,
                      limit: int = 50) -> list[dict]:
        sql = ["SELECT * FROM episodes WHERE project_id = ?"]
        params: list = [project_id]
        if event_type:
            sql.append("AND event_type = ?")
            params.append(event_type)
        if since:
            sql.append("AND created_at >= ?")
            params.append(since)
        sql.append("ORDER BY created_at DESC, id LIMIT ?")
        params.append(max(1, min(int(limit), 500)))
        return [self._episode_row(row) for row in self._read(" ".join(sql), params)]

    @staticmethod
    def _episode_row(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "kind": "episode",
            "project_id": row["project_id"],
            "event_type": row["event_type"],
            "content": row["content"],
            "context": row["context"],
            "tags": json.loads(row["tags_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "agent_id": row["agent_id"],
            "session_id": row["session_id"],
            "source": row["source"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
        }

    def expire_episodes(self, *, now: Optional[str] = None, actor: str = "system") -> int:
        """Delete expired episodes that no memory cites. Returns how many were removed."""

        now = now or utc_now()
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT e.id, e.project_id FROM episodes e WHERE e.expires_at IS NOT NULL AND e.expires_at <= ?"
                " AND NOT EXISTS (SELECT 1 FROM memory_sources s WHERE s.source_kind = 'episode' AND s.source_ref = e.id)",
                (now,),
            ).fetchall()
            for row in rows:
                self._delete_episode(conn, row["id"])
            if rows:
                self._insert_audit(
                    conn,
                    AuditRecord(action="expire_episodes", actor=actor,
                                payload={"count": len(rows), "projects": sorted({r["project_id"] for r in rows})}),
                )
        return len(rows)

    @staticmethod
    def _delete_episode(conn: sqlite3.Connection, episode_id: str) -> None:
        mapped = conn.execute("SELECT rowid FROM episodes_fts_map WHERE episode_id = ?", (episode_id,)).fetchone()
        if mapped:
            conn.execute("DELETE FROM episodes_fts WHERE rowid = ?", (mapped["rowid"],))
            conn.execute("DELETE FROM episodes_fts_map WHERE rowid = ?", (mapped["rowid"],))
        conn.execute("DELETE FROM embeddings WHERE item_kind = 'episode' AND item_id = ?", (episode_id,))
        conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))

    # -- memories -----------------------------------------------------------------

    def add_memory(self, memory: Memory, *, audit: AuditRecord, sources: Iterable[tuple[str, str]] = (),
                   supersedes: Optional[str] = None) -> tuple[str, bool]:
        """Insert a memory, its sources and its audit row atomically.

        `sources` are (kind, ref) pairs with kind in episode|memory|external; episode and
        memory refs must exist in the same project or in the global project.
        With `supersedes`, the old memory is closed in the same transaction.
        Returns (id, created); an identical active memory returns its id with created=False.
        """

        memory.project_id = self.validate_project_id(memory.project_id)
        if memory.type not in MEMORY_TYPES:
            raise StoreError(f"invalid memory type: {memory.type!r}")
        digest = content_hash(memory.type, memory.statement)
        sources = list(dict.fromkeys(sources))
        with self.transaction() as conn:
            self._require_project(conn, memory.project_id)
            existing = conn.execute(
                "SELECT id FROM memories WHERE project_id = ? AND content_hash = ? AND valid_to IS NULL",
                (memory.project_id, digest),
            ).fetchone()
            if existing:
                return existing["id"], False
            for kind, ref in sources:
                self._check_source(conn, memory.project_id, kind, ref)
            if supersedes:
                self._close_memory(conn, supersedes, memory.project_id, memory.valid_from, superseded_by=None)
            conn.execute(
                "INSERT INTO memories(id, project_id, type, statement, why, how, impact, impact_note, tags_json,"
                " metadata_json, source, actor, confidence, content_hash, created_at, updated_at, valid_from, valid_to,"
                " superseded_by, last_used_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    memory.id, memory.project_id, memory.type, memory.statement, memory.why, memory.how, memory.impact,
                    memory.impact_note, json.dumps(memory.tags, ensure_ascii=False),
                    json.dumps(memory.metadata, ensure_ascii=False), memory.source, memory.actor,
                    max(0.0, min(1.0, float(memory.confidence))), digest, memory.created_at, memory.updated_at,
                    memory.valid_from, memory.valid_to, memory.superseded_by, memory.last_used_at,
                ),
            )
            if supersedes:
                conn.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (memory.id, supersedes))
            for kind, ref in sources:
                conn.execute(
                    "INSERT INTO memory_sources(memory_id, source_kind, source_ref) VALUES (?,?,?)", (memory.id, kind, ref)
                )
            self._index_memory(conn, memory)
            audit.entry_id = audit.entry_id or memory.id
            self._insert_audit(conn, audit)
        return memory.id, True

    @staticmethod
    def _check_source(conn: sqlite3.Connection, project_id: str, kind: str, ref: str) -> None:
        if kind == "external":
            if not str(ref).strip():
                raise StoreError("empty external source")
            return
        table = {"episode": "episodes", "memory": "memories"}.get(kind)
        if table is None:
            raise StoreError(f"invalid source kind: {kind!r}")
        row = conn.execute(f"SELECT project_id FROM {table} WHERE id = ?", (ref,)).fetchone()
        if row is None:
            raise NotFoundError(f"source {kind} {ref} does not exist")
        if row["project_id"] not in (project_id, GLOBAL_PROJECT):
            raise StoreError(f"source {kind} {ref} belongs to another project")

    @staticmethod
    def _index_memory(conn: sqlite3.Connection, memory: Memory) -> None:
        cursor = conn.execute("INSERT INTO memories_fts_map(memory_id) VALUES (?)", (memory.id,))
        conn.execute(
            "INSERT INTO memories_fts(rowid, statement, why, how, tags) VALUES (?,?,?,?,?)",
            (cursor.lastrowid, memory.statement, memory.why, memory.how, " ".join(memory.tags)),
        )

    def _close_memory(self, conn: sqlite3.Connection, memory_id: str, project_id: Optional[str], at: str,
                      superseded_by: Optional[str]) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"unknown memory: {memory_id}")
        if project_id is not None and row["project_id"] != project_id:
            raise StoreError("cannot supersede a memory of another project")
        if row["valid_to"] is not None:
            raise ConflictError(f"memory {memory_id} is already closed")
        conn.execute(
            "UPDATE memories SET valid_to = ?, superseded_by = ?, updated_at = ? WHERE id = ?",
            (at, superseded_by, at, memory_id),
        )
        return row

    def supersede(self, memory_id: str, *, audit: AuditRecord, replacement_id: Optional[str] = None,
                  at: Optional[str] = None) -> None:
        """Close a memory, optionally pointing at the memory that replaces it."""

        at = at or utc_now()
        with self.transaction() as conn:
            row = conn.execute("SELECT project_id FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"unknown memory: {memory_id}")
            if replacement_id is not None:
                self._check_source(conn, row["project_id"], "memory", replacement_id)
            self._close_memory(conn, memory_id, None, at, superseded_by=replacement_id)
            audit.entry_id = audit.entry_id or memory_id
            self._insert_audit(conn, audit)

    def get_memory(self, memory_id: str) -> Optional[dict]:
        rows = self._read("SELECT * FROM memories WHERE id = ?", (memory_id,))
        if not rows:
            return None
        memory = self._memory_row(rows[0])
        memory["sources"] = [
            {"kind": row["source_kind"], "ref": row["source_ref"]}
            for row in self._read(
                "SELECT source_kind, source_ref FROM memory_sources WHERE memory_id = ? ORDER BY source_kind, source_ref",
                (memory_id,),
            )
        ]
        return memory

    def list_memories(self, project_id: str, *, active_only: bool = True, type: Optional[str] = None,
                      limit: int = 100) -> list[dict]:
        sql = ["SELECT * FROM memories WHERE project_id = ?"]
        params: list = [project_id]
        if active_only:
            sql.append("AND valid_to IS NULL")
        if type:
            sql.append("AND type = ?")
            params.append(type)
        sql.append("ORDER BY updated_at DESC, id LIMIT ?")
        params.append(max(1, min(int(limit), 500)))
        return [self._memory_row(row) for row in self._read(" ".join(sql), params)]

    def touch_memories(self, memory_ids: Iterable[str], *, at: Optional[str] = None) -> None:
        """Record that memories were returned to an agent (input for confidence decay)."""

        ids = list(memory_ids)
        if not ids:
            return
        at = at or utc_now()
        with self.transaction() as conn:
            conn.executemany("UPDATE memories SET last_used_at = ? WHERE id = ?", [(at, i) for i in ids])

    @staticmethod
    def _memory_row(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "kind": "memory",
            "project_id": row["project_id"],
            "type": row["type"],
            "statement": row["statement"],
            "why": row["why"],
            "how": row["how"],
            "impact": row["impact"],
            "impact_note": row["impact_note"],
            "tags": json.loads(row["tags_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "source": row["source"],
            "actor": row["actor"],
            "confidence": row["confidence"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "valid_from": row["valid_from"],
            "valid_to": row["valid_to"],
            "superseded_by": row["superseded_by"],
            "last_used_at": row["last_used_at"],
        }

    # -- full-text search -------------------------------------------------------

    @staticmethod
    def fts_query(text: str) -> Optional[str]:
        """Turn free text into a safe FTS5 query: every token quoted, OR-joined."""

        tokens = [tok.strip("-_") for tok in _FTS_TOKEN.findall(text or "")]
        tokens = [tok for tok in tokens if tok]
        if not tokens:
            return None
        return " OR ".join('"' + tok.replace('"', '""') + '"' for tok in dict.fromkeys(tokens))

    def search_text(self, text: str, project_ids: Iterable[str], *, kinds: Iterable[str] = ("memory", "episode"),
                    include_closed: bool = False, limit: int = 50) -> list[SearchHit]:
        query = self.fts_query(text)
        projects = list(dict.fromkeys(project_ids))
        if not query or not projects:
            return []
        marks = ",".join("?" for _ in projects)
        hits: list[SearchHit] = []
        kinds = set(kinds)
        if "memory" in kinds:
            closed = "" if include_closed else "AND m.valid_to IS NULL"
            for row in self._read(
                f"SELECT m.id, m.project_id, bm25(memories_fts, 3.0, 1.0, 1.0, 0.5) AS score FROM memories_fts"
                f" JOIN memories_fts_map map ON map.rowid = memories_fts.rowid JOIN memories m ON m.id = map.memory_id"
                f" WHERE memories_fts MATCH ? AND m.project_id IN ({marks}) {closed} ORDER BY score LIMIT ?",
                [query, *projects, limit],
            ):
                hits.append(SearchHit("memory", row["id"], row["project_id"], -float(row["score"])))
        if "episode" in kinds:
            for row in self._read(
                f"SELECT e.id, e.project_id, bm25(episodes_fts, 2.0, 1.0, 0.5) AS score FROM episodes_fts"
                f" JOIN episodes_fts_map map ON map.rowid = episodes_fts.rowid JOIN episodes e ON e.id = map.episode_id"
                f" WHERE episodes_fts MATCH ? AND e.project_id IN ({marks}) ORDER BY score LIMIT ?",
                [query, *projects, limit],
            ):
                hits.append(SearchHit("episode", row["id"], row["project_id"], -float(row["score"])))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    # -- embeddings and vector search -------------------------------------------

    def register_embedding_version(self, *, provider: str, model: str, dim: int, config: Optional[dict] = None,
                                   activate: bool = True) -> str:
        config = config or {}
        version_id = hashlib.sha256(
            json.dumps({"provider": provider, "model": model, "dim": dim, "config": config}, sort_keys=True).encode()
        ).hexdigest()[:16]
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO embedding_versions(version_id, provider, model, dim, config_json, created_at, active)"
                " VALUES (?,?,?,?,?,?,0)",
                (version_id, provider, model, int(dim), json.dumps(config, sort_keys=True), utc_now()),
            )
            if activate:
                conn.execute("UPDATE embedding_versions SET active = 0 WHERE active = 1 AND version_id != ?", (version_id,))
                conn.execute("UPDATE embedding_versions SET active = 1 WHERE version_id = ?", (version_id,))
        return version_id

    def active_embedding_version(self) -> Optional[dict]:
        rows = self._read("SELECT * FROM embedding_versions WHERE active = 1")
        if not rows:
            return None
        row = rows[0]
        return {"version_id": row["version_id"], "provider": row["provider"], "model": row["model"], "dim": row["dim"],
                "config": json.loads(row["config_json"])}

    def _version_dim(self, version_id: str) -> int:
        rows = self._read("SELECT dim FROM embedding_versions WHERE version_id = ?", (version_id,))
        if not rows:
            raise NotFoundError(f"unknown embedding version: {version_id}")
        return int(rows[0]["dim"])

    def _vec_table(self, conn: sqlite3.Connection, version_id: str, dim: int) -> str:
        name = f"vec_{version_id}"
        if name not in self._vec_tables:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0("
                f"item_key TEXT PRIMARY KEY, project_id TEXT PARTITION KEY, embedding float[{dim}] distance_metric=cosine)"
            )
            self._vec_tables.add(name)
        return name

    def put_embedding(self, item_kind: str, item_id: str, version_id: str, vector: list[float]) -> None:
        dim = self._version_dim(version_id)
        if len(vector) != dim:
            raise StoreError(f"vector has {len(vector)} dimensions, version {version_id} expects {dim}")
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            raise StoreError("zero vector")
        vector = [v / norm for v in vector]
        table = {"episode": "episodes", "memory": "memories"}[item_kind]
        blob = pack_vector(vector)
        with self.transaction() as conn:
            row = conn.execute(f"SELECT project_id FROM {table} WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"unknown {item_kind}: {item_id}")
            conn.execute(
                "INSERT OR REPLACE INTO embeddings(item_kind, item_id, version_id, project_id, vector, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (item_kind, item_id, version_id, row["project_id"], blob, utc_now()),
            )
            if self.vec_enabled:
                name = self._vec_table(conn, version_id, dim)
                key = f"{item_kind}:{item_id}"
                conn.execute(f"DELETE FROM {name} WHERE item_key = ?", (key,))
                conn.execute(f"INSERT INTO {name}(item_key, project_id, embedding) VALUES (?,?,?)",
                             (key, row["project_id"], blob))

    def missing_embeddings(self, version_id: str, *, limit: int = 100) -> list[tuple[str, str, str]]:
        """Items without a vector for this version, oldest first: (kind, id, text to embed)."""

        rows = self._read(
            "SELECT 'memory' AS kind, m.id, m.statement || CASE WHEN m.why != '' THEN char(10) || m.why ELSE '' END AS text,"
            " m.created_at FROM memories m WHERE NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.item_kind = 'memory'"
            " AND e.item_id = m.id AND e.version_id = ?)"
            # ciphertext carries no meaning: encrypted v1 memories are never sent for embedding
            " AND m.tags_json NOT LIKE '%\"v1-encrypted\"%'"
            " UNION ALL SELECT 'episode', p.id, p.content, p.created_at FROM episodes p WHERE NOT EXISTS (SELECT 1 FROM"
            " embeddings e WHERE e.item_kind = 'episode' AND e.item_id = p.id AND e.version_id = ?)"
            " ORDER BY created_at LIMIT ?",
            (version_id, version_id, limit),
        )
        return [(row["kind"], row["id"], row["text"]) for row in rows]

    def search_vector(self, vector: list[float], project_ids: Iterable[str], version_id: str, *,
                      kinds: Iterable[str] = ("memory", "episode"), include_closed: bool = False,
                      limit: int = 50) -> list[SearchHit]:
        projects = list(dict.fromkeys(project_ids))
        if not projects:
            return []
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        query = [v / norm for v in vector]
        kinds = set(kinds)
        if self.vec_enabled:
            candidates = self._vec_candidates(query, projects, version_id, limit * 4)
        else:
            candidates = self._scan_candidates(query, projects, version_id)
        closed = set()
        if not include_closed:
            memory_ids = [item_id for kind, item_id, _, _ in candidates if kind == "memory"]
            closed = self._closed_memories(memory_ids)
        hits = [
            SearchHit(kind, item_id, project_id, similarity)
            for kind, item_id, project_id, similarity in candidates
            if kind in kinds and item_id not in closed
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _vec_candidates(self, query: list[float], projects: list[str], version_id: str, k: int):
        name = f"vec_{version_id}"
        dim = self._version_dim(version_id)
        out = []
        with self._lock:
            self._vec_table(self._conn, version_id, dim)
            for project_id in projects:
                for row in self._conn.execute(
                    f"SELECT item_key, distance FROM {name} WHERE embedding MATCH ? AND k = ? AND project_id = ?",
                    (pack_vector(query), k, project_id),
                ):
                    kind, item_id = row["item_key"].split(":", 1)
                    out.append((kind, item_id, project_id, 1.0 - float(row["distance"])))
        return out

    def _scan_candidates(self, query: list[float], projects: list[str], version_id: str):
        marks = ",".join("?" for _ in projects)
        out = []
        for row in self._read(
            f"SELECT item_kind, item_id, project_id, vector FROM embeddings WHERE version_id = ? AND project_id IN ({marks})",
            [version_id, *projects],
        ):
            vec = unpack_vector(row["vector"])
            out.append((row["item_kind"], row["item_id"], row["project_id"], sum(a * b for a, b in zip(query, vec))))
        return out

    def _closed_memories(self, memory_ids: list[str]) -> set[str]:
        closed: set[str] = set()
        for start in range(0, len(memory_ids), 500):
            chunk = memory_ids[start:start + 500]
            marks = ",".join("?" for _ in chunk)
            closed.update(
                row["id"] for row in self._read(
                    f"SELECT id FROM memories WHERE id IN ({marks}) AND valid_to IS NOT NULL", chunk
                )
            )
        return closed

    def rebuild_vector_index(self, version_id: str) -> int:
        """Recreate the sqlite-vec index of a version from the canonical embeddings table."""

        if not self.vec_enabled:
            return 0
        dim = self._version_dim(version_id)
        name = f"vec_{version_id}"
        with self.transaction() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {name}")
            self._vec_tables.discard(name)
            self._vec_table(conn, version_id, dim)
            rows = conn.execute(
                "SELECT item_kind, item_id, project_id, vector FROM embeddings WHERE version_id = ?", (version_id,)
            ).fetchall()
            conn.executemany(
                f"INSERT INTO {name}(item_key, project_id, embedding) VALUES (?,?,?)",
                [(f"{r['item_kind']}:{r['item_id']}", r["project_id"], r["vector"]) for r in rows],
            )
        return len(rows)

    # -- audit --------------------------------------------------------------------

    def add_audit(self, record: AuditRecord) -> int:
        with self.transaction() as conn:
            return self._insert_audit(conn, record)

    def list_audit(self, *, limit: int = 200, entry_id: Optional[str] = None, action: Optional[str] = None,
                   actor: Optional[str] = None, reason: Optional[str] = None, since: Optional[str] = None,
                   after_id: Optional[int] = None) -> list[AuditRecord]:
        """Same semantics as v1: newest first; with after_id, rows with a greater id oldest first."""

        sql = ["SELECT * FROM audit_log WHERE 1 = 1"]
        params: list = []
        for column, value in (("entry_id", entry_id), ("action", action), ("actor", actor), ("reason", reason)):
            if value:
                sql.append(f"AND {column} = ?")
                params.append(value)
        if since:
            sql.append("AND created_at >= ?")
            params.append(since)
        if after_id is not None:
            sql.append("AND id > ? ORDER BY id ASC LIMIT ?")
            params.append(int(after_id))
        else:
            sql.append("ORDER BY created_at DESC, id DESC LIMIT ?")
        params.append(int(limit))
        return [
            AuditRecord(
                id=int(row["id"]), entry_id=row["entry_id"], action=row["action"], actor=row["actor"],
                reason=row["reason"], payload=json.loads(row["payload_json"]), created_at=row["created_at"],
            )
            for row in self._read(" ".join(sql), params)
        ]

    def count_audit(self) -> int:
        return int(self._read("SELECT COUNT(1) AS c FROM audit_log")[0]["c"])
