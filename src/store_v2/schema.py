"""Schema of the v2 store, as numbered migrations tracked in PRAGMA user_version.

Each migration runs once, in order, inside one transaction. Never edit a migration
that has shipped: add a new one.
"""

from __future__ import annotations

import sqlite3

GLOBAL_PROJECT = "__global__"

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE projects (
            project_id   TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            description  TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );

        -- Raw episodic notes. Never edited after insert; they expire unless a memory cites them.
        -- Identical notes are kept: the same event happening twice is signal (recurrence).
        CREATE TABLE episodes (
            id            TEXT PRIMARY KEY,
            project_id    TEXT NOT NULL REFERENCES projects(project_id),
            event_type    TEXT NOT NULL,
            content       TEXT NOT NULL,
            context       TEXT NOT NULL DEFAULT '',
            tags_json     TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            agent_id      TEXT NOT NULL,
            session_id    TEXT,
            source        TEXT NOT NULL DEFAULT 'agent',
            content_hash  TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            expires_at    TEXT
        );
        CREATE INDEX ix_episodes_hash ON episodes(project_id, content_hash);
        CREATE INDEX ix_episodes_project_time ON episodes(project_id, created_at DESC);
        CREATE INDEX ix_episodes_expiry ON episodes(expires_at) WHERE expires_at IS NOT NULL;

        -- Structured memories: decisions, lessons, facts, constraints, preferences.
        -- A contradiction closes the old row (valid_to, superseded_by); rows are not deleted.
        CREATE TABLE memories (
            id            TEXT PRIMARY KEY,
            project_id    TEXT NOT NULL REFERENCES projects(project_id),
            type          TEXT NOT NULL CHECK (type IN ('decision','lesson','fact','constraint','preference')),
            statement     TEXT NOT NULL,
            why           TEXT NOT NULL DEFAULT '',
            how           TEXT NOT NULL DEFAULT '',
            impact        TEXT CHECK (impact IS NULL OR impact IN ('low','medium','high')),
            impact_note   TEXT NOT NULL DEFAULT '',
            tags_json     TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            source        TEXT NOT NULL CHECK (source IN ('agent','human','distilled','migrated')),
            actor         TEXT NOT NULL,
            confidence    REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
            content_hash  TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            valid_from    TEXT NOT NULL,
            valid_to      TEXT,
            superseded_by TEXT REFERENCES memories(id),
            last_used_at  TEXT
        );
        CREATE UNIQUE INDEX ux_memories_hash_active ON memories(project_id, content_hash) WHERE valid_to IS NULL;
        CREATE INDEX ix_memories_project_active ON memories(project_id, valid_to, updated_at DESC);

        -- Provenance of a memory: episodes, other memories or external references.
        CREATE TABLE memory_sources (
            memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            source_kind TEXT NOT NULL CHECK (source_kind IN ('episode','memory','external')),
            source_ref  TEXT NOT NULL,
            PRIMARY KEY (memory_id, source_kind, source_ref)
        );
        CREATE INDEX ix_memory_sources_ref ON memory_sources(source_kind, source_ref);

        -- Same shape as v1 so ids and the /admin/audit contract survive the migration.
        CREATE TABLE audit_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id     TEXT,
            action       TEXT NOT NULL,
            actor        TEXT NOT NULL,
            reason       TEXT,
            payload_json TEXT NOT NULL,
            created_at   TEXT NOT NULL
        );
        CREATE INDEX ix_audit_entry ON audit_log(entry_id, created_at DESC);
        CREATE INDEX ix_audit_created ON audit_log(created_at DESC);

        CREATE TABLE embedding_versions (
            version_id  TEXT PRIMARY KEY,
            provider    TEXT NOT NULL,
            model       TEXT NOT NULL,
            dim         INTEGER NOT NULL,
            config_json TEXT NOT NULL DEFAULT '{}',
            created_at  TEXT NOT NULL,
            active      INTEGER NOT NULL DEFAULT 0
        );
        CREATE UNIQUE INDEX ux_embedding_versions_active ON embedding_versions(active) WHERE active = 1;

        -- Canonical vector storage (float32 little endian). The sqlite-vec index, when
        -- available, is derived from this table and can be rebuilt at any time.
        CREATE TABLE embeddings (
            item_kind  TEXT NOT NULL CHECK (item_kind IN ('episode','memory')),
            item_id    TEXT NOT NULL,
            version_id TEXT NOT NULL REFERENCES embedding_versions(version_id),
            project_id TEXT NOT NULL,
            vector     BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (item_kind, item_id, version_id)
        );
        CREATE INDEX ix_embeddings_scope ON embeddings(version_id, project_id);

        -- Full-text indexes over the text an agent searches. External content tables,
        -- kept in sync by triggers.
        CREATE VIRTUAL TABLE episodes_fts USING fts5(
            content, context, tags,
            content='', contentless_delete=1,
            tokenize='unicode61 remove_diacritics 2 tokenchars ''-_'''
        );
        CREATE TABLE episodes_fts_map (rowid INTEGER PRIMARY KEY, episode_id TEXT NOT NULL UNIQUE);

        CREATE VIRTUAL TABLE memories_fts USING fts5(
            statement, why, how, tags,
            content='', contentless_delete=1,
            tokenize='unicode61 remove_diacritics 2 tokenchars ''-_'''
        );
        CREATE TABLE memories_fts_map (rowid INTEGER PRIMARY KEY, memory_id TEXT NOT NULL UNIQUE);

        CREATE TABLE links (
            item_id   TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation  TEXT NOT NULL,
            PRIMARY KEY (item_id, target_id, relation)
        );
        """,
    ),
]

LATEST_VERSION = MIGRATIONS[-1][0]


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Bring the database to the latest schema version. Returns the version reached."""

    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > LATEST_VERSION:
        raise RuntimeError(
            f"database schema version {current} is newer than this code supports ({LATEST_VERSION})"
        )
    for version, sql in MIGRATIONS:
        if version <= current:
            continue
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in _split_statements(sql):
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {version}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _split_statements(sql: str) -> list[str]:
    # executescript() would commit on its own; split so the migration stays one transaction.
    statements: list[str] = []
    buffer: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        buffer.append(line)
        candidate = "\n".join(buffer)
        if stripped.endswith(";") and sqlite3.complete_statement(candidate):
            statements.append(candidate.strip())
            buffer = []
    if "".join(buffer).strip():
        statements.append("\n".join(buffer).strip())
    return statements
