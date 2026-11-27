"""One-shot migration from a v1 llm-memory database to the v2 store.

    python -m src.store_v2.migrate_v1 --from data/memory.db --to data/memory-v2.db [--dry-run]

What it keeps:
- entry ids (strong entries become memories, fast entries become episodes), so ids that
  Yesod or other clients hold stay valid;
- the audit log with its ids, and the AUTOINCREMENT counter, so an after_id cursor keeps
  working and new rows continue after the old ones;
- projects, links, tags, metadata, timestamps, invalidation and supersession.

What it changes, on purpose:
- v1 types assumption and unknown become type fact with a tag; v1 invalidation marker
  entries are not copied (the invalidation is kept on the target as valid_to);
- workspace- and global-scope entries go to the global project, tagged with their origin;
- vectors are not copied: v1 vectors come from another model, so everything is queued
  for embedding with the v2 provider;
- migrated episodes never expire (expires_at NULL): nothing is lost by the migration.

Encrypted v1 entries are copied as-is and tagged `v1-encrypted`; they need the v1 key
to be read and are reported in the summary.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .schema import GLOBAL_PROJECT
from .store import MemoryStore, content_hash

_TYPE_MAP = {"fact": "fact", "decision": "decision", "assumption": "fact", "unknown": "fact"}


def _project_for(row: sqlite3.Row) -> tuple[str, list[str]]:
    level = row["scope_level"] if "scope_level" in row.keys() else "project"
    if level == "project":
        return MemoryStore.validate_project_id(row["project_id"]), []
    return GLOBAL_PROJECT, [f"v1-scope:{level}", f"v1-workspace:{row['workspace_id']}"]


def migrate(source: Path | str, target: Path | str, *, dry_run: bool = False) -> dict:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    tables = {row[0] for row in src.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "entries" not in tables or "audit_log" not in tables:
        src.close()
        raise ValueError(f"{source} does not look like a v1 llm-memory database")

    report = {"projects": 0, "memories": 0, "memories_closed": 0, "invalidation_markers_skipped": 0,
              "encrypted_copied": 0, "episodes": 0, "links": 0, "audit_rows": 0, "max_audit_id": 0,
              "dry_run": dry_run}

    store = MemoryStore(":memory:" if dry_run else target, use_sqlite_vec=False)
    conn = store._conn  # the migration writes rows verbatim, below the public API

    try:
        if not dry_run and (store.list_projects() or store.count_audit()):
            raise ValueError(f"{target} is not empty: refusing to migrate into it")
        with store.transaction():
            projects: dict[str, tuple[str, str, str]] = {}
            if "projects" in tables:
                for row in src.execute("SELECT * FROM projects ORDER BY created_at"):
                    projects.setdefault(row["project_id"], (row["display_name"], row["description"], row["created_at"]))
            needed = {GLOBAL_PROJECT, "default"}
            for table in ("entries", "fast_memory_entries"):
                if table in tables:
                    needed.update(r[0] for r in src.execute(f"SELECT DISTINCT project_id FROM {table}"))
            needed.discard("__workspace__")
            needed.discard("__global__")
            needed.add(GLOBAL_PROJECT)
            for project_id in sorted(needed):
                name, description, created = projects.get(project_id, (project_id, "", None))
                created = created or "1970-01-01T00:00:00+00:00"
                conn.execute(
                    "INSERT INTO projects(project_id, display_name, description, metadata_json, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?)",
                    (MemoryStore.validate_project_id(project_id), name, description, "{}", created, created),
                )
                report["projects"] += 1

            entries = src.execute("SELECT * FROM entries ORDER BY created_at, id").fetchall()
            closed_by: dict[str, str] = {}
            for row in entries:
                if row["type"] == "invalidated":
                    report["invalidation_markers_skipped"] += 1
                    continue
                project_id, extra_tags = _project_for(row)
                tags = list(json.loads(row["tags_json"])) + extra_tags
                if row["type"] in ("assumption", "unknown"):
                    tags.append(f"v1-type:{row['type']}")
                if row["encrypted"]:
                    tags.append("v1-encrypted")
                    report["encrypted_copied"] += 1
                metadata = json.loads(row["metadata_json"])
                metadata.update({"v1_tier": row["tier"], "v1_visibility": row["visibility"], "v1_source": row["source"],
                                 "v1_status": row["status"]})
                valid_to = row["updated_at"] if row["status"] in ("invalidated", "superseded") else None
                if valid_to:
                    report["memories_closed"] += 1
                    closed_by[row["id"]] = valid_to
                memory_type = _TYPE_MAP.get(row["type"], "fact")
                conn.execute(
                    "INSERT INTO memories(id, project_id, type, statement, why, how, impact, impact_note, tags_json,"
                    " metadata_json, source, actor, confidence, content_hash, created_at, updated_at, valid_from, valid_to,"
                    " superseded_by, last_used_at) VALUES (?,?,?,?,?,?,NULL,'',?,?,'migrated',?,?,?,?,?,?,?,NULL,NULL)",
                    (
                        row["id"], project_id, memory_type, row["content"], "", row["context"] or "",
                        json.dumps(tags, ensure_ascii=False), json.dumps(metadata, ensure_ascii=False),
                        row["agent_id"] or "v1", float(row["confidence"]),
                        # closed rows get a distinct hash so they never block a new active duplicate
                        content_hash(memory_type, row["content"]) + ("" if valid_to is None else f":{row['id']}"),
                        row["created_at"], row["updated_at"], row["created_at"], valid_to,
                    ),
                )
                cursor = conn.execute("INSERT INTO memories_fts_map(memory_id) VALUES (?)", (row["id"],))
                conn.execute(
                    "INSERT INTO memories_fts(rowid, statement, why, how, tags) VALUES (?,?,?,?,?)",
                    (cursor.lastrowid, "" if row["encrypted"] else row["content"], "", row["context"] or "", " ".join(tags)),
                )
                report["memories"] += 1

            if "links" in tables:
                for row in src.execute("SELECT entry_id, target_id, relation FROM links"):
                    conn.execute("INSERT OR IGNORE INTO links(item_id, target_id, relation) VALUES (?,?,?)",
                                 (row["entry_id"], row["target_id"], row["relation"]))
                    report["links"] += 1

            if "fast_memory_entries" in tables:
                for row in src.execute("SELECT * FROM fast_memory_entries ORDER BY created_at, id"):
                    metadata = json.loads(row["metadata_json"])
                    metadata.update({
                        "v1_resolved": bool(row["resolved"]), "v1_distillation_status": row["distillation_status"],
                        "v1_recurrence_count": row["recurrence_count"], "v1_selection_score": row["selection_score"],
                        "v1_cluster_id": row["cluster_id"], "v1_user_id": row["user_id"],
                    })
                    project_id = MemoryStore.validate_project_id(row["project_id"])
                    tags = json.loads(row["tags_json"])
                    conn.execute(
                        "INSERT INTO episodes(id, project_id, event_type, content, context, tags_json, metadata_json,"
                        " agent_id, session_id, source, content_hash, created_at, expires_at)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
                        (
                            row["id"], project_id, row["event_type"], row["content"], row["context"] or "",
                            json.dumps(tags, ensure_ascii=False), json.dumps(metadata, ensure_ascii=False),
                            row["agent_id"], row["session_id"], row["source"],
                            content_hash(row["event_type"], row["content"]), row["created_at"],
                        ),
                    )
                    cursor = conn.execute("INSERT INTO episodes_fts_map(episode_id) VALUES (?)", (row["id"],))
                    conn.execute("INSERT INTO episodes_fts(rowid, content, context, tags) VALUES (?,?,?,?)",
                                 (cursor.lastrowid, row["content"], row["context"] or "", " ".join(tags)))
                    report["episodes"] += 1

            for row in src.execute("SELECT * FROM audit_log ORDER BY id"):
                conn.execute(
                    "INSERT INTO audit_log(id, entry_id, action, actor, reason, payload_json, created_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (row["id"], row["entry_id"], row["action"], row["actor"], row["reason"], row["payload_json"],
                     row["created_at"]),
                )
                report["audit_rows"] += 1
                report["max_audit_id"] = max(report["max_audit_id"], int(row["id"]))
            # keep the AUTOINCREMENT counter at least where v1 left it (ids of deleted rows are never reused)
            v1_seq = src.execute("SELECT seq FROM sqlite_sequence WHERE name = 'audit_log'").fetchone()
            seq = max(report["max_audit_id"], int(v1_seq[0]) if v1_seq else 0)
            if seq:
                conn.execute("DELETE FROM sqlite_sequence WHERE name = 'audit_log'")
                conn.execute("INSERT INTO sqlite_sequence(name, seq) VALUES ('audit_log', ?)", (seq,))
            report["audit_sequence"] = seq
            if dry_run:
                raise _DryRun()
    except _DryRun:
        pass
    finally:
        src.close()
        store.close()
    return report


class _DryRun(Exception):
    """Raised to roll back the transaction of a dry run."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate a v1 llm-memory database to the v2 store.")
    parser.add_argument("--from", dest="source", required=True, type=Path)
    parser.add_argument("--to", dest="target", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and args.target is None:
        parser.error("--to is required unless --dry-run")
    report = migrate(args.source, args.target or Path(":memory:"), dry_run=args.dry_run)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
