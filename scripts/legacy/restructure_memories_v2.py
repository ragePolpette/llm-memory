"""Ristrutturazione completa memorie legacy -> schema v2 con pulizia controllata.

Flow:
1) Backup sorgenti (Markdown + eventuale SQLite)
2) Migrazione legacy Markdown in SQLite v2
3) Invalidazione automatica contenuti non persistenti (tecnico/operativo/regole)
4) Reembed incrementale
5) Report finale JSON+Markdown
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import frontmatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, MemoryScope, Tier, get_config
from src.models import (
    AuditEvent,
    EntryLink,
    EntryStatus,
    EntryType,
    MemoryEntry,
    ScopeRef,
    compute_content_hash,
)
from src.bootstrap import build_runtime
from src.service.memory_service import ActorContext


@dataclass
class LegacyMemory:
    file_path: Path
    legacy_id: str
    agent_id: str
    created_at: str
    scope: MemoryScope
    context: str
    tags: list[str]
    metadata: dict
    content: str


TECHNICAL_PATTERNS = [
    r"\bsql\b",
    r"\bdb\b",
    r"\bdatabase\b",
    r"\bfrontend\b",
    r"\bbackend\b",
    r"\bdto\b",
    r"\bapi\b",
    r"\bjson\b",
    r"\bserializer\b",
    r"\bwcf\b",
    r"\bbug\b",
    r"\brefactor\b",
    r"\bmigration\b",
    r"\bbranch\b",
    r"\bmcp\b",
    r"\btoolchain\b",
    r"\bnode\b",
    r"\bnvm\b",
    r"\bcodex\b",
    r"\bclaude\b",
    r"\bfix\b",
    r"\bpayload\b",
    r"\bconfigsessione\b",
    r"\bcomponent\b",
    r"\bmodal\b",
    r"\bbpopilot\b",
    r"\bdev[a-z]*-\d+\b",
]

OPERATIONAL_PATTERNS = [
    r"\bworkflow\b",
    r"\bprocedur",
    r"\boperativ",
    r"\brunbook\b",
    r"\bistruzion",
    r"\bchecklist\b",
]

RULE_PATTERNS = [
    r"\bregol",
    r"\bmai\b",
    r"\bobbligator",
    r"\bnon usare\b",
    r"\bdeve\b",
]

RX_TECH = re.compile("|".join(TECHNICAL_PATTERNS), flags=re.IGNORECASE)
RX_OPERATIONAL = re.compile("|".join(OPERATIONAL_PATTERNS), flags=re.IGNORECASE)
RX_RULE = re.compile("|".join(RULE_PATTERNS), flags=re.IGNORECASE)
RX_CODE = re.compile(r"`[^`]+`|```", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restructure llm-memory into v2 schema")
    parser.add_argument("--source-markdown-dir", default="./memories", help="Legacy markdown memories dir")
    parser.add_argument("--target-sqlite", default="./data/memory.db", help="Target SQLite db path")
    parser.add_argument("--workspace", default="default", help="Workspace ID")
    parser.add_argument("--project", default="default", help="Project ID")
    parser.add_argument("--actor", default="memory-admin", help="Actor used for migration/audit")
    parser.add_argument("--backup-dir", default="./data/restructure_backups", help="Backup root dir")
    parser.add_argument("--report-dir", default="./data/restructure_reports", help="Report output dir")
    parser.add_argument("--keep-technical", action="store_true", help="Do not invalidate technical entries")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_legacy_memories(source_dir: Path) -> list[LegacyMemory]:
    memories: list[LegacyMemory] = []
    for md_file in sorted(source_dir.rglob("*.md")):
        post = frontmatter.load(md_file)
        legacy_id = str(post.get("id") or md_file.stem)
        scope_raw = str(post.get("scope", "shared"))
        try:
            scope = MemoryScope(scope_raw)
        except Exception:
            scope = MemoryScope.SHARED

        created_at = str(post.get("created_at") or now_utc())

        memories.append(
            LegacyMemory(
                file_path=md_file,
                legacy_id=legacy_id,
                agent_id=str(post.get("agent_id") or "legacy-agent"),
                created_at=created_at,
                scope=scope,
                context=str(post.get("context") or ""),
                tags=list(post.get("tags") or []),
                metadata=dict(post.get("metadata") or {}),
                content=(post.content or "").strip(),
            )
        )
    return memories


def infer_type(memory: LegacyMemory) -> EntryType:
    context = memory.context.lower()
    tags = [t.lower() for t in memory.tags]
    content = memory.content.lower()

    if "invalid" in context or "invalid" in content:
        return EntryType.INVALIDATED
    if "assunzion" in context or "assunzion" in content:
        return EntryType.ASSUMPTION
    if "unknown" in context or "open_unknown" in context:
        return EntryType.UNKNOWN
    if "decision" in context or "decision-log" in tags or "decision log" in content:
        return EntryType.DECISION
    return EntryType.FACT


def infer_tier(entry_type: EntryType, invalidated: bool) -> Tier:
    if invalidated:
        return Tier.TIER_3
    if entry_type in {EntryType.DECISION, EntryType.FACT}:
        return Tier.TIER_3
    return Tier.TIER_2


def classify_non_persistent(memory: LegacyMemory) -> list[str]:
    text = " ".join([memory.context, memory.content, " ".join(memory.tags)]).strip()
    reasons: list[str] = []
    if RX_TECH.search(text):
        reasons.append("technical_context")
    if RX_CODE.search(memory.content):
        reasons.append("code_or_snippet_context")
    if RX_OPERATIONAL.search(text):
        reasons.append("operational_consideration")
    if RX_RULE.search(text):
        reasons.append("rule_like_content")
    return reasons


def sanitize_content(content: str) -> str:
    # Evita payload vuoti in SQLite.
    cleaned = content.strip()
    return cleaned if cleaned else "[EMPTY_CONTENT]"


def prepare_backup(source_dir: Path, target_db: Path, backup_root: Path) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"restructure_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    if source_dir.exists():
        shutil.copytree(source_dir, backup_dir / "legacy_memories", dirs_exist_ok=False)

    if target_db.exists():
        (backup_dir / "sqlite").mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_db, backup_dir / "sqlite" / target_db.name)

    return backup_dir


def write_report(report_dir: Path, payload: dict) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = report_dir / f"restructure_report_{stamp}.json"
    md_path = report_dir / f"restructure_report_{stamp}.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Restructure Report",
        "",
        f"- timestamp: {payload['timestamp']}",
        f"- dry_run: {payload['dry_run']}",
        f"- source_markdown: {payload['source_markdown']}",
        f"- target_sqlite: {payload['target_sqlite']}",
        f"- backup_dir: {payload['backup_dir']}",
        "",
        "## Counters",
        "",
        f"- total_legacy_files: {payload['counters']['total_legacy_files']}",
        f"- migrated_entries: {payload['counters']['migrated_entries']}",
        f"- skipped_duplicates: {payload['counters']['skipped_duplicates']}",
        f"- invalidated_non_persistent: {payload['counters']['invalidated_non_persistent']}",
        f"- kept_active: {payload['counters']['kept_active']}",
        f"- reembed_processed: {payload['counters']['reembed_processed']}",
        f"- reembed_remaining: {payload['counters']['reembed_remaining']}",
        "",
        "## Non-persistent Reasons",
        "",
    ]
    for reason, count in payload["non_persistent_reasons"].items():
        lines.append(f"- {reason}: {count}")

    lines.extend(["", "## Verification", ""])
    for key, value in payload["verification"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Preview Kept Active", ""])
    for item in payload.get("preview", {}).get("kept_active", [])[:15]:
        lines.append(f"- {item['legacy_id']} | {item['context']} | {item['file']}")

    lines.extend(["", "## Preview Invalidated", ""])
    for item in payload.get("preview", {}).get("invalidated", [])[:15]:
        reasons = ",".join(item.get("reasons", []))
        lines.append(f"- {item['legacy_id']} | {item['context']} | reasons={reasons}")

    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return json_path, md_path


async def run() -> int:
    args = parse_args()
    source_dir = Path(args.source_markdown_dir).expanduser().resolve()
    target_db = Path(args.target_sqlite).expanduser().resolve()
    backup_root = Path(args.backup_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source markdown dir not found: {source_dir}")

    backup_dir = prepare_backup(source_dir=source_dir, target_db=target_db, backup_root=backup_root)

    legacy_memories = read_legacy_memories(source_dir)
    reason_counter: Counter[str] = Counter()

    if args.dry_run:
        would_invalidate = 0
        kept_preview: list[dict] = []
        invalidated_preview: list[dict] = []
        for item in legacy_memories:
            reasons = classify_non_persistent(item)
            if reasons and not args.keep_technical:
                would_invalidate += 1
                reason_counter.update(reasons)
                if len(invalidated_preview) < 25:
                    invalidated_preview.append(
                        {
                            "legacy_id": item.legacy_id,
                            "file": str(item.file_path),
                            "context": item.context,
                            "reasons": reasons,
                        }
                    )
            else:
                if len(kept_preview) < 25:
                    kept_preview.append(
                        {
                            "legacy_id": item.legacy_id,
                            "file": str(item.file_path),
                            "context": item.context,
                        }
                    )

        payload = {
            "timestamp": now_utc(),
            "dry_run": True,
            "source_markdown": str(source_dir),
            "target_sqlite": str(target_db),
            "backup_dir": str(backup_dir),
            "counters": {
                "total_legacy_files": len(legacy_memories),
                "migrated_entries": 0,
                "skipped_duplicates": 0,
                "invalidated_non_persistent": would_invalidate,
                "kept_active": len(legacy_memories) - would_invalidate,
                "reembed_processed": 0,
                "reembed_remaining": 0,
            },
            "non_persistent_reasons": dict(reason_counter),
            "preview": {
                "kept_active": kept_preview,
                "invalidated": invalidated_preview,
            },
            "verification": {
                "sqlite_created": target_db.exists(),
                "mcp_bootstrap": "not-executed (dry-run)",
                "search_smoke": "not-executed (dry-run)",
            },
        }
        json_path, md_path = write_report(report_dir, payload)
        print(f"[DRY-RUN] report_json={json_path}")
        print(f"[DRY-RUN] report_md={md_path}")
        return 0

    # Build runtime on target DB
    cfg: Config = get_config()
    cfg.sqlite_db_path = target_db
    cfg.default_workspace_id = args.workspace
    cfg.default_project_id = args.project
    runtime = build_runtime(cfg)

    # Reset target DB for clean migration (backup already taken)
    if target_db.exists():
        target_db.unlink()
        runtime = build_runtime(cfg)

    scope = ScopeRef(
        workspace_id=args.workspace,
        project_id=args.project,
        user_id=None,
        agent_id=None,
    )

    migrated_entries = 0
    skipped_duplicates = 0
    invalidated_non_persistent = 0
    kept_preview: list[dict] = []
    invalidated_preview: list[dict] = []

    for item in legacy_memories:
        content = sanitize_content(item.content)
        content_hash = compute_content_hash(content)

        duplicate = runtime.store.find_by_hash(scope=scope, content_hash=content_hash)
        if duplicate is not None:
            skipped_duplicates += 1
            continue

        reasons = classify_non_persistent(item)
        invalidate = bool(reasons and not args.keep_technical)
        if invalidate:
            invalidated_non_persistent += 1
            reason_counter.update(reasons)
            if len(invalidated_preview) < 25:
                invalidated_preview.append(
                    {
                        "legacy_id": item.legacy_id,
                        "file": str(item.file_path),
                        "context": item.context,
                        "reasons": reasons,
                    }
                )
        else:
            if len(kept_preview) < 25:
                kept_preview.append(
                    {
                        "legacy_id": item.legacy_id,
                        "file": str(item.file_path),
                        "context": item.context,
                    }
                )

        entry_type = infer_type(item)
        status = EntryStatus.INVALIDATED if invalidate else EntryStatus.ACTIVE
        tier = infer_tier(entry_type=entry_type, invalidated=invalidate)

        entry = MemoryEntry(
            id=item.legacy_id,
            tier=tier,
            scope=scope,
            visibility=item.scope,
            source="legacy-markdown-migration",
            type=entry_type,
            status=status,
            content=content,
            context=item.context,
            tags=item.tags,
            metadata={
                "legacy_file": str(item.file_path),
                "legacy_agent_id": item.agent_id,
                "legacy_metadata": item.metadata,
                "cleanup_reasons": reasons,
            },
            links=[],
            confidence=0.8 if not invalidate else 0.1,
            created_at=item.created_at,
            updated_at=now_utc(),
            content_hash=content_hash,
            encrypted=False,
            redacted=False,
        )
        runtime.store.add_entry(entry)
        migrated_entries += 1

        runtime.store.add_audit(
            AuditEvent(
                entry_id=entry.id,
                action="migration_import",
                actor=args.actor,
                reason="legacy markdown to v2",
                payload={
                    "legacy_file": str(item.file_path),
                    "invalidated": invalidate,
                    "cleanup_reasons": reasons,
                },
            )
        )

        if invalidate:
            invalidation_entry = MemoryEntry(
                tier=Tier.TIER_3,
                scope=scope,
                visibility=item.scope,
                source="cleanup-policy",
                type=EntryType.INVALIDATED,
                status=EntryStatus.ACTIVE,
                content=(
                    "Entry invalidated during restructuring: non-persistent content "
                    f"({', '.join(reasons)})"
                ),
                context="automatic cleanup invalidation",
                links=[EntryLink(target_id=entry.id, relation="invalidates")],
                confidence=1.0,
                created_at=now_utc(),
                updated_at=now_utc(),
            )
            runtime.store.add_entry(invalidation_entry)
            runtime.store.add_audit(
                AuditEvent(
                    entry_id=entry.id,
                    action="cleanup_invalidate",
                    actor=args.actor,
                    reason="non persistent content",
                    payload={"reasons": reasons, "invalidation_entry_id": invalidation_entry.id},
                )
            )

    actor = ActorContext(
        agent_id=args.actor,
        user_id=None,
        workspace_id=args.workspace,
        project_id=args.project,
    )
    reembed_result = await runtime.service.reembed(actor=actor, activate=True)

    smoke_results = await runtime.service.search(
        query="decisioni stabili e memoria persistente",
        actor=actor,
        limit=5,
        include_invalidated=False,
    )

    kept_active = migrated_entries - invalidated_non_persistent
    payload = {
        "timestamp": now_utc(),
        "dry_run": False,
        "source_markdown": str(source_dir),
        "target_sqlite": str(target_db),
        "backup_dir": str(backup_dir),
        "counters": {
            "total_legacy_files": len(legacy_memories),
            "migrated_entries": migrated_entries,
            "skipped_duplicates": skipped_duplicates,
            "invalidated_non_persistent": invalidated_non_persistent,
            "kept_active": kept_active,
            "reembed_processed": reembed_result.processed,
            "reembed_remaining": reembed_result.remaining,
        },
        "non_persistent_reasons": dict(reason_counter),
        "preview": {
            "kept_active": kept_preview,
            "invalidated": invalidated_preview,
        },
        "verification": {
            "sqlite_created": target_db.exists(),
            "active_entries_present": kept_active > 0,
            "search_smoke_results": len(smoke_results),
            "search_smoke_top_ids": [item.entry_id for item in smoke_results],
        },
    }

    json_path, md_path = write_report(report_dir, payload)
    print(f"[APPLY] report_json={json_path}")
    print(f"[APPLY] report_md={md_path}")
    print(json.dumps(payload["counters"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(run()))
