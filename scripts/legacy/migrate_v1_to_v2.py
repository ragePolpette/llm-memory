"""Migrazione store legacy Markdown (v1) -> SQLite tiered (v2)."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import frontmatter

from src.config import MemoryScope, Tier, get_config
from src.bootstrap import build_runtime
from src.models import EntryStatus, EntryType, MemoryEntry, ScopeRef, compute_content_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate llm-memory v1 markdown store to v2 sqlite")
    parser.add_argument("--source-dir", required=True, help="Directory legacy markdown memories")
    parser.add_argument("--workspace", default="default", help="Target workspace id")
    parser.add_argument("--project", default="default", help="Target project id")
    parser.add_argument("--agent", default="migration", help="Actor/agent id for migration metadata")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"source dir not found: {source_dir}")

    config = get_config()
    runtime = build_runtime(config)

    imported = 0
    duplicates = 0

    md_files = sorted(source_dir.rglob("*.md"))
    for md_file in md_files:
        post = frontmatter.load(md_file)
        content = post.content or ""
        content_hash = post.get("content_hash") or compute_content_hash(content)

        scope = ScopeRef(
            workspace_id=args.workspace,
            project_id=args.project,
            user_id=None,
            agent_id=post.get("agent_id") or args.agent,
        )

        duplicate = runtime.store.find_by_hash(scope=scope, content_hash=content_hash)
        if duplicate:
            duplicates += 1
            continue

        created_at_raw = post.get("created_at")
        try:
            created_at = datetime.fromisoformat(created_at_raw).isoformat() if created_at_raw else datetime.utcnow().isoformat()
        except Exception:
            created_at = datetime.utcnow().isoformat()

        visibility = MemoryScope(post.get("scope", "shared"))

        entry = MemoryEntry(
            id=str(post.get("id") or md_file.stem),
            tier=Tier.TIER_2,
            scope=scope,
            visibility=visibility,
            source="migration_v1",
            type=EntryType.FACT,
            status=EntryStatus.ACTIVE,
            content=content,
            context=post.get("context", ""),
            tags=post.get("tags", []),
            metadata=post.get("metadata", {}),
            confidence=0.7,
            created_at=created_at,
            updated_at=created_at,
            content_hash=content_hash,
            redacted=False,
            encrypted=False,
        )

        runtime.store.add_entry(entry)
        imported += 1

    print(f"Migration completed: imported={imported} duplicates={duplicates} source_files={len(md_files)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
