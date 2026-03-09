"""Parser/renderer deterministico per formato canonico memory.md."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone

from ..config import MemoryScope, Tier
from ..models import EntryLink, EntryStatus, EntryType, MemoryEntry, ScopeRef, compute_content_hash

_SECTION_ORDER = [
    "PURPOSE",
    "STABLE_FACTS",
    "ASSUMPTIONS",
    "OPEN_UNKNOWNs",
    "DECISIONS",
    "INVALIDATED",
]

_SECTION_TO_TYPE = {
    "STABLE_FACTS": EntryType.FACT,
    "ASSUMPTIONS": EntryType.ASSUMPTION,
    "OPEN_UNKNOWNs": EntryType.UNKNOWN,
    "DECISIONS": EntryType.DECISION,
    "INVALIDATED": EntryType.INVALIDATED,
}

_TYPE_TO_SECTION = {value: key for key, value in _SECTION_TO_TYPE.items()}

_ITEM_RE = re.compile(r"^- \[(?P<id>[^\]]+)\] (?P<content>.*)$")


def _normalize_dt(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def render_memory_markdown(entries: list[MemoryEntry], purpose: str = "Local memory snapshot") -> str:
    """Render deterministico memory.md da entries interne."""

    grouped: dict[str, list[MemoryEntry]] = defaultdict(list)
    for entry in entries:
        section = _TYPE_TO_SECTION.get(entry.type)
        if section:
            grouped[section].append(entry)

    for section in grouped:
        grouped[section].sort(key=lambda e: (_normalize_dt(e.created_at), e.id))

    lines: list[str] = []
    lines.append("# PURPOSE")
    lines.append(purpose.strip() or "Local memory snapshot")
    lines.append("")

    for section in _SECTION_ORDER[1:]:
        lines.append(f"# {section}")
        section_entries = grouped.get(section, [])
        if not section_entries:
            lines.append("- [none] N/A")
            lines.append("  meta: {}")
            lines.append("")
            continue

        for entry in section_entries:
            snippet = entry.content.replace("\n", " ").strip()
            lines.append(f"- [{entry.id}] {snippet}")
            meta = {
                "confidence": entry.confidence,
                "context": entry.context,
                "created_at": _normalize_dt(entry.created_at),
                "encrypted": bool(entry.encrypted),
                "redacted": bool(entry.redacted),
                "links": [link.model_dump() for link in entry.links],
                "scope": entry.scope.model_dump(),
                "source": entry.source,
                "status": entry.status.value,
                "tags": entry.tags,
                "sensitivity_tags": entry.sensitivity_tags,
                "tier": entry.tier.value,
                "updated_at": _normalize_dt(entry.updated_at),
                "visibility": entry.visibility.value,
            }
            lines.append(f"  meta: {json.dumps(meta, ensure_ascii=True, sort_keys=True)}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def parse_memory_markdown(markdown: str, base_scope: ScopeRef) -> list[MemoryEntry]:
    """Parse memory.md in entries interne (deterministico)."""

    entries: list[MemoryEntry] = []
    current_section: str | None = None
    pending_item: dict | None = None

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("# "):
            header = line[2:].strip()
            current_section = header if header in _SECTION_ORDER else None
            pending_item = None
            continue

        if current_section is None or current_section == "PURPOSE":
            continue

        item_match = _ITEM_RE.match(line)
        if item_match:
            entry_id = item_match.group("id").strip()
            content = item_match.group("content").strip()
            if entry_id == "none" and content == "N/A":
                pending_item = None
                continue
            pending_item = {
                "id": entry_id,
                "content": content,
                "section": current_section,
                "meta": {},
            }
            continue

        if pending_item is not None and line.startswith("  meta: "):
            payload = line[len("  meta: ") :].strip()
            meta = json.loads(payload) if payload else {}
            pending_item["meta"] = meta

            scope_payload = meta.get("scope", {})
            scope = ScopeRef(
                workspace_id=scope_payload.get("workspace_id", base_scope.workspace_id),
                project_id=scope_payload.get("project_id", base_scope.project_id),
                user_id=scope_payload.get("user_id", base_scope.user_id),
                agent_id=scope_payload.get("agent_id", base_scope.agent_id),
            )

            links = [EntryLink(**item) for item in meta.get("links", [])]
            entry = MemoryEntry(
                id=pending_item["id"],
                tier=Tier(meta.get("tier", Tier.TIER_2.value)),
                scope=scope,
                visibility=MemoryScope(meta.get("visibility", MemoryScope.SHARED.value)),
                source=meta.get("source", "import"),
                type=_SECTION_TO_TYPE[pending_item["section"]],
                status=EntryStatus(meta.get("status", EntryStatus.ACTIVE.value)),
                content=pending_item["content"],
                context=meta.get("context", ""),
                tags=meta.get("tags", []),
                sensitivity_tags=meta.get("sensitivity_tags", []),
                metadata={"imported_from": "memory.md"},
                links=links,
                confidence=float(meta.get("confidence", 0.5)),
                created_at=_normalize_dt(meta.get("created_at")),
                updated_at=_normalize_dt(meta.get("updated_at")),
                content_hash=compute_content_hash(pending_item["content"]),
                encrypted=bool(meta.get("encrypted", False)),
                redacted=bool(meta.get("redacted", False)),
            )
            entries.append(entry)
            pending_item = None

    entries.sort(key=lambda entry: (_normalize_dt(entry.created_at), entry.id))
    return entries
