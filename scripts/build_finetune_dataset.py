"""Build a fine-tuning dataset with novelty/external guards from llm-memory SQLite."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Candidate:
    entry_id: str
    content: str
    metadata: dict
    importance_score: int
    novelty_score: float
    is_external: bool


def load_candidates(db_path: Path) -> list[Candidate]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, content, metadata_json
            FROM entries
            WHERE status != 'invalidated'
            ORDER BY updated_at DESC
            """
        ).fetchall()
    finally:
        conn.close()

    candidates: list[Candidate] = []
    for row in rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        importance_score = int(metadata.get("importance_score", 0))
        novelty_score = float(metadata.get("novelty_score", 1.0))
        is_external = bool(metadata.get("is_external", False))
        candidates.append(
            Candidate(
                entry_id=row["id"],
                content=row["content"],
                metadata=metadata,
                importance_score=max(0, min(100, importance_score)),
                novelty_score=max(0.0, min(1.0, novelty_score)),
                is_external=is_external,
            )
        )
    return candidates


def split_buckets(candidates: list[Candidate]) -> tuple[list[Candidate], list[Candidate], list[Candidate]]:
    top = [c for c in candidates if c.importance_score >= 70]
    mid = [c for c in candidates if 40 <= c.importance_score <= 69]
    low = [c for c in candidates if c.importance_score <= 39]
    return top, mid, low


def sample_bucket(rng: random.Random, items: list[Candidate], count: int) -> list[Candidate]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    return rng.sample(items, count)


def ensure_external_quota(
    selected: list[Candidate],
    pool: list[Candidate],
    *,
    minimum_ratio: float,
    rng: random.Random,
) -> list[Candidate]:
    if not selected:
        return selected

    minimum_ratio = max(0.0, min(1.0, minimum_ratio))
    required_external = int(round(len(selected) * minimum_ratio))
    current_external = sum(1 for item in selected if item.is_external)
    if current_external >= required_external:
        return selected

    need = required_external - current_external
    external_pool = [c for c in pool if c.is_external and c.entry_id not in {s.entry_id for s in selected}]
    internal_selected = [c for c in selected if not c.is_external]
    if not external_pool or not internal_selected:
        return selected

    swaps = min(need, len(external_pool), len(internal_selected))
    add_items = sample_bucket(rng, external_pool, swaps)
    drop_items = sample_bucket(rng, internal_selected, swaps)
    drop_ids = {item.entry_id for item in drop_items}
    merged = [item for item in selected if item.entry_id not in drop_ids]
    merged.extend(add_items)
    return merged


def build_dataset(
    candidates: list[Candidate],
    *,
    novelty_min: float,
    sample_size: int | None,
    top_ratio: float,
    mid_ratio: float,
    low_ratio: float,
    external_min_ratio: float,
    seed: int,
) -> list[Candidate]:
    filtered = [c for c in candidates if c.novelty_score >= novelty_min]
    if not filtered:
        return []

    rng = random.Random(seed)
    if sample_size is None:
        sample_size = len(filtered)
    sample_size = max(1, min(sample_size, len(filtered)))

    top, mid, low = split_buckets(filtered)
    top_target = int(round(sample_size * top_ratio))
    mid_target = int(round(sample_size * mid_ratio))
    low_target = sample_size - top_target - mid_target
    low_target = max(0, low_target)

    selected: list[Candidate] = []
    selected.extend(sample_bucket(rng, top, top_target))
    selected.extend(sample_bucket(rng, mid, mid_target))
    selected.extend(sample_bucket(rng, low, low_target))

    if len(selected) < sample_size:
        selected_ids = {item.entry_id for item in selected}
        remainder = [c for c in filtered if c.entry_id not in selected_ids]
        selected.extend(sample_bucket(rng, remainder, sample_size - len(selected)))

    selected = ensure_external_quota(
        selected,
        filtered,
        minimum_ratio=external_min_ratio,
        rng=rng,
    )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FT dataset from llm-memory metadata.")
    parser.add_argument("--db", required=True, type=Path, help="Path to memory.db")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--novelty-min", type=float, default=0.2)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--top-ratio", type=float, default=0.60)
    parser.add_argument("--mid-ratio", type=float, default=0.25)
    parser.add_argument("--low-ratio", type=float, default=0.15)
    parser.add_argument("--external-min-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratio_sum = args.top_ratio + args.mid_ratio + args.low_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit("top/mid/low ratios must sum to 1.0")

    candidates = load_candidates(args.db)
    selected = build_dataset(
        candidates,
        novelty_min=max(0.0, min(1.0, args.novelty_min)),
        sample_size=args.sample_size,
        top_ratio=args.top_ratio,
        mid_ratio=args.mid_ratio,
        low_ratio=args.low_ratio,
        external_min_ratio=max(0.0, min(1.0, args.external_min_ratio)),
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in selected:
            record = {
                "entry_id": item.entry_id,
                "content": item.content,
                "metadata": item.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=True))
            f.write("\n")

    external_count = sum(1 for item in selected if item.is_external)
    print(
        json.dumps(
            {
                "candidates": len(candidates),
                "selected": len(selected),
                "external_count": external_count,
                "external_ratio": (external_count / len(selected)) if selected else 0.0,
                "output": str(args.output),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
