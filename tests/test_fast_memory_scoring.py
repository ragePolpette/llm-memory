from __future__ import annotations

from src.service.importance_scoring import build_fast_selection_metadata


def test_fast_selection_boosts_recurring_low_meta_entries():
    low_meta = build_fast_selection_metadata(
        metadata={"importance_score": 20, "novelty_score": 0.1},
        recurrence_count=1,
        event_type="note",
    )
    recurring = build_fast_selection_metadata(
        metadata={
            "importance_score": 20,
            "novelty_score": 0.1,
            "distinct_session_count": 3,
            "distinct_day_count": 2,
        },
        recurrence_count=5,
        event_type="incident",
    )

    assert recurring["selection_score"] > low_meta["selection_score"]
    assert recurring["recurrence_score"] > 0.0
    assert recurring["recurrence_boost"] > 0.0


def test_fast_selection_noise_penalty_dampens_retry_loops():
    clean = build_fast_selection_metadata(
        metadata={
            "importance_score": 30,
            "distinct_session_count": 3,
        },
        recurrence_count=4,
        event_type="incident",
    )
    noisy = build_fast_selection_metadata(
        metadata={
            "importance_score": 30,
            "distinct_session_count": 3,
            "duplicate_ratio": 0.9,
            "same_session_ratio": 1.0,
            "burst_retry_count": 5,
        },
        recurrence_count=4,
        event_type="retry",
    )

    assert noisy["noise_penalty"] > clean["noise_penalty"]
    assert noisy["selection_score"] < clean["selection_score"]


def test_fast_selection_high_meta_entries_get_limited_recurrence_boost():
    low_meta = build_fast_selection_metadata(
        metadata={
            "importance_score": 25,
            "distinct_session_count": 4,
            "distinct_day_count": 2,
        },
        recurrence_count=6,
        event_type="incident",
    )
    high_meta = build_fast_selection_metadata(
        metadata={
            "importance_score": 85,
            "novelty_score": 0.8,
            "confidence": 0.9,
            "distinct_session_count": 4,
            "distinct_day_count": 2,
        },
        recurrence_count=6,
        event_type="incident",
    )

    assert high_meta["meta_score"] > low_meta["meta_score"]
    assert high_meta["recurrence_boost"] < low_meta["recurrence_boost"]
    assert 0.0 <= high_meta["selection_score"] <= 1.0
