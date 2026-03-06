from __future__ import annotations

import pytest

from src.vectordb.lance_store import LanceVectorStore


def test_lance_filter_sanitizer_allows_safe_and_clauses():
    clause = LanceVectorStore._sanitize_filters("scope = 'shared' AND agent_id = 'alpha'")
    assert clause == "scope = 'shared' AND agent_id = 'alpha'"


def test_lance_filter_sanitizer_escapes_embedded_quotes():
    clause = LanceVectorStore._sanitize_filters("context = 'Bob''s note'")
    assert clause == "context = 'Bob''s note'"


@pytest.mark.parametrize(
    "raw_filter",
    [
        "scope = 'shared' OR agent_id = 'alpha'",
        "scope = 'shared'; delete from memories",
        "vector = 'x'",
    ],
)
def test_lance_filter_sanitizer_rejects_unsafe_filters(raw_filter: str):
    with pytest.raises(ValueError):
        LanceVectorStore._sanitize_filters(raw_filter)


def test_lance_equality_clause_escapes_values():
    clause = LanceVectorStore._build_equality_clause("content_hash", "abc' OR '1'='1")
    assert clause == "content_hash = 'abc'' OR ''1''=''1'"
