from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.mcp_server.tools import _require_explicit_project_scope


def _service_stub(*, multi_project_enabled: bool):
    return SimpleNamespace(config=SimpleNamespace(multi_project_enabled=multi_project_enabled))


def test_multi_project_scope_guard_rejects_missing_project_scope():
    with pytest.raises(ValueError, match="explicit scope.project_id"):
        _require_explicit_project_scope(
            "memory.search",
            {"query": "hello", "agent_id": "agent-a"},
            _service_stub(multi_project_enabled=True),
        )


def test_multi_project_scope_guard_allows_workspace_scope_without_project_id():
    _require_explicit_project_scope(
        "memory.search",
        {"query": "hello", "agent_id": "agent-a", "scope": {"scope_level": "workspace"}},
        _service_stub(multi_project_enabled=True),
    )


def test_multi_project_scope_guard_is_disabled_in_single_project_mode():
    _require_explicit_project_scope(
        "memory.search",
        {"query": "hello", "agent_id": "agent-a"},
        _service_stub(multi_project_enabled=False),
    )
