"""Test configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.bootstrap import build_runtime
from src.config import Config
from src.security.no_network import restore_network


@pytest.fixture(autouse=True)
def _cleanup_network_guard():
    restore_network()
    yield
    restore_network()


@pytest.fixture
def test_config() -> Config:
    tmpdir = Path(tempfile.mkdtemp())
    return Config(
        storage_dir=tmpdir / "memories",
        lancedb_dir=tmpdir / "lancedb",
        sqlite_db_path=tmpdir / "memory.db",
        embedding_provider="hash-local",
        embedding_model="local-hash-test",
        embedding_dim=96,
        allow_outbound_network=False,
        default_workspace_id="ws-test",
        default_project_id="prj-test",
        dedup_semantic_threshold=0.92,
    )


@pytest.fixture
def runtime(test_config):
    return build_runtime(test_config)


@pytest.fixture
def service(runtime):
    return runtime.service
