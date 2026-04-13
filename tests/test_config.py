from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Config, get_config


def test_config_rejects_invalid_mcp_port():
    with pytest.raises(ValidationError, match="mcp_port"):
        Config(mcp_port=70000)


def test_config_rejects_out_of_range_ranking_weight():
    with pytest.raises(ValidationError, match="ranking_similarity_weight"):
        Config(ranking_similarity_weight=1.5)


def test_config_rejects_zero_total_ranking_weight():
    with pytest.raises(ValidationError, match="ranking weights must have a total greater than 0"):
        Config(
            ranking_similarity_weight=0.0,
            ranking_recency_weight=0.0,
            ranking_tier_weight=0.0,
            ranking_status_weight=0.0,
        )


def test_get_config_creates_runtime_directories(monkeypatch, tmp_path: Path):
    sqlite_path = tmp_path / "data" / "memory.db"
    exchange_path = tmp_path / "exchange"
    models_path = tmp_path / "models"

    monkeypatch.setenv("MEMORY_SQLITE_PATH", str(sqlite_path))
    monkeypatch.setenv("MEMORY_IMPORT_EXPORT_BASE_DIR", str(exchange_path))
    monkeypatch.setenv("MCP_MODELS_DIR", str(models_path))
    monkeypatch.setenv("HF_HOME", str(models_path / "huggingface"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(models_path / "huggingface" / "transformers"))
    monkeypatch.setenv(
        "SENTENCE_TRANSFORMERS_HOME",
        str(models_path / "huggingface" / "sentence_transformers"),
    )

    config = get_config()

    assert config.sqlite_db_path.parent.exists()
    assert config.import_export_base_dir.exists()
    assert config.mcp_models_dir.exists()
    assert config.hf_home.exists()
    assert config.transformers_cache.exists()
    assert config.sentence_transformers_home.exists()


def test_startup_diagnostics_contains_runtime_summary(tmp_path: Path):
    config = Config(
        sqlite_db_path=tmp_path / "memory.db",
        import_export_base_dir=tmp_path / "exchange",
        embedding_model="local-hash-test",
        embedding_dim=96,
        mcp_port=8767,
        allow_outbound_network=False,
        multi_project_enabled=True,
        self_eval_enforced=True,
    )

    diagnostics = config.startup_diagnostics()

    assert diagnostics["sqlite_db"] == str(tmp_path / "memory.db")
    assert diagnostics["import_export_base_dir"] == str(tmp_path / "exchange")
    assert diagnostics["embedding_model"] == "local-hash-test"
    assert diagnostics["embedding_dim"] == 96
    assert diagnostics["multi_project_enabled"] is True
    assert diagnostics["self_eval_enforced"] is True
    assert diagnostics["fast_memory_agent_distillation_enabled"] is False
    assert diagnostics["fast_memory_agent_distillation_apply_enabled"] is False
