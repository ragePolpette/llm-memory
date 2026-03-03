"""Configurazione centralizzata per LLM Memory."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class IndexingMode(str, Enum):
    """Modalità di indicizzazione."""
    SYNC = "sync"      # Indicizza subito, bloccante
    ASYNC = "async"    # Coda background
    HYBRID = "hybrid"  # Sync per piccoli, async per grandi


class MemoryScope(str, Enum):
    """Scope accessibilità memoria."""
    PRIVATE = "private"   # Solo l'agente proprietario
    SHARED = "shared"     # Tutti gli agenti autenticati
    GLOBAL = "global"     # Pubblico, read-only per tutti


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path_from_env(env_key: str, default_relative: str) -> Path:
    raw_value = os.getenv(env_key, default_relative)
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


class Config(BaseModel):
    """Configurazione principale del sistema."""
    
    # Storage
    storage_dir: Path = Field(
        default_factory=lambda: _project_path_from_env("MEMORY_STORAGE_DIR", "./memories")
    )
    
    # LanceDB
    lancedb_dir: Path = Field(
        default_factory=lambda: _project_path_from_env("LANCEDB_DIR", "./data/lancedb")
    )

    # Local model directories
    mcp_models_dir: Path = Field(
        default_factory=lambda: _project_path_from_env("MCP_MODELS_DIR", "./.local/models")
    )
    hf_home: Path = Field(
        default_factory=lambda: _project_path_from_env("HF_HOME", "./.local/models/huggingface")
    )
    transformers_cache: Path = Field(
        default_factory=lambda: _project_path_from_env(
            "TRANSFORMERS_CACHE",
            "./.local/models/huggingface/transformers",
        )
    )
    sentence_transformers_home: Path = Field(
        default_factory=lambda: _project_path_from_env(
            "SENTENCE_TRANSFORMERS_HOME",
            "./.local/models/huggingface/sentence_transformers",
        )
    )
    
    # Embedding
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", 
            "intfloat/multilingual-e5-small"
        )
    )
    embedding_dim: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "384"))
    )
    embedding_device: Optional[str] = Field(
        default_factory=lambda: os.getenv("EMBEDDING_DEVICE")
    )
    
    # Indexing
    indexing_mode: IndexingMode = Field(
        default_factory=lambda: IndexingMode(
            os.getenv("INDEXING_MODE", "sync").lower()
        )
    )
    hybrid_threshold_bytes: int = Field(
        default_factory=lambda: int(os.getenv("HYBRID_THRESHOLD_BYTES", "1024"))
    )
    
    # Queue
    queue_max_size: int = Field(
        default_factory=lambda: int(os.getenv("QUEUE_MAX_SIZE", "1000"))
    )
    
    model_config = ConfigDict(use_enum_values=True)


def get_config() -> Config:
    """Crea e ritorna la configurazione dal environment."""
    config = Config()
    os.environ.setdefault("MCP_MODELS_DIR", str(config.mcp_models_dir))
    os.environ.setdefault("HF_HOME", str(config.hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(config.transformers_cache))
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME",
        str(config.sentence_transformers_home),
    )
    return config
