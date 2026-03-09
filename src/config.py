"""Configurazione centralizzata per LLM Memory."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class MemoryScope(str, Enum):
    """Visibilita della memoria nel runtime v2."""

    PRIVATE = "private"
    SHARED = "shared"
    GLOBAL = "global"


class Tier(str, Enum):
    """Tier di memoria."""

    TIER_1 = "tier-1"
    TIER_2 = "tier-2"
    TIER_3 = "tier-3"


class StorageBackend(str, Enum):
    """Backend metadata store."""

    SQLITE = "sqlite"


class VectorBackend(str, Enum):
    """Backend vector store."""

    SQLITE = "sqlite"


class EmbeddingProviderKind(str, Enum):
    """Provider embedding supportati."""

    HASH_LOCAL = "hash-local"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path_from_env(env_key: str, default_relative: str) -> Path:
    raw_value = os.getenv(env_key, default_relative)
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(key: str, default: list[str]) -> list[str]:
    raw = os.getenv(key)
    if raw is None:
        return default
    values = [v.strip() for v in raw.split(",")]
    return [v for v in values if v]


class Config(BaseModel):
    """Configurazione principale del sistema."""

    # Runtime storage (v2)
    storage_backend: StorageBackend = Field(
        default_factory=lambda: StorageBackend(os.getenv("MEMORY_STORAGE_BACKEND", "sqlite"))
    )
    vector_backend: VectorBackend = Field(
        default_factory=lambda: VectorBackend(os.getenv("MEMORY_VECTOR_BACKEND", "sqlite"))
    )
    sqlite_db_path: Path = Field(
        default_factory=lambda: _project_path_from_env("MEMORY_SQLITE_PATH", "./data/memory.db")
    )
    import_export_base_dir: Path = Field(
        default_factory=lambda: _project_path_from_env(
            "MEMORY_IMPORT_EXPORT_BASE_DIR",
            "./data/exchange",
        )
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
    embedding_provider: EmbeddingProviderKind = Field(
        default_factory=lambda: EmbeddingProviderKind(
            os.getenv("EMBEDDING_PROVIDER", "hash-local")
        )
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "local-hash-v1")
    )
    embedding_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "384")))
    embedding_device: Optional[str] = Field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE"))
    embedding_batch_size: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    )

    # Governance
    dedup_hash_enabled: bool = Field(default_factory=lambda: _env_bool("DEDUP_HASH_ENABLED", True))
    dedup_semantic_enabled: bool = Field(
        default_factory=lambda: _env_bool("DEDUP_SEMANTIC_ENABLED", True)
    )
    dedup_semantic_threshold: float = Field(
        default_factory=lambda: float(os.getenv("DEDUP_SEMANTIC_THRESHOLD", "0.97"))
    )
    promotion_default_target_tier: Tier = Field(
        default_factory=lambda: Tier(os.getenv("PROMOTION_TARGET_TIER", "tier-3"))
    )
    self_eval_enforced: bool = Field(
        default_factory=lambda: _env_bool("MEMORY_SELF_EVAL_ENFORCED", False)
    )

    # Ranking
    ranking_similarity_weight: float = Field(
        default_factory=lambda: float(os.getenv("RANKING_SIMILARITY_WEIGHT", "0.55"))
    )
    ranking_recency_weight: float = Field(
        default_factory=lambda: float(os.getenv("RANKING_RECENCY_WEIGHT", "0.2"))
    )
    ranking_tier_weight: float = Field(
        default_factory=lambda: float(os.getenv("RANKING_TIER_WEIGHT", "0.2"))
    )
    ranking_status_weight: float = Field(
        default_factory=lambda: float(os.getenv("RANKING_STATUS_WEIGHT", "0.05"))
    )

    # Default scopes/context
    default_workspace_id: str = Field(default_factory=lambda: os.getenv("MEMORY_WORKSPACE_ID", "default"))
    default_project_id: str = Field(default_factory=lambda: os.getenv("MEMORY_PROJECT_ID", "default"))

    # Privacy policy
    privacy_sensitive_tags: list[str] = Field(
        default_factory=lambda: _env_csv(
            "MEMORY_PRIVACY_SENSITIVE_TAGS",
            ["pii", "secret", "credential"],
        )
    )
    privacy_drop_metadata_keys: list[str] = Field(
        default_factory=lambda: _env_csv(
            "MEMORY_PRIVACY_DROP_METADATA_KEYS",
            ["password", "token", "secret", "api_key"],
        )
    )
    privacy_encrypt_sensitive: bool = Field(
        default_factory=lambda: _env_bool("MEMORY_PRIVACY_ENCRYPT_SENSITIVE", False)
    )

    # Encryption
    encryption_enabled: bool = Field(default_factory=lambda: _env_bool("MEMORY_ENCRYPTION_ENABLED", False))
    encryption_key_env: str = Field(default_factory=lambda: os.getenv("MEMORY_ENCRYPTION_KEY_ENV", "MEMORY_ENCRYPTION_KEY"))

    # Security
    allow_outbound_network: bool = Field(
        default_factory=lambda: _env_bool("MEMORY_ALLOW_OUTBOUND_NETWORK", False)
    )

    # MCP transport (HTTP)
    mcp_host: str = Field(default_factory=lambda: os.getenv("MCP_MEMORY_HOST", "127.0.0.1"))
    mcp_port: int = Field(default_factory=lambda: int(os.getenv("MCP_MEMORY_PORT", "8767")))
    mcp_sse_enabled: bool = Field(
        default_factory=lambda: _env_bool("MCP_MEMORY_SSE_ENABLED", False)
    )
    mcp_allowed_hosts: list[str] = Field(
        default_factory=lambda: _env_csv(
            "MCP_MEMORY_ALLOWED_HOSTS",
            ["localhost:*", "127.0.0.1:*", "[::1]:*"],
        )
    )
    mcp_allowed_origins: list[str] = Field(
        default_factory=lambda: _env_csv(
            "MCP_MEMORY_ALLOWED_ORIGINS",
            [
                "http://localhost:*",
                "http://127.0.0.1:*",
                "https://localhost:*",
                "https://127.0.0.1:*",
            ],
        )
    )

    model_config = ConfigDict(use_enum_values=True)


def get_config() -> Config:
    """Crea e ritorna la configurazione dal environment."""

    config = Config()
    os.environ.setdefault("MCP_MODELS_DIR", str(config.mcp_models_dir))
    os.environ.setdefault("HF_HOME", str(config.hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(config.transformers_cache))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(config.sentence_transformers_home))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    return config
