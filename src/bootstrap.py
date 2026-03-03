"""Bootstrap runtime componenti memoria v2."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .embedding.embedding_service import get_embedding_provider
from .security import PrivacyPolicy, block_outbound_network, build_cipher
from .service.memory_service import MemoryService
from .storage.sqlite_store import SQLiteMemoryStore
from .vectordb.sqlite_vector_store import SQLiteVectorStore


@dataclass
class MemoryRuntime:
    """Container componenti runtime."""

    config: Config
    store: SQLiteMemoryStore
    vector_store: SQLiteVectorStore
    service: MemoryService


def build_runtime(config: Config) -> MemoryRuntime:
    """Costruisce runtime locale completo."""

    if not config.allow_outbound_network:
        block_outbound_network(allow_loopback=True)

    embedding_provider = get_embedding_provider(config)
    store = SQLiteMemoryStore(config.sqlite_db_path)
    vector_store = SQLiteVectorStore(store)

    privacy_policy = PrivacyPolicy(
        sensitive_tags=config.privacy_sensitive_tags,
        drop_metadata_keys=config.privacy_drop_metadata_keys,
        encrypt_sensitive=config.privacy_encrypt_sensitive,
    )
    cipher = build_cipher(config.encryption_enabled, config.encryption_key_env)

    service = MemoryService(
        config=config,
        store=store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        privacy_policy=privacy_policy,
        cipher=cipher,
    )

    return MemoryRuntime(
        config=config,
        store=store,
        vector_store=vector_store,
        service=service,
    )
