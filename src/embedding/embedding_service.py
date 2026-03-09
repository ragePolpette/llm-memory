"""Embedding layer swappabile, completamente locale."""

from __future__ import annotations

import asyncio
import hashlib
import os
from abc import ABC, abstractmethod
from typing import Optional

from ..config import Config, EmbeddingProviderKind


class EmbeddingProvider(ABC):
    """Interfaccia astratta provider embedding locale."""

    async def prepare(self) -> None:
        """Prepara il provider prima di servire richieste."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Genera embedding per una lista di testi."""

    @abstractmethod
    def dimension(self) -> int:
        """Ritorna la dimensione vettoriale."""

    @abstractmethod
    def provider_id(self) -> str:
        """Identificatore provider."""

    @abstractmethod
    def model_id(self) -> str:
        """Identificatore modello embedding."""

    @abstractmethod
    def fingerprint(self) -> str:
        """Fingerprint deterministico di configurazione+modello."""


class HashEmbeddingProvider(EmbeddingProvider):
    """Embedding deterministico hash-based (offline hard-safe)."""

    def __init__(self, model_name: str = "local-hash-v1", dim: int = 384):
        self._model_name = model_name
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(self._embed_one(text))
        return vectors

    def _embed_one(self, text: str) -> list[float]:
        # Feature hashing su token con segno stabile.
        vector = [0.0] * self._dim
        tokens = [tok for tok in text.lower().split() if tok]
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vector[idx] += sign

        # Normalizzazione L2.
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def dimension(self) -> int:
        return self._dim

    def provider_id(self) -> str:
        return "hash-local"

    def model_id(self) -> str:
        return self._model_name

    def fingerprint(self) -> str:
        raw = f"{self.provider_id()}::{self._model_name}::{self._dim}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SentenceTransformerProvider(EmbeddingProvider):
    """Provider `sentence-transformers` in modalità offline/local-only."""

    def __init__(
        self,
        model_name: str,
        dim_hint: int,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None
        self._dimension: int = dim_hint
        self._load_lock = asyncio.Lock()

    def _load_model(self):
        if self._model is not None:
            return

        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("DO_NOT_TRACK", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers non installato. Usa EMBEDDING_PROVIDER=hash-local "
                "oppure installa la dipendenza localmente."
            ) from exc

        kwargs = {"trust_remote_code": False}
        # `local_files_only` è fondamentale per evitare download accidentalmente.
        kwargs["local_files_only"] = True

        if self.device:
            kwargs["device"] = self.device

        self._model = SentenceTransformer(self.model_name, **kwargs)
        dim = self._model.get_sentence_embedding_dimension()
        if dim:
            self._dimension = int(dim)

    async def prepare(self) -> None:
        if self._model is not None:
            return

        async with self._load_lock:
            if self._model is not None:
                return
            await asyncio.to_thread(self._load_model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        await self.prepare()
        vectors = await asyncio.to_thread(
            self._model.encode,
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return [self._to_float_list(v) for v in vectors]

    @staticmethod
    def _to_float_list(values) -> list[float]:
        if hasattr(values, "tolist"):
            return values.tolist()
        return [float(v) for v in values]

    def dimension(self) -> int:
        return self._dimension

    def provider_id(self) -> str:
        return "sentence-transformers"

    def model_id(self) -> str:
        return self.model_name

    def fingerprint(self) -> str:
        raw = (
            f"{self.provider_id()}::{self.model_name}::{self._dimension}::"
            f"{self.device or 'auto'}::{self.batch_size}::{self.normalize}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_embedding_provider(config: Config) -> EmbeddingProvider:
    """Factory provider embedding locale swappabile."""

    if config.embedding_provider == EmbeddingProviderKind.SENTENCE_TRANSFORMERS:
        return SentenceTransformerProvider(
            model_name=config.embedding_model,
            dim_hint=config.embedding_dim,
            device=config.embedding_device,
            batch_size=config.embedding_batch_size,
        )

    return HashEmbeddingProvider(model_name=config.embedding_model, dim=config.embedding_dim)
