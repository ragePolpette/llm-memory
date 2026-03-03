"""Servizio di embedding locale e multilingua."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Interfaccia astratta per provider di embedding."""
    
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Genera embedding per una lista di testi."""
        pass
    
    @abstractmethod
    def dimension(self) -> int:
        """Ritorna la dimensione del vettore embedding."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Provider di embedding basato su sentence-transformers.
    
    Supporta modelli locali multilingua senza chiamate a servizi esterni.
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = None
        self._dimension: Optional[int] = None
    
    def _load_model(self):
        """Lazy loading del modello."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install sentence-transformers: pip install sentence-transformers"
                ) from exc
            
            logger.info(f"Loading embedding model: {self.model_name}")
            if self.device:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            else:
                self._model = SentenceTransformer(self.model_name)
            
            # Determina dimensione dal modello
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self._dimension}")
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Genera embedding per una lista di testi.
        
        Args:
            texts: Lista di stringhe da codificare
            
        Returns:
            Lista di vettori float
        """
        if not texts:
            return []
        
        self._load_model()
        
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        
        return [self._to_float_list(v) for v in vectors]
    
    async def embed_single(self, text: str) -> list[float]:
        """Genera embedding per un singolo testo."""
        result = await self.embed([text])
        return result[0] if result else []
    
    def dimension(self) -> int:
        """Ritorna la dimensione del vettore embedding."""
        self._load_model()
        return self._dimension or 384
    
    @staticmethod
    def _to_float_list(values) -> list[float]:
        """Converte qualsiasi tipo iterabile in list[float]."""
        if hasattr(values, "tolist"):
            return values.tolist()
        return [float(v) for v in values]


def get_embedding_provider(
    model_name: str = "intfloat/multilingual-e5-small",
    device: Optional[str] = None
) -> EmbeddingProvider:
    """Factory per creare il provider di embedding."""
    return SentenceTransformerProvider(model_name=model_name, device=device)
