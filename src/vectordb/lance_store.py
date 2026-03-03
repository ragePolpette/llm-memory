"""Integrazione LanceDB per indicizzazione vettoriale locale."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa

from ..embedding.embedding_service import EmbeddingProvider
from ..models import Memory, SearchResult

logger = logging.getLogger(__name__)


class LanceVectorStore:
    """
    Vector store basato su LanceDB.
    
    LanceDB è embedded, non richiede Docker, ed è ottimizzato per disk-based storage.
    """
    
    def __init__(self, persist_dir: Path, embedding_provider: EmbeddingProvider):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider
        self.db = lancedb.connect(str(self.persist_dir))
        self._ensure_table()
    
    def _ensure_table(self) -> None:
        """Crea la tabella se non esiste."""
        try:
            if "memories" not in self.db.list_tables():
                dim = self.embedding_provider.dimension()
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                    pa.field("content", pa.string()),
                    pa.field("agent_id", pa.string()),
                    pa.field("scope", pa.string()),
                    pa.field("context", pa.string()),
                    pa.field("tags", pa.list_(pa.string())),
                    pa.field("created_at", pa.string()),
                    pa.field("content_hash", pa.string()),
                ])
                self.db.create_table("memories", schema=schema)
                logger.info(f"Created LanceDB table 'memories' with dimension {dim}")
        except Exception as e:
            # Tabella potrebbe già esistere (race condition)
            logger.warning(f"Table creation skipped: {e}")
        
        self.table = self.db.open_table("memories")
    
    async def index(self, memory: Memory) -> None:
        """
        Indicizza una memoria nel vector store.
        
        Genera l'embedding e lo salva insieme ai metadati.
        """
        # Genera embedding
        embeddings = await self.embedding_provider.embed([memory.content])
        embedding = embeddings[0]
        
        # Prepara record
        record = {
            "id": str(memory.id),
            "vector": embedding,
            "content": memory.content,
            "agent_id": memory.agent_id,
            "scope": memory.scope.value if hasattr(memory.scope, 'value') else memory.scope,
            "context": memory.context,
            "tags": memory.tags,
            "created_at": memory.created_at.isoformat(),
            "content_hash": memory.content_hash,
        }
        
        # Upsert nel database
        self.table.add([record])
        logger.debug(f"Indexed memory {memory.id} in LanceDB")
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[str] = None  # SQL-like: "scope = 'shared' AND agent_id = 'alpha'"
    ) -> list[SearchResult]:
        """
        Ricerca semantica con filtri SQL-like.
        
        Args:
            query: Query testuale
            limit: Numero massimo di risultati
            filters: Filtro SQL-like opzionale
            
        Returns:
            Lista di SearchResult ordinati per rilevanza
        """
        # Genera embedding della query
        query_embeddings = await self.embedding_provider.embed([query])
        query_vec = query_embeddings[0]
        
        # Esegui ricerca
        search = self.table.search(query_vec).limit(limit)
        
        if filters:
            search = search.where(filters)
        
        results = search.to_list()
        
        # Converti in SearchResult
        return [
            SearchResult(
                memory_id=r["id"],
                content=r["content"],
                context=r["context"],
                agent_id=r["agent_id"],
                scope=r["scope"],
                score=float(r.get("_distance", 0.0)),  # LanceDB usa _distance
                tags=r.get("tags", []),
                created_at=r["created_at"],
                indexed=True,
            )
            for r in results
        ]
    
    async def check_duplicate(self, content_hash: str) -> Optional[str]:
        """
        Verifica se esiste già una memoria con lo stesso content hash.
        
        Usato per deduplicazione.
        """
        try:
            results = (
                self.table.search()
                .where(f"content_hash = '{content_hash}'")
                .limit(1)
                .to_list()
            )
            return results[0]["id"] if results else None
        except Exception as e:
            logger.warning(f"Error checking duplicate: {e}")
            return None
    
    async def delete(self, memory_id: str) -> None:
        """
        Rimuove una memoria dall'indice.
        
        Nota: questo è un soft delete nell'indice, il file MD rimane.
        """
        try:
            self.table.delete(f"id = '{memory_id}'")
            logger.info(f"Deleted memory {memory_id} from index")
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
