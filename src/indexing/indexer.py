"""Pipeline di indicizzazione ibrida (sync/async)."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..config import IndexingMode
from ..models import IndexResult, Memory
from ..vectordb.lance_store import LanceVectorStore

logger = logging.getLogger(__name__)


class MemoryIndexer:
    """
    Indexer con supporto modalità sync, async, e hybrid.
    
    - SYNC: indicizza immediatamente (bloccante, ~50-200ms)
    - ASYNC: accoda per indicizzazione background
    - HYBRID: sync per contenuti piccoli, async per grandi
    """
    
    def __init__(
        self,
        vector_store: LanceVectorStore,
        mode: IndexingMode = IndexingMode.SYNC,
        hybrid_threshold_bytes: int = 1024,
        queue_max_size: int = 1000,
    ):
        self.vector_store = vector_store
        self.mode = mode
        self.hybrid_threshold = hybrid_threshold_bytes
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def index(self, memory: Memory) -> IndexResult:
        """
        Indicizza una memoria secondo la modalità configurata.
        
        Args:
            memory: Memoria da indicizzare
            
        Returns:
            IndexResult con stato indicizzazione
        """
        if self.mode == IndexingMode.SYNC:
            return await self._index_sync(memory)
        elif self.mode == IndexingMode.ASYNC:
            return await self._index_async(memory)
        else:  # HYBRID
            content_size = len(memory.content.encode("utf-8"))
            if content_size < self.hybrid_threshold:
                return await self._index_sync(memory)
            return await self._index_async(memory)
    
    async def _index_sync(self, memory: Memory) -> IndexResult:
        """Indicizzazione sincrona immediata."""
        try:
            await self.vector_store.index(memory)
            return IndexResult(indexed=True, mode="sync")
        except Exception as e:
            logger.error(f"Error indexing memory {memory.id}: {e}")
            return IndexResult(indexed=False, mode="sync", error=str(e))
    
    async def _index_async(self, memory: Memory) -> IndexResult:
        """Aggiunge alla coda per indicizzazione background."""
        try:
            await self._queue.put(memory)
            return IndexResult(indexed=False, mode="async", queued=True)
        except asyncio.QueueFull:
            logger.warning(f"Queue full, falling back to sync for {memory.id}")
            return await self._index_sync(memory)
    
    async def start_worker(self) -> None:
        """
        Avvia il worker background per modalità async.
        
        Il worker processa la coda con timeout di 5 secondi.
        """
        if self._worker_task is not None:
            logger.warning("Worker already running")
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info("Indexing worker started")
    
    async def stop_worker(self) -> None:
        """Ferma il worker background."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.info("Indexing worker stopped")
    
    async def _process_queue(self) -> None:
        """Worker che processa la coda in background."""
        logger.info("Queue processor started")
        
        while self._running:
            try:
                # Wait max 5 secondi per un item
                memory = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                
                try:
                    await self.vector_store.index(memory)
                    logger.debug(f"Background indexed memory {memory.id}")
                except Exception as e:
                    logger.error(f"Error in background indexing {memory.id}: {e}")
                finally:
                    self._queue.task_done()
                    
            except asyncio.TimeoutError:
                # Nessun item nella coda, continua polling
                continue
            except Exception as e:
                logger.error(f"Unexpected error in queue processor: {e}")
                await asyncio.sleep(1)  # Backoff
        
        logger.info("Queue processor stopped")
    
    async def wait_for_queue(self, timeout: Optional[float] = None) -> bool:
        """
        Attende che la coda sia vuota.
        
        Utile per testing o shutdown graceful.
        """
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    def queue_size(self) -> int:
        """Ritorna il numero di item in coda."""
        return self._queue.qsize()
