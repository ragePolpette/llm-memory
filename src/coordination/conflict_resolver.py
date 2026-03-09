"""Strategie anti-conflitto per scritture concorrenti."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from filelock import FileLock

from ..models import Memory
from ..storage.markdown_store import MarkdownStore
from ..vectordb.lance_store import LanceVectorStore

logger = logging.getLogger(__name__)


class ConflictResolver:
    """
    Gestisce conflitti e deduplicazione nelle scritture.
    
    Strategie:
    1. UUID univoci per ogni memoria (no collisioni)
    2. Append-only storage (no riscritture)
    3. File locking per scritture concorrenti
    4. Deduplication basata su content hash
    """
    
    def __init__(
        self,
        markdown_store: MarkdownStore,
        vector_store: LanceVectorStore
    ):
        self.markdown_store = markdown_store
        self.vector_store = vector_store
    
    async def check_duplicate(self, memory: Memory) -> Optional[str]:
        """
        Verifica se esiste già una memoria con contenuto identico.
        
        Controlla sia nel vector store che nel filesystem.
        
        Args:
            memory: Memoria da verificare
            
        Returns:
            ID della memoria duplicata se trovata, None altrimenti
        """
        # Check nel vector store (più veloce)
        duplicate_id = await self.vector_store.check_duplicate(memory.content_hash)
        if duplicate_id:
            logger.info(f"Duplicate found in vector store: {duplicate_id}")
            return duplicate_id
        
        # Fallback: check nel filesystem
        duplicate_id = await self.markdown_store.find_by_hash(memory.content_hash)
        if duplicate_id:
            logger.info(f"Duplicate found in filesystem: {duplicate_id}")
            return duplicate_id
        
        return None
    
    def acquire_write_lock(self, base_path: Path, memory_id: str) -> FileLock:
        """
        Acquisisce un lock esclusivo per scrittura.
        
        Previene race condition quando più agenti scrivono simultaneamente.
        
        Args:
            base_path: Directory base
            memory_id: ID della memoria
            
        Returns:
            FileLock context manager
        """
        lock_file = base_path / f".lock_{memory_id}"
        return FileLock(str(lock_file), timeout=10)
