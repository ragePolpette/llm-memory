"""Test integration end-to-end."""

import pytest
from src.config import MemoryScope
from src.coordination.conflict_resolver import ConflictResolver
from src.coordination.scope_manager import ScopeManager
from src.embedding.embedding_service import get_embedding_provider
from src.indexing.indexer import MemoryIndexer
from src.models import Memory
from src.storage.markdown_store import MarkdownStore
from src.vectordb.lance_store import LanceVectorStore


@pytest.mark.asyncio
async def test_full_write_search_flow(test_config):
    """Test completo: write -> index -> search."""
    
    # Setup componenti
    embedding_provider = get_embedding_provider()
    markdown_store = MarkdownStore(test_config.storage_dir)
    vector_store = LanceVectorStore(test_config.lancedb_dir, embedding_provider)
    indexer = MemoryIndexer(vector_store, mode=test_config.indexing_mode)
    scope_manager = ScopeManager()
    conflict_resolver = ConflictResolver(markdown_store, vector_store)
    
    # Crea memoria
    memory = Memory(
        content="Python is a programming language used for AI and data science",
        context="programming_knowledge",
        agent_id="agent-alpha",
        scope=MemoryScope.SHARED,
        tags=["python", "programming"],
    )
    
    # Scrivi
    await markdown_store.write(memory)
    
    # Indicizza
    result = await indexer.index(memory)
    assert result.indexed is True
    
    # Cerca
    search_results = await vector_store.search(
        query="What is Python used for?",
        limit=5
    )
    
    assert len(search_results) > 0
    assert search_results[0].memory_id == memory.id


@pytest.mark.asyncio
async def test_scope_permissions(test_config):
    """Test permessi scope."""
    scope_manager = ScopeManager()
    
    # Private memory
    private_memory = Memory(
        content="Private data",
        context="test",
        agent_id="agent-alpha",
        scope=MemoryScope.PRIVATE,
    )
    
    # Agent proprietario può leggere
    assert scope_manager.can_read("agent-alpha", private_memory) is True
    
    # Altro agente non può leggere
    assert scope_manager.can_read("agent-beta", private_memory) is False
    
    # Shared memory
    shared_memory = Memory(
        content="Shared data",
        context="test",
        agent_id="agent-alpha",
        scope=MemoryScope.SHARED,
    )
    
    # Tutti possono leggere
    assert scope_manager.can_read("agent-beta", shared_memory) is True
