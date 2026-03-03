"""Test storage layer."""

import pytest
from src.config import MemoryScope
from src.models import Memory
from src.storage.markdown_store import MarkdownStore


@pytest.mark.asyncio
async def test_write_and_read_memory(test_config):
    """Test scrittura e lettura memoria."""
    store = MarkdownStore(test_config.storage_dir)
    
    memory = Memory(
        content="Test content for memory system",
        context="testing",
        agent_id="test-agent",
        scope=MemoryScope.SHARED,
        tags=["test", "demo"],
    )
    
    # Scrivi
    path = await store.write(memory)
    assert path.exists()
    
    # Leggi
    read_memory = await store.read(str(memory.id))
    assert read_memory is not None
    assert read_memory.content == memory.content
    assert read_memory.agent_id == memory.agent_id


@pytest.mark.asyncio
async def test_list_memories(test_config):
    """Test listing memorie."""
    store = MarkdownStore(test_config.storage_dir)
    
    # Crea alcune memorie
    for i in range(3):
        memory = Memory(
            content=f"Memory {i}",
            context="test",
            agent_id="test-agent",
            scope=MemoryScope.SHARED,
        )
        await store.write(memory)
    
    # Lista
    paths = await store.list_memories(scope=MemoryScope.SHARED, limit=10)
    assert len(paths) >= 3


@pytest.mark.asyncio
async def test_duplicate_detection(test_config):
    """Test rilevamento duplicati."""
    store = MarkdownStore(test_config.storage_dir)
    
    memory = Memory(
        content="Duplicate test content",
        context="test",
        agent_id="test-agent",
        scope=MemoryScope.SHARED,
    )
    
    await store.write(memory)
    
    # Cerca per hash
    found_id = await store.find_by_hash(memory.content_hash)
    assert found_id == str(memory.id)
