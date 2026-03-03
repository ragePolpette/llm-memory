"""Test embedding service."""

import pytest
from src.embedding.embedding_service import get_embedding_provider


@pytest.mark.asyncio
async def test_embedding_generation():
    """Test generazione embedding."""
    provider = get_embedding_provider()
    
    texts = ["Hello world", "Test embedding"]
    embeddings = await provider.embed(texts)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384  # multilingual-e5-small dimension
    assert all(isinstance(v, float) for v in embeddings[0])


@pytest.mark.asyncio
async def test_embedding_dimension():
    """Test dimensione embedding."""
    provider = get_embedding_provider()
    dim = provider.dimension()
    
    assert dim == 384
