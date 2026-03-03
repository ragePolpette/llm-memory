"""Test configuration."""

import pytest


@pytest.fixture
def test_config():
    """Configurazione per i test."""
    from src.config import Config, IndexingMode
    from pathlib import Path
    import tempfile
    
    tmpdir = Path(tempfile.mkdtemp())
    
    return Config(
        storage_dir=tmpdir / "memories",
        lancedb_dir=tmpdir / "lancedb",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dim=384,
        indexing_mode=IndexingMode.SYNC,
    )
