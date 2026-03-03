"""Utility per scaricare il modello embedding in locale."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model(model_name: str = "intfloat/multilingual-e5-small") -> Path:
    """
    Scarica il modello embedding in locale.
    
    Il modello viene salvato dentro la cartella progetto:
    - <project-root>/.local/models/huggingface/
    
    Args:
        model_name: Nome del modello da scaricare
        
    Returns:
        Path alla directory del modello
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed")
        logger.error("Run: pip install sentence-transformers")
        sys.exit(1)
    
    project_root = Path(__file__).resolve().parent.parent
    models_dir = Path(os.getenv("MCP_MODELS_DIR", project_root / ".local" / "models"))
    if not models_dir.is_absolute():
        models_dir = (project_root / models_dir).resolve()
    hf_home = Path(os.getenv("HF_HOME", models_dir / "huggingface"))
    if not hf_home.is_absolute():
        hf_home = (project_root / hf_home).resolve()
    transformers_cache = Path(os.getenv("TRANSFORMERS_CACHE", hf_home / "transformers"))
    if not transformers_cache.is_absolute():
        transformers_cache = (project_root / transformers_cache).resolve()
    st_home = Path(
        os.getenv("SENTENCE_TRANSFORMERS_HOME", hf_home / "sentence_transformers")
    )
    if not st_home.is_absolute():
        st_home = (project_root / st_home).resolve()

    hf_home.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    st_home.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MCP_MODELS_DIR", str(models_dir))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(st_home))

    logger.info(f"Downloading model: {model_name}")
    logger.info(f"MCP_MODELS_DIR: {models_dir}")
    logger.info("This may take a few minutes on first run...")
    
    # Il download avviene automaticamente al primo caricamento
    model = SentenceTransformer(model_name)
    
    # Ottieni il path del modello
    model_path = Path(model._model_card_vars.get("model_path", ""))
    if not model_path.exists():
        # Fallback: cerca nella cache di HF
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(repo_id=model_name, cache_dir=str(hf_home)))
    
    logger.info(f"Model downloaded successfully to: {model_path}")
    logger.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Test rapido
    logger.info("Testing model...")
    test_embedding = model.encode(["Hello world"], show_progress_bar=False)
    logger.info(f"Test embedding shape: {test_embedding.shape}")
    logger.info("✓ Model is working correctly")
    
    return model_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download embedding model")
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-small",
        help="Model name to download"
    )
    
    args = parser.parse_args()
    download_model(args.model)
