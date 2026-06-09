"""
ChromaDB client and collection management.

Provides a module-level persistent client and a helper to obtain
the main vector collection with the configured embedding function.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

from src.rag_api.config import settings

# ---------------------------------------------------------------------------
# Persistent client (shared across the process)
# ---------------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

# ---------------------------------------------------------------------------
# Embedding function
# ---------------------------------------------------------------------------
embedding_fn = OllamaEmbeddingFunction(
    model_name=settings.OLLAMA_EMBED_MODEL,
    url=settings.OLLAMA_URL,
)


@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    """Return the main document collection, creating it if needed.

    Uses ``lru_cache`` so the collection object is created once and
    reused across all callers.
    """
    return chroma_client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


# Default collection instance for convenience — lazily created on first
# attribute access via module ``__getattr__``.  Existing code that
# imports ``collection`` directly continues to work unchanged.
def __getattr__(name: str):
    if name == "collection":
        return get_collection()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
