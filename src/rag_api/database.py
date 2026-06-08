"""
ChromaDB client and collection management.

Provides a module-level persistent client and a helper to obtain
the main vector collection with the configured embedding function.
"""

from __future__ import annotations

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


def get_collection() -> chromadb.Collection:
    """Return the main document collection, creating it if needed."""
    return chroma_client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


# Default collection instance for convenience
collection = get_collection()
