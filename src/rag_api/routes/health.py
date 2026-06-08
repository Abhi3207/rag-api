"""
Health check and admin statistics routes.
"""

from __future__ import annotations

from collections import Counter

import ollama
from fastapi import APIRouter, HTTPException

from src.rag_api.config import settings
from src.rag_api.database import collection
from src.rag_api.models import AdminStatsResponse, HealthResponse

router = APIRouter(tags=["System"])


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check connectivity to Ollama and ChromaDB."""
    ollama_status = "ok"
    try:
        ollama.list()
    except Exception:
        ollama_status = "unreachable"

    chroma_status = "ok"
    doc_count = 0
    try:
        doc_count = collection.count()
    except Exception:
        chroma_status = "unreachable"

    overall = "healthy" if ollama_status == "ok" and chroma_status == "ok" else "degraded"
    return HealthResponse(
        status=overall,
        ollama=ollama_status,
        chromadb=chroma_status,
        collection_count=doc_count,
    )


# ---------------------------------------------------------------------------
# GET /admin/stats
# ---------------------------------------------------------------------------

@router.get("/admin/stats", response_model=AdminStatsResponse)
def admin_stats():
    """Return operational statistics about the knowledge base."""
    try:
        all_docs = collection.get()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ChromaDB get failed: {exc}") from exc

    # Count documents per user
    user_counts: Counter[str] = Counter()
    for meta in all_docs.get("metadatas", []):
        user_name = meta.get("user_name", "unknown")
        user_counts[user_name] += 1

    return AdminStatsResponse(
        total_documents=len(all_docs.get("ids", [])),
        users=sorted(user_counts.keys()),
        user_document_counts=dict(user_counts),
        collection_name=settings.COLLECTION_NAME,
        embedding_model=settings.OLLAMA_EMBED_MODEL,
        chat_model=settings.OLLAMA_CHAT_MODEL,
    )
