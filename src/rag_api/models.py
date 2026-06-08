"""
Pydantic request / response models for the RAG API.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.rag_api.config import settings


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChunkStrategy(str, Enum):
    """Supported document chunking strategies."""
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentSubmission(BaseModel):
    """JSON body for adding document chunks."""
    user_name: str = Field(..., min_length=1, description="Owner of this profile")
    content: str = Field(..., min_length=1, description="Profile text to chunk and store")
    chunk_strategy: ChunkStrategy = Field(
        default=ChunkStrategy(settings.DEFAULT_CHUNK_STRATEGY),
        description="Chunking strategy to apply",
    )
    chunk_size: int = Field(
        default=settings.DEFAULT_CHUNK_SIZE,
        ge=50,
        le=5000,
        description="Target chunk size in characters (for recursive/semantic strategies)",
    )
    chunk_overlap: int = Field(
        default=settings.DEFAULT_CHUNK_OVERLAP,
        ge=0,
        le=500,
        description="Overlap between consecutive chunks (for recursive strategy)",
    )


class DocumentOut(BaseModel):
    """Single document chunk returned by the list endpoint."""
    id: str
    document: str
    metadata: dict


# ---------------------------------------------------------------------------
# RAG — Ask
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Body for a single-turn RAG query."""
    question: str = Field(..., min_length=1)
    user: Optional[str] = Field(None, description="Filter by user name")
    n_results: int = Field(default=settings.DEFAULT_N_RESULTS, ge=1, le=20)


class AskResponse(BaseModel):
    question: str
    answer: str
    context_used: list[str]
    filtered_by_user: Optional[str]


# ---------------------------------------------------------------------------
# RAG — Chat
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    """Body for multi-turn conversational RAG."""
    messages: list[ChatMessage] = Field(..., min_length=1)
    user: Optional[str] = None
    n_results: int = Field(default=settings.DEFAULT_N_RESULTS, ge=1, le=20)


class ChatResponse(BaseModel):
    reply: str
    context_used: list[str]
    filtered_by_user: Optional[str]


# ---------------------------------------------------------------------------
# Health & Admin
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    ollama: str
    chromadb: str
    collection_count: int


class AdminStatsResponse(BaseModel):
    total_documents: int
    users: list[str]
    user_document_counts: dict[str, int]
    collection_name: str
    embedding_model: str
    chat_model: str
