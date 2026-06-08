"""
Document CRUD routes — add, list, upload, and delete document chunks.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from src.rag_api.config import settings
from src.rag_api.database import collection
from src.rag_api.middleware.rate_limit import limiter
from src.rag_api.models import ChunkStrategy, DocumentOut, DocumentSubmission
from src.rag_api.services.chunking import chunk_text

router = APIRouter(prefix="/documents", tags=["Documents"])


# ---------------------------------------------------------------------------
# POST /documents — JSON body
# ---------------------------------------------------------------------------

@router.post("")
@limiter.limit(settings.RATE_LIMIT_DOCS)
async def add_document(submission: DocumentSubmission, request: Request):
    """Add profile text as chunks using the specified chunking strategy."""
    chunks = chunk_text(
        submission.content,
        strategy=submission.chunk_strategy.value,
        chunk_size=submission.chunk_size,
        chunk_overlap=submission.chunk_overlap,
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="No non-empty chunks found in content.")

    ids = [f"{submission.user_name}-chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": "profile",
            "user_name": submission.user_name,
            "chunk_index": i,
            "chunk_strategy": submission.chunk_strategy.value,
        }
        for i in range(len(chunks))
    ]

    try:
        await asyncio.to_thread(collection.upsert, ids=ids, documents=chunks, metadatas=metadatas)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ChromaDB upsert failed: {exc}") from exc

    return {
        "message": f"Upserted {len(chunks)} chunks for user '{submission.user_name}'.",
        "user_name": submission.user_name,
        "chunks_added": len(chunks),
        "chunk_strategy": submission.chunk_strategy.value,
    }


# ---------------------------------------------------------------------------
# POST /documents/upload — file upload
# ---------------------------------------------------------------------------

@router.post("/upload")
@limiter.limit(settings.RATE_LIMIT_DOCS)
async def upload_document(
    request: Request,
    user_name: str = Query(..., min_length=1),
    file: UploadFile = File(...),
    chunk_strategy: ChunkStrategy = Query(
        default=ChunkStrategy(settings.DEFAULT_CHUNK_STRATEGY),
        description="Chunking strategy to apply",
    ),
    chunk_size: int = Query(default=settings.DEFAULT_CHUNK_SIZE, ge=50, le=5000),
    chunk_overlap: int = Query(default=settings.DEFAULT_CHUNK_OVERLAP, ge=0, le=500),
):
    """Upload a .txt file and store its paragraphs as chunks."""
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")

    chunks = chunk_text(
        text,
        strategy=chunk_strategy.value,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="File contained no non-empty paragraphs.")

    ids = [f"{user_name}-chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": "upload",
            "user_name": user_name,
            "chunk_index": i,
            "filename": file.filename,
            "chunk_strategy": chunk_strategy.value,
        }
        for i in range(len(chunks))
    ]

    try:
        await asyncio.to_thread(collection.upsert, ids=ids, documents=chunks, metadatas=metadatas)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ChromaDB upsert failed: {exc}") from exc

    return {
        "message": f"Uploaded and stored {len(chunks)} chunks from '{file.filename}' for user '{user_name}'.",
        "user_name": user_name,
        "chunks_added": len(chunks),
        "chunk_strategy": chunk_strategy.value,
    }


# ---------------------------------------------------------------------------
# GET /documents — list
# ---------------------------------------------------------------------------

@router.get("", response_model=list[DocumentOut])
@limiter.limit(settings.RATE_LIMIT_DOCS)
async def list_documents(
    request: Request,
    user: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
):
    """List stored document chunks, optionally filtered by user."""
    try:
        get_params: dict = {"limit": limit}
        if user:
            get_params["where"] = {"user_name": user}
        results = await asyncio.to_thread(collection.get, **get_params)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ChromaDB get failed: {exc}") from exc

    return [
        DocumentOut(id=doc_id, document=doc, metadata=meta)
        for doc_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
    ]


# ---------------------------------------------------------------------------
# DELETE /documents/{user_name}
# ---------------------------------------------------------------------------

@router.delete("/{user_name}")
@limiter.limit(settings.RATE_LIMIT_DOCS)
async def delete_user_documents(user_name: str, request: Request):
    """Delete all chunks belonging to a specific user."""
    try:
        results = await asyncio.to_thread(collection.get, where={"user_name": user_name})
        if not results["ids"]:
            raise HTTPException(
                status_code=404,
                detail=f"No documents found for user '{user_name}'.",
            )
        await asyncio.to_thread(collection.delete, ids=results["ids"])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ChromaDB delete failed: {exc}") from exc

    return {
        "message": f"Deleted {len(results['ids'])} chunks for user '{user_name}'.",
        "deleted_count": len(results["ids"]),
    }
