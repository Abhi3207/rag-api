"""
RAG API — Retrieval-Augmented Generation powered by Ollama & ChromaDB.

Endpoints
---------
GET  /health                  Health check (Ollama + ChromaDB)
POST /documents               Add document chunks via JSON
POST /documents/upload        Add document chunks via .txt file upload
GET  /documents               List all stored chunks
DELETE /documents/{user_name} Remove all chunks belonging to a user
POST /ask                     Single-turn RAG query
POST /chat                    Multi-turn conversational RAG
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

import chromadb
import ollama
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChromaDB setup (module-level so it's shared across requests)
# ---------------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

embedding_fn = OllamaEmbeddingFunction(
    model_name=settings.OLLAMA_EMBED_MODEL,
    url=settings.OLLAMA_URL,
)

collection = chroma_client.get_or_create_collection(
    name=settings.COLLECTION_NAME,
    embedding_function=embedding_fn,
)

# ---------------------------------------------------------------------------
# Lifespan — validate external services on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks then yield control to the app."""
    try:
        ollama.list()
        logger.info("✅  Connected to Ollama at %s", settings.OLLAMA_URL)
    except Exception as exc:
        logger.warning("⚠️  Ollama not reachable (%s). Endpoints will fail until it's available.", exc)
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="A Retrieval-Augmented Generation API backed by Ollama and ChromaDB.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class DocumentSubmission(BaseModel):
    """JSON body for adding document chunks."""
    user_name: str = Field(..., min_length=1, description="Owner of this profile")
    content: str = Field(..., min_length=1, description="Profile text (paragraphs separated by blank lines)")


class DocumentOut(BaseModel):
    """Single document chunk returned by the list endpoint."""
    id: str
    document: str
    metadata: dict


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


class HealthResponse(BaseModel):
    status: str
    ollama: str
    chromadb: str
    collection_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retrieve_context(question: str, user: Optional[str], n_results: int) -> tuple[list[str], list[dict]]:
    """Query ChromaDB and return (documents, metadatas)."""
    query_params: dict = {
        "query_texts": [question],
        "n_results": n_results,
    }
    if user:
        query_params["where"] = {"user_name": user}

    try:
        results = collection.query(**query_params)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ChromaDB query failed: {exc}") from exc

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    return documents, metadatas


def _build_augmented_prompt(context: str, question: str) -> str:
    """Build the system + user prompt for the LLM."""
    return (
        "You are a helpful assistant. Use the following context to answer the question. "
        "If the context doesn't contain relevant information, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def _chat_with_ollama(messages: list[dict]) -> str:
    """Send messages to Ollama and return the assistant reply."""
    try:
        response = ollama.chat(
            model=settings.OLLAMA_CHAT_MODEL,
            messages=messages,
        )
        return response["message"]["content"]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama chat failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# ---- Health ---------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
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


# ---- Documents CRUD ------------------------------------------------------

@app.post("/documents", tags=["Documents"])
def add_document(submission: DocumentSubmission):
    """Add profile text as chunks (split on blank lines)."""
    chunks = [c.strip() for c in submission.content.split("\n\n") if c.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="No non-empty chunks found in content.")

    ids = [f"{submission.user_name}-chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": "profile", "user_name": submission.user_name, "chunk_index": i}
        for i in range(len(chunks))
    ]

    try:
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ChromaDB upsert failed: {exc}") from exc

    return {
        "message": f"Upserted {len(chunks)} chunks for user '{submission.user_name}'.",
        "user_name": submission.user_name,
        "chunks_added": len(chunks),
    }


@app.post("/documents/upload", tags=["Documents"])
async def upload_document(user_name: str = Query(..., min_length=1), file: UploadFile = File(...)):
    """Upload a .txt file and store its paragraphs as chunks."""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="File contained no non-empty paragraphs.")

    ids = [f"{user_name}-chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": "upload", "user_name": user_name, "chunk_index": i, "filename": file.filename}
        for i in range(len(chunks))
    ]

    try:
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ChromaDB upsert failed: {exc}") from exc

    return {
        "message": f"Uploaded and stored {len(chunks)} chunks from '{file.filename}' for user '{user_name}'.",
        "user_name": user_name,
        "chunks_added": len(chunks),
    }


@app.get("/documents", response_model=list[DocumentOut], tags=["Documents"])
def list_documents(user: Optional[str] = None, limit: int = Query(50, ge=1, le=500)):
    """List stored document chunks, optionally filtered by user."""
    try:
        get_params: dict = {"limit": limit}
        if user:
            get_params["where"] = {"user_name": user}
        results = collection.get(**get_params)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ChromaDB get failed: {exc}") from exc

    return [
        DocumentOut(id=doc_id, document=doc, metadata=meta)
        for doc_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
    ]


@app.delete("/documents/{user_name}", tags=["Documents"])
def delete_user_documents(user_name: str):
    """Delete all chunks belonging to a specific user."""
    try:
        # Find all IDs for this user
        results = collection.get(where={"user_name": user_name})
        if not results["ids"]:
            raise HTTPException(status_code=404, detail=f"No documents found for user '{user_name}'.")
        collection.delete(ids=results["ids"])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ChromaDB delete failed: {exc}") from exc

    return {
        "message": f"Deleted {len(results['ids'])} chunks for user '{user_name}'.",
        "deleted_count": len(results["ids"]),
    }


# ---- RAG Query ------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(request: AskRequest):
    """Single-turn RAG: retrieve context and generate an answer."""
    documents, _ = _retrieve_context(request.question, request.user, request.n_results)

    if not documents:
        return AskResponse(
            question=request.question,
            answer="No relevant context found in the knowledge base.",
            context_used=[],
            filtered_by_user=request.user,
        )

    context = "\n\n".join(documents)
    prompt = _build_augmented_prompt(context, request.question)
    answer = _chat_with_ollama([{"role": "user", "content": prompt}])

    return AskResponse(
        question=request.question,
        answer=answer,
        context_used=documents,
        filtered_by_user=request.user,
    )


# ---- Conversational RAG ---------------------------------------------------

@app.post("/chat", response_model=ChatResponse, tags=["RAG"])
def chat(request: ChatRequest):
    """Multi-turn conversational RAG.

    The last user message is used for retrieval.  The full conversation
    history (including retrieved context) is forwarded to the LLM.
    """
    # Use the last user message for retrieval
    last_user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        None,
    )
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="At least one user message is required.")

    documents, _ = _retrieve_context(last_user_msg, request.user, request.n_results)
    context = "\n\n".join(documents) if documents else "(no relevant context found)"

    # Build the messages list for Ollama
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Use the following retrieved context to inform your answers. "
            "If the context is not relevant, say so.\n\n"
            f"Context:\n{context}"
        ),
    }
    ollama_messages = [system_msg] + [{"role": m.role, "content": m.content} for m in request.messages]
    reply = _chat_with_ollama(ollama_messages)

    return ChatResponse(
        reply=reply,
        context_used=documents,
        filtered_by_user=request.user,
    )
