"""
Ollama LLM interaction — async chat and streaming.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import ollama
from fastapi import HTTPException

from src.rag_api.config import settings


# ---------------------------------------------------------------------------
# Async client (re-uses connections)
# ---------------------------------------------------------------------------
_async_client = ollama.AsyncClient(host=settings.OLLAMA_URL)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_augmented_prompt(context: str, question: str) -> str:
    """Build the system + user prompt for a single-turn RAG query."""
    return (
        "You are a helpful assistant. Use the following context to answer the question. "
        "If the context doesn't contain relevant information, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def build_system_message(context: str) -> dict:
    """Build a system message for multi-turn chat with retrieved context."""
    return {
        "role": "system",
        "content": (
            "You are a helpful assistant. Use the following retrieved context "
            "to inform your answers. If the context is not relevant, say so.\n\n"
            f"Context:\n{context}"
        ),
    }


# ---------------------------------------------------------------------------
# Chat (non-streaming)
# ---------------------------------------------------------------------------

async def chat(messages: list[dict]) -> str:
    """Send messages to Ollama and return the assistant reply.

    Uses the async client so the FastAPI event loop is not blocked.
    """
    try:
        response = await _async_client.chat(
            model=settings.OLLAMA_CHAT_MODEL,
            messages=messages,
        )
        return response["message"]["content"]
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama chat failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Chat (streaming)
# ---------------------------------------------------------------------------

async def stream_chat(messages: list[dict]) -> AsyncIterator[str]:
    """Stream tokens from Ollama as they are generated.

    Yields individual content fragments suitable for Server-Sent Events.
    """
    try:
        stream = await _async_client.chat(
            model=settings.OLLAMA_CHAT_MODEL,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama stream failed: {exc}",
        ) from exc
