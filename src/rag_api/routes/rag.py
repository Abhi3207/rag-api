"""
RAG query routes — single-turn ask, multi-turn chat, and SSE streaming variants.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from src.rag_api.config import settings
from src.rag_api.middleware.rate_limit import limiter
from src.rag_api.models import AskRequest, AskResponse, ChatRequest, ChatResponse
from src.rag_api.services import llm, retrieval

router = APIRouter(tags=["RAG"])


# ---------------------------------------------------------------------------
# POST /ask — single-turn RAG
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=AskResponse)
@limiter.limit(settings.RATE_LIMIT_RAG)
async def ask(body: AskRequest, request: Request):
    """Single-turn RAG: retrieve context and generate an answer."""
    documents, _ = await retrieval.retrieve_context(
        body.question, body.user, body.n_results,
    )

    if not documents:
        return AskResponse(
            question=body.question,
            answer="No relevant context found in the knowledge base.",
            context_used=[],
            filtered_by_user=body.user,
        )

    context = "\n\n".join(documents)
    prompt = llm.build_augmented_prompt(context, body.question)
    answer = await llm.chat([{"role": "user", "content": prompt}])

    return AskResponse(
        question=body.question,
        answer=answer,
        context_used=documents,
        filtered_by_user=body.user,
    )


# ---------------------------------------------------------------------------
# POST /ask/stream — single-turn RAG with SSE streaming
# ---------------------------------------------------------------------------

@router.post("/ask/stream")
@limiter.limit(settings.RATE_LIMIT_RAG)
async def ask_stream(body: AskRequest, request: Request):
    """Single-turn RAG with Server-Sent Events streaming.

    Events
    ------
    - ``event: context`` — JSON array of the retrieved context documents.
    - ``event: token``   — Individual token from the LLM.
    - ``event: done``    — Signals completion.
    """
    documents, _ = await retrieval.retrieve_context(
        body.question, body.user, body.n_results,
    )

    if not documents:
        async def _empty():
            yield {
                "event": "context",
                "data": json.dumps([]),
            }
            yield {
                "event": "token",
                "data": "No relevant context found in the knowledge base.",
            }
            yield {"event": "done", "data": ""}

        return EventSourceResponse(_empty())

    context = "\n\n".join(documents)
    prompt = llm.build_augmented_prompt(context, body.question)

    async def _stream():
        # First emit the context
        yield {
            "event": "context",
            "data": json.dumps(documents),
        }
        # Then stream tokens
        async for token in llm.stream_chat([{"role": "user", "content": prompt}]):
            yield {"event": "token", "data": token}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(_stream())


# ---------------------------------------------------------------------------
# POST /chat — multi-turn conversational RAG
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_RAG)
async def chat(body: ChatRequest, request: Request):
    """Multi-turn conversational RAG.

    The last user message is used for retrieval.  The full conversation
    history (including retrieved context) is forwarded to the LLM.
    """
    last_user_msg = next(
        (m.content for m in reversed(body.messages) if m.role == "user"),
        None,
    )
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="At least one user message is required.")

    documents, _ = await retrieval.retrieve_context(
        last_user_msg, body.user, body.n_results,
    )
    context = "\n\n".join(documents) if documents else "(no relevant context found)"

    system_msg = llm.build_system_message(context)
    ollama_messages = [system_msg] + [
        {"role": m.role, "content": m.content} for m in body.messages
    ]
    reply = await llm.chat(ollama_messages)

    return ChatResponse(
        reply=reply,
        context_used=documents,
        filtered_by_user=body.user,
    )


# ---------------------------------------------------------------------------
# POST /chat/stream — multi-turn conversational RAG with SSE streaming
# ---------------------------------------------------------------------------

@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_RAG)
async def chat_stream(body: ChatRequest, request: Request):
    """Multi-turn conversational RAG with Server-Sent Events streaming.

    Events
    ------
    - ``event: context`` — JSON array of the retrieved context documents.
    - ``event: token``   — Individual token from the LLM.
    - ``event: done``    — Signals completion.
    """
    last_user_msg = next(
        (m.content for m in reversed(body.messages) if m.role == "user"),
        None,
    )
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="At least one user message is required.")

    documents, _ = await retrieval.retrieve_context(
        last_user_msg, body.user, body.n_results,
    )
    context = "\n\n".join(documents) if documents else "(no relevant context found)"

    system_msg = llm.build_system_message(context)
    ollama_messages = [system_msg] + [
        {"role": m.role, "content": m.content} for m in body.messages
    ]

    async def _stream():
        yield {
            "event": "context",
            "data": json.dumps(documents),
        }
        async for token in llm.stream_chat(ollama_messages):
            yield {"event": "token", "data": token}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(_stream())
