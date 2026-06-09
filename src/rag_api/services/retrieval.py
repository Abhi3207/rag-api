"""
Context retrieval from ChromaDB.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import HTTPException

from src.rag_api.database import get_collection


async def retrieve_context(
    question: str,
    user: Optional[str],
    n_results: int,
) -> tuple[list[str], list[dict]]:
    """Query ChromaDB for relevant documents.

    Runs the blocking ChromaDB call in a thread so it doesn't block the
    async event loop.

    Returns
    -------
    tuple[list[str], list[dict]]
        (documents, metadatas) matching the query.
    """
    query_params: dict = {
        "query_texts": [question],
        "n_results": n_results,
    }
    if user:
        query_params["where"] = {"user_name": user}

    try:
        results = await asyncio.to_thread(get_collection().query, **query_params)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"ChromaDB query failed: {exc}",
        ) from exc

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    return documents, metadatas
