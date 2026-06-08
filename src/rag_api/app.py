"""
FastAPI application factory.

Creates and configures the app with:
- Lifespan (startup validation)
- CORS middleware
- API key authentication middleware (optional)
- Rate limiting
- All route modules
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import ollama
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.rag_api.config import settings
from src.rag_api.middleware.auth import APIKeyMiddleware
from src.rag_api.middleware.rate_limit import install_rate_limiter
from src.rag_api.routes import documents, health, rag

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
        logger.warning(
            "⚠️  Ollama not reachable (%s). Endpoints will fail until it's available.",
            exc,
        )
    logger.info("🚀  RAG API v%s ready", settings.APP_VERSION)
    yield


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="A Retrieval-Augmented Generation API backed by Ollama and ChromaDB.",
    lifespan=lifespan,
)

# --- Middleware (order matters: last added = first executed) ----------------

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key auth (no-op when disabled)
app.add_middleware(APIKeyMiddleware)

# Rate limiting
install_rate_limiter(app)

# --- Routes ----------------------------------------------------------------

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(rag.router)
