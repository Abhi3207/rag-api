"""
Centralized configuration for the RAG API.

All settings can be overridden via environment variables or a .env file.
Example:
    OLLAMA_URL=http://my-server:11434 uvicorn src.rag_api.app:app
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Ollama ---
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "qwen2.5:0.5b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    # --- ChromaDB ---
    CHROMA_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "personal_profile"

    # --- Retrieval defaults ---
    DEFAULT_N_RESULTS: int = 3

    # --- Chunking ---
    DEFAULT_CHUNK_STRATEGY: str = "paragraph"  # paragraph | recursive | semantic
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50

    # --- Rate Limiting ---
    RATE_LIMIT_RAG: str = "60/minute"
    RATE_LIMIT_DOCS: str = "200/minute"

    # --- Authentication ---
    API_KEY_ENABLED: bool = False
    API_KEY: str = ""

    # --- API ---
    APP_TITLE: str = "RAG API"
    APP_VERSION: str = "2.0.0"
    CORS_ORIGINS: list[str] = ["*"]

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Singleton instance used throughout the app
settings = Settings()
