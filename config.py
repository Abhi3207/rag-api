"""
Centralized configuration for the RAG API.

All settings can be overridden via environment variables or a .env file.
Example:
    OLLAMA_URL=http://my-server:11434 uvicorn main:app
"""

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

    # --- API ---
    APP_TITLE: str = "RAG API"
    APP_VERSION: str = "1.0.0"
    CORS_ORIGINS: list[str] = ["*"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Singleton instance used throughout the app
settings = Settings()
