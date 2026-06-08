"""
Shared test fixtures for the RAG API test suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Patch ollama.list before importing the app to prevent the lifespan
# startup check from making a real network call.
with patch("ollama.list", return_value=MagicMock()):
    from src.rag_api.app import app
    from src.rag_api.database import collection

from fastapi.testclient import TestClient


TEST_USER = "test_user_pytest"


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup():
    """Remove test documents after each test."""
    yield
    try:
        results = collection.get(where={"user_name": TEST_USER})
        if results["ids"]:
            collection.delete(ids=results["ids"])
    except Exception:
        pass


@pytest.fixture
def seeded_docs(client):
    """Seed the database with test documents and return the client."""
    client.post("/documents", json={
        "user_name": TEST_USER,
        "content": "Alice is a software engineer at Acme Corp.\n\nShe specializes in distributed systems and Python.",
    })
    return client
