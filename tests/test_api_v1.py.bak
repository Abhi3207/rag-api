"""
Tests for the RAG API.

These tests use FastAPI's TestClient and mock Ollama calls so they run
without a live Ollama instance.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Patch ollama.list before importing `main` to prevent the lifespan
# startup check from making a real network call.
with patch("ollama.list", return_value=MagicMock()):
    from main import app, collection

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_USER = "test_user_pytest"


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


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")
        assert "collection_count" in body


# ---------------------------------------------------------------------------
# Documents CRUD
# ---------------------------------------------------------------------------

class TestDocuments:
    def test_add_document(self):
        resp = client.post("/documents", json={
            "user_name": TEST_USER,
            "content": "Paragraph one.\n\nParagraph two.",
        })
        assert resp.status_code == 200
        assert resp.json()["chunks_added"] == 2

    def test_add_document_empty_content(self):
        resp = client.post("/documents", json={
            "user_name": TEST_USER,
            "content": "   ",
        })
        assert resp.status_code == 400

    def test_list_documents(self):
        # Seed data
        client.post("/documents", json={
            "user_name": TEST_USER,
            "content": "Chunk A.\n\nChunk B.",
        })
        resp = client.get("/documents", params={"user": TEST_USER})
        assert resp.status_code == 200
        docs = resp.json()
        assert len(docs) >= 2

    def test_delete_user_documents(self):
        client.post("/documents", json={
            "user_name": TEST_USER,
            "content": "To be deleted.",
        })
        resp = client.delete(f"/documents/{TEST_USER}")
        assert resp.status_code == 200
        assert resp.json()["deleted_count"] >= 1

    def test_delete_nonexistent_user(self):
        resp = client.delete("/documents/nonexistent_user_xyz_12345")
        assert resp.status_code == 404

    def test_upload_txt_file(self):
        content = b"Upload paragraph one.\n\nUpload paragraph two."
        resp = client.post(
            "/documents/upload",
            params={"user_name": TEST_USER},
            files={"file": ("test.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json()["chunks_added"] == 2

    def test_upload_non_txt_rejected(self):
        resp = client.post(
            "/documents/upload",
            params={"user_name": TEST_USER},
            files={"file": ("data.csv", b"a,b,c", "text/csv")},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# RAG — /ask
# ---------------------------------------------------------------------------

class TestAsk:
    @patch("main.ollama.chat")
    def test_ask_returns_answer(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "Mocked answer."}}

        # Seed
        client.post("/documents", json={
            "user_name": TEST_USER,
            "content": "Alice is a software engineer.\n\nShe works at Acme Corp.",
        })

        resp = client.post("/ask", json={
            "question": "Who is Alice?",
            "user": TEST_USER,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Mocked answer."
        assert body["filtered_by_user"] == TEST_USER

    def test_ask_no_context(self):
        """When there's no matching context, the API should still respond gracefully."""
        resp = client.post("/ask", json={
            "question": "Something completely unrelated xyz 999",
            "user": "nonexistent_user_abc_999",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["context_used"] == []


# ---------------------------------------------------------------------------
# RAG — /chat
# ---------------------------------------------------------------------------

class TestChat:
    @patch("main.ollama.chat")
    def test_chat_returns_reply(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "Chat reply."}}

        client.post("/documents", json={
            "user_name": TEST_USER,
            "content": "Bob likes hiking and photography.",
        })

        resp = client.post("/chat", json={
            "messages": [
                {"role": "user", "content": "Tell me about Bob."},
            ],
            "user": TEST_USER,
        })
        assert resp.status_code == 200
        assert resp.json()["reply"] == "Chat reply."

    def test_chat_requires_user_message(self):
        resp = client.post("/chat", json={
            "messages": [
                {"role": "system", "content": "System prompt only."},
            ],
        })
        assert resp.status_code == 400
