"""
Tests for /ask and /chat RAG endpoints (non-streaming).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from tests.conftest import TEST_USER


class TestAsk:
    @patch("src.rag_api.services.llm._async_client")
    def test_ask_returns_answer(self, mock_client, seeded_docs):
        mock_client.chat = AsyncMock(
            return_value={"message": {"content": "Mocked answer."}}
        )
        resp = seeded_docs.post("/ask", json={
            "question": "Who is Alice?",
            "user": TEST_USER,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Mocked answer."
        assert body["filtered_by_user"] == TEST_USER
        assert len(body["context_used"]) > 0

    def test_ask_no_context(self, client):
        """When there's no matching context, the API should still respond gracefully."""
        resp = client.post("/ask", json={
            "question": "Something completely unrelated xyz 999",
            "user": "nonexistent_user_abc_999",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["context_used"] == []
        assert "No relevant context" in body["answer"]

    def test_ask_missing_question(self, client):
        resp = client.post("/ask", json={
            "user": TEST_USER,
        })
        assert resp.status_code == 422  # Pydantic validation error


class TestChat:
    @patch("src.rag_api.services.llm._async_client")
    def test_chat_returns_reply(self, mock_client, seeded_docs):
        mock_client.chat = AsyncMock(
            return_value={"message": {"content": "Chat reply."}}
        )
        resp = seeded_docs.post("/chat", json={
            "messages": [
                {"role": "user", "content": "Tell me about Alice."},
            ],
            "user": TEST_USER,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["reply"] == "Chat reply."
        assert len(body["context_used"]) > 0

    @patch("src.rag_api.services.llm._async_client")
    def test_chat_multi_turn(self, mock_client, seeded_docs):
        mock_client.chat = AsyncMock(
            return_value={"message": {"content": "She knows Python."}}
        )
        resp = seeded_docs.post("/chat", json={
            "messages": [
                {"role": "user", "content": "Tell me about Alice."},
                {"role": "assistant", "content": "Alice is a software engineer."},
                {"role": "user", "content": "What does she specialize in?"},
            ],
            "user": TEST_USER,
        })
        assert resp.status_code == 200
        assert resp.json()["reply"] == "She knows Python."

    def test_chat_requires_user_message(self, client):
        resp = client.post("/chat", json={
            "messages": [
                {"role": "system", "content": "System prompt only."},
            ],
        })
        assert resp.status_code == 400
