"""
Tests for SSE streaming endpoints (/ask/stream, /chat/stream).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from tests.conftest import TEST_USER


def _parse_sse_events(text: str) -> list[dict]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events = []
    current_event = None
    current_data = []

    for line in text.split("\n"):
        if line.startswith("event:"):
            if current_event is not None:
                events.append({"event": current_event, "data": "\n".join(current_data)})
            current_event = line[len("event:"):].strip()
            current_data = []
        elif line.startswith("data:"):
            current_data.append(line[len("data:"):].strip())
        elif line == "" and current_event is not None:
            events.append({"event": current_event, "data": "\n".join(current_data)})
            current_event = None
            current_data = []

    if current_event is not None:
        events.append({"event": current_event, "data": "\n".join(current_data)})

    return events


class TestAskStream:
    @patch("src.rag_api.services.llm._async_client")
    def test_ask_stream_returns_sse(self, mock_client, seeded_docs):
        # Mock streaming: return an async iterator of chunks
        async def _mock_stream(*args, **kwargs):
            chunks = [
                {"message": {"content": "Hello"}},
                {"message": {"content": " world"}},
                {"message": {"content": "!"}},
            ]
            for c in chunks:
                yield c

        mock_client.chat = AsyncMock(return_value=_mock_stream())

        resp = seeded_docs.post("/ask/stream", json={
            "question": "Who is Alice?",
            "user": TEST_USER,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_ask_stream_no_context(self, client):
        resp = client.post("/ask/stream", json={
            "question": "Unknown topic xyz",
            "user": "nonexistent_user_abc_999",
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


class TestChatStream:
    @patch("src.rag_api.services.llm._async_client")
    def test_chat_stream_returns_sse(self, mock_client, seeded_docs):
        async def _mock_stream(*args, **kwargs):
            chunks = [
                {"message": {"content": "Hi"}},
                {"message": {"content": " there"}},
            ]
            for c in chunks:
                yield c

        mock_client.chat = AsyncMock(return_value=_mock_stream())

        resp = seeded_docs.post("/chat/stream", json={
            "messages": [
                {"role": "user", "content": "Tell me about Alice."},
            ],
            "user": TEST_USER,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_chat_stream_requires_user_message(self, client):
        resp = client.post("/chat/stream", json={
            "messages": [
                {"role": "system", "content": "System only."},
            ],
        })
        assert resp.status_code == 400
