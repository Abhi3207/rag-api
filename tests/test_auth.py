"""
Tests for API key authentication middleware.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestAuthDisabled:
    """When API_KEY_ENABLED=false (default), all requests pass through."""

    def test_health_accessible(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_documents_accessible_without_key(self, client):
        resp = client.get("/documents")
        assert resp.status_code == 200

    def test_ask_accessible_without_key(self, client):
        resp = client.post("/ask", json={
            "question": "test",
            "user": "nobody",
        })
        # Should not be 401, may be 200 (no context)
        assert resp.status_code != 401


class TestAuthEnabled:
    """When API_KEY_ENABLED=true, requests need X-API-Key header."""

    @pytest.fixture
    def auth_client(self):
        """Create a test client with auth enabled."""
        # We need to re-create the app with auth enabled
        with patch.dict("os.environ", {
            "API_KEY_ENABLED": "true",
            "API_KEY": "test-secret-key",
        }):
            # Re-import to pick up new settings
            with patch("ollama.list", return_value=MagicMock()):
                # Force settings reload
                from src.rag_api.config import Settings
                test_settings = Settings()

                with patch("src.rag_api.middleware.auth.settings", test_settings):
                    from src.rag_api.app import app
                    yield TestClient(app), "test-secret-key"

    def test_health_always_public(self, auth_client):
        client, _ = auth_client
        resp = client.get("/health")
        # Health should be accessible even with auth enabled
        assert resp.status_code == 200

    def test_documents_rejected_without_key(self, auth_client):
        client, _ = auth_client
        resp = client.get("/documents")
        assert resp.status_code == 401

    def test_documents_accepted_with_key(self, auth_client):
        client, key = auth_client
        resp = client.get("/documents", headers={"X-API-Key": key})
        assert resp.status_code == 200

    def test_wrong_key_rejected(self, auth_client):
        client, _ = auth_client
        resp = client.get("/documents", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401
