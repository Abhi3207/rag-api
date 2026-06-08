"""
Tests for the health and admin stats endpoints.
"""

from __future__ import annotations


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")
        assert "collection_count" in body
        assert "ollama" in body
        assert "chromadb" in body

    def test_health_has_correct_schema(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert isinstance(body["collection_count"], int)
        assert body["collection_count"] >= 0


class TestAdminStats:
    def test_admin_stats_returns_200(self, client):
        resp = client.get("/admin/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "total_documents" in body
        assert "users" in body
        assert "user_document_counts" in body
        assert "collection_name" in body
        assert "embedding_model" in body
        assert "chat_model" in body

    def test_admin_stats_reflects_documents(self, seeded_docs):
        resp = seeded_docs.get("/admin/stats")
        body = resp.json()
        assert body["total_documents"] >= 2
        assert "test_user_pytest" in body["users"]
        assert body["user_document_counts"]["test_user_pytest"] >= 2
