"""
Tests for document CRUD endpoints.
"""

from __future__ import annotations

from tests.conftest import TEST_USER


class TestAddDocument:
    def test_add_document_default_strategy(self, client):
        resp = client.post("/documents", json={
            "user_name": TEST_USER,
            "content": "Paragraph one.\n\nParagraph two.",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunks_added"] == 2
        assert body["chunk_strategy"] == "paragraph"

    def test_add_document_recursive_strategy(self, client):
        # Create a long enough text to see recursive splitting
        long_text = "This is a test sentence. " * 50
        resp = client.post("/documents", json={
            "user_name": TEST_USER,
            "content": long_text,
            "chunk_strategy": "recursive",
            "chunk_size": 200,
            "chunk_overlap": 20,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunks_added"] >= 1
        assert body["chunk_strategy"] == "recursive"

    def test_add_document_semantic_strategy(self, client):
        text = (
            "Alice is a software engineer. She works at Acme Corp. "
            "Bob is a data scientist. He works at Beta Inc. "
            "Carol is a designer. She works at Gamma LLC."
        )
        resp = client.post("/documents", json={
            "user_name": TEST_USER,
            "content": text,
            "chunk_strategy": "semantic",
            "chunk_size": 100,
        })
        assert resp.status_code == 200
        assert resp.json()["chunk_strategy"] == "semantic"

    def test_add_document_empty_content(self, client):
        resp = client.post("/documents", json={
            "user_name": TEST_USER,
            "content": "   ",
        })
        assert resp.status_code == 400


class TestListDocuments:
    def test_list_documents(self, seeded_docs):
        resp = seeded_docs.get("/documents", params={"user": TEST_USER})
        assert resp.status_code == 200
        docs = resp.json()
        assert len(docs) >= 2
        assert all("id" in d and "document" in d and "metadata" in d for d in docs)

    def test_list_documents_with_limit(self, seeded_docs):
        resp = seeded_docs.get("/documents", params={"user": TEST_USER, "limit": 1})
        assert resp.status_code == 200
        assert len(resp.json()) <= 1


class TestDeleteDocuments:
    def test_delete_user_documents(self, seeded_docs):
        resp = seeded_docs.delete(f"/documents/{TEST_USER}")
        assert resp.status_code == 200
        assert resp.json()["deleted_count"] >= 1

    def test_delete_nonexistent_user(self, client):
        resp = client.delete("/documents/nonexistent_user_xyz_12345")
        assert resp.status_code == 404


class TestUploadDocument:
    def test_upload_txt_file(self, client):
        content = b"Upload paragraph one.\n\nUpload paragraph two."
        resp = client.post(
            "/documents/upload",
            params={"user_name": TEST_USER},
            files={"file": ("test.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunks_added"] == 2
        assert body["chunk_strategy"] == "paragraph"

    def test_upload_with_recursive_strategy(self, client):
        content = b"A long sentence. " * 50
        resp = client.post(
            "/documents/upload",
            params={
                "user_name": TEST_USER,
                "chunk_strategy": "recursive",
                "chunk_size": 200,
            },
            files={"file": ("test.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json()["chunk_strategy"] == "recursive"

    def test_upload_non_txt_rejected(self, client):
        resp = client.post(
            "/documents/upload",
            params={"user_name": TEST_USER},
            files={"file": ("data.csv", b"a,b,c", "text/csv")},
        )
        assert resp.status_code == 400
