"""
Unit tests for chunking strategies.
"""

from __future__ import annotations

import pytest

from src.rag_api.services.chunking import (
    chunk_by_paragraph,
    chunk_recursive,
    chunk_semantic,
    chunk_text,
)


# ---------------------------------------------------------------------------
# Paragraph chunking
# ---------------------------------------------------------------------------

class TestParagraphChunking:
    def test_basic_split(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_by_paragraph(text)
        assert chunks == ["Paragraph one.", "Paragraph two.", "Paragraph three."]

    def test_empty_text(self):
        assert chunk_by_paragraph("") == []
        assert chunk_by_paragraph("   ") == []

    def test_no_blank_lines(self):
        text = "Single block of text with no blank lines."
        chunks = chunk_by_paragraph(text)
        assert chunks == ["Single block of text with no blank lines."]

    def test_multiple_blank_lines(self):
        text = "One.\n\n\n\nTwo."
        chunks = chunk_by_paragraph(text)
        assert len(chunks) == 2

    def test_strips_whitespace(self):
        text = "  Padded.  \n\n  Also padded.  "
        chunks = chunk_by_paragraph(text)
        assert all(c == c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# Recursive chunking
# ---------------------------------------------------------------------------

class TestRecursiveChunking:
    def test_short_text_returns_single_chunk(self):
        text = "Short text."
        chunks = chunk_recursive(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_splits_long_text(self):
        text = "Word " * 200  # ~1000 chars
        chunks = chunk_recursive(text, chunk_size=200, chunk_overlap=0)
        assert len(chunks) > 1
        assert all(len(c) <= 250 for c in chunks)  # some tolerance for merging

    def test_overlap(self):
        text = "A. " * 100  # ~300 chars
        chunks = chunk_recursive(text, chunk_size=100, chunk_overlap=20)
        # With overlap, chunks should share content at boundaries
        assert len(chunks) >= 2

    def test_empty_text(self):
        assert chunk_recursive("") == []
        assert chunk_recursive("   ") == []

    def test_respects_paragraph_boundaries(self):
        text = "Paragraph one content.\n\nParagraph two content."
        chunks = chunk_recursive(text, chunk_size=30, chunk_overlap=0)
        # Should split on \n\n first since each paragraph > 30 chars
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

class TestSemanticChunking:
    def test_basic_sentence_grouping(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_semantic(text, chunk_size=60)
        assert len(chunks) >= 2
        assert all(len(c) > 0 for c in chunks)

    def test_short_text_single_chunk(self):
        text = "Just one sentence."
        chunks = chunk_semantic(text, chunk_size=500)
        assert len(chunks) == 1

    def test_empty_text(self):
        assert chunk_semantic("") == []
        assert chunk_semantic("   ") == []


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TestChunkTextDispatcher:
    def test_paragraph_strategy(self):
        chunks = chunk_text("A.\n\nB.", strategy="paragraph")
        assert chunks == ["A.", "B."]

    def test_recursive_strategy(self):
        chunks = chunk_text("Hello world.", strategy="recursive")
        assert len(chunks) >= 1

    def test_semantic_strategy(self):
        chunks = chunk_text("First. Second.", strategy="semantic")
        assert len(chunks) >= 1

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_text("text", strategy="unknown")
