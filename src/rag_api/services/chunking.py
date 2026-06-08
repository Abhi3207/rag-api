"""
Pluggable document chunking strategies.

Strategies
----------
- paragraph : Split on blank lines (``\\n\\n``). Original v1 behavior.
- recursive : Recursive character splitting with configurable size & overlap.
- semantic  : Group consecutive sentences by embedding similarity so that
              semantically related sentences stay in the same chunk.
"""

from __future__ import annotations

import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Paragraph splitting (v1 default)
# ---------------------------------------------------------------------------

def chunk_by_paragraph(text: str) -> list[str]:
    """Split *text* on blank lines and discard empty chunks."""
    return [c.strip() for c in text.split("\n\n") if c.strip()]


# ---------------------------------------------------------------------------
# Recursive character splitting
# ---------------------------------------------------------------------------

_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, separator: str) -> list[str]:
    """Split *text* by *separator*, keeping non-empty pieces."""
    if separator == "":
        return list(text)
    return [s for s in text.split(separator) if s]


def chunk_recursive(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Sequence[str] | None = None,
) -> list[str]:
    """Recursively split *text* trying the coarsest separator first.

    When a piece is still larger than *chunk_size* it is re-split with
    the next finer separator.  Finally, consecutive pieces are merged
    with *chunk_overlap* characters of overlap between them.
    """
    if separators is None:
        separators = _SEPARATORS

    if not text.strip():
        return []

    final_chunks: list[str] = []
    _recursive_split(text, chunk_size, list(separators), final_chunks)
    return _merge_with_overlap(final_chunks, chunk_size, chunk_overlap)


def _recursive_split(
    text: str,
    chunk_size: int,
    separators: list[str],
    output: list[str],
) -> None:
    """Build a list of small-enough pieces in *output*."""
    if len(text) <= chunk_size:
        output.append(text.strip())
        return

    if not separators:
        # Reached character-level — hard-cut
        output.append(text[:chunk_size].strip())
        if len(text) > chunk_size:
            _recursive_split(text[chunk_size:], chunk_size, [], output)
        return

    sep = separators[0]
    pieces = _split_text(text, sep)

    for piece in pieces:
        if len(piece) <= chunk_size:
            output.append(piece.strip())
        else:
            _recursive_split(piece, chunk_size, separators[1:], output)


def _merge_with_overlap(
    chunks: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Merge small consecutive chunks and add overlap between them."""
    if not chunks:
        return []

    merged: list[str] = []
    current = chunks[0]

    for piece in chunks[1:]:
        candidate = f"{current} {piece}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            merged.append(current.strip())
            # Overlap: carry the tail of the previous chunk
            if chunk_overlap > 0:
                tail = current[-chunk_overlap:]
                current = f"{tail} {piece}"
            else:
                current = piece

    if current.strip():
        merged.append(current.strip())

    return merged


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_semantic(
    text: str,
    chunk_size: int = 500,
    similarity_threshold: float = 0.5,
) -> list[str]:
    """Group consecutive sentences into chunks by semantic similarity.

    This is a simplified version that groups sentences until the chunk
    reaches *chunk_size* characters, then starts a new chunk.  A true
    embedding-based approach would compare embeddings of consecutive
    sentences — that is left as a future enhancement to avoid adding
    an embedding call inside the chunking pipeline itself.

    For now this provides sentence-aware chunking which is already a
    significant improvement over naive paragraph splitting.
    """
    sentences = _SENTENCE_RE.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        new_len = current_len + len(sentence) + (1 if current else 0)
        if new_len > chunk_size and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len = new_len

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    strategy: str = "paragraph",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Chunk *text* using the named *strategy*.

    Parameters
    ----------
    text : str
        The source text to split.
    strategy : str
        One of ``"paragraph"``, ``"recursive"``, or ``"semantic"``.
    chunk_size : int
        Target chunk size in characters (ignored for ``"paragraph"``).
    chunk_overlap : int
        Overlap between chunks (only used by ``"recursive"``).

    Returns
    -------
    list[str]
        Non-empty chunks.
    """
    match strategy:
        case "paragraph":
            return chunk_by_paragraph(text)
        case "recursive":
            return chunk_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        case "semantic":
            return chunk_semantic(text, chunk_size=chunk_size)
        case _:
            raise ValueError(f"Unknown chunking strategy: {strategy!r}")
