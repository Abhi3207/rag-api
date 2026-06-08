"""
Build (or update) the ChromaDB knowledge base from a text file.

Usage
-----
    python build_knowledge_base.py                        # defaults: profile.txt, user "default"
    python build_knowledge_base.py -f resume.txt -u alice  # custom file and user
    python build_knowledge_base.py --strategy recursive --chunk-size 300
"""

from __future__ import annotations

import argparse
import sys

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

from src.rag_api.config import settings
from src.rag_api.services.chunking import chunk_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a text file into the RAG knowledge base.")
    parser.add_argument(
        "-f", "--file",
        default="profile.txt",
        help="Path to the text file to ingest (default: profile.txt)",
    )
    parser.add_argument(
        "-u", "--user",
        default="default",
        help="User name to tag the chunks with (default: 'default')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing chunks for this user (default: skip duplicates)",
    )
    parser.add_argument(
        "--strategy",
        choices=["paragraph", "recursive", "semantic"],
        default=settings.DEFAULT_CHUNK_STRATEGY,
        help=f"Chunking strategy (default: {settings.DEFAULT_CHUNK_STRATEGY})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.DEFAULT_CHUNK_SIZE,
        help=f"Target chunk size in characters (default: {settings.DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {settings.DEFAULT_CHUNK_OVERLAP})",
    )
    return parser.parse_args()


def load_file(filepath: str) -> str:
    """Read a text file and return its contents."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌  File not found: {filepath}")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    # --- Load and chunk text ---
    text = load_file(args.file)
    chunks = chunk_text(
        text,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if not chunks:
        print(f"⚠️  No non-empty chunks found in {args.file}")
        sys.exit(1)

    print(f"📄  Loaded {len(chunks)} chunks from {args.file} (strategy: {args.strategy})")

    # --- Connect to ChromaDB ---
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

    ef = OllamaEmbeddingFunction(
        model_name=settings.OLLAMA_EMBED_MODEL,
        url=settings.OLLAMA_URL,
    )

    collection = client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        embedding_function=ef,
    )

    # --- Duplicate detection ---
    ids = [f"{args.user}-chunk{i}" for i in range(len(chunks))]

    if not args.force:
        existing = collection.get(ids=ids)
        already = [eid for eid in existing["ids"] if eid]
        if already:
            print(f"⏭️  Skipping {len(already)} chunks that already exist. Use --force to overwrite.")
            # Filter out existing
            new_pairs = [(cid, chunk) for cid, chunk in zip(ids, chunks) if cid not in already]
            if not new_pairs:
                print("Nothing new to add.")
                return
            ids, chunks = zip(*new_pairs)
            ids, chunks = list(ids), list(chunks)

    metadatas = [
        {
            "source": "profile",
            "user_name": args.user,
            "chunk_index": i,
            "chunk_strategy": args.strategy,
        }
        for i in range(len(chunks))
    ]

    # --- Upsert ---
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)

    print(f"✅  Upserted {len(chunks)} chunks into collection '{settings.COLLECTION_NAME}'.")
    print(f"   Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    main()
