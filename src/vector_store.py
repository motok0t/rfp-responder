"""
AI RFP Responder — Vector Store module

ChromaDB wrapper for the RAG knowledge base.
Provides: get_client, get_or_create_collection, upsert_chunks, query_similar.
"""

from __future__ import annotations

import chromadb

from . import config as cfg


def get_client() -> chromadb.PersistentClient:
    """Connect to the persisted ChromaDB client using config settings."""
    return chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)


def get_or_create_collection(name: str | None = None) -> chromadb.Collection:
    """Get or create the ChromaDB collection with the given name.

    Uses cfg.COLLECTION_NAME by default.
    """
    client = get_client()
    col_name = name or cfg.COLLECTION_NAME
    return client.get_or_create_collection(name=col_name)


def upsert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    collection: chromadb.Collection | None = None,
) -> None:
    """Write chunks, their embeddings, and metadata into the collection in a single call.

    Extracts ids and metadatas from the chunks list and passes the entire
    embeddings list directly to coll.upsert().

    Each chunk dict is expected to have keys: id (or chunk_id), text, metadata (or similar).
    """
    coll = collection or get_or_create_collection()

    # Extract ids: prefer chunk.get("id") then chunk.get("chunk_id"), fall back to hash
    ids = [chunk.get("id") or chunk.get("chunk_id") or str(hash(chunk.get("text", ""))) for chunk in chunks]

    # Extract metadatas: each chunk's metadata dict; ensure text is included if present
    metadatas = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        if "text" in chunk and "text" not in meta:
            meta = dict(meta)  # shallow copy so we dont mutate the original
            meta["text"] = chunk["text"]
        metadatas.append(meta)

    # Single upsert call with all vectors, ids, and metadatas
    coll.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_similar(
    query_embedding: list[float],
    top_k: int | None = None,
    collection: chromadb.Collection | None = None,
) -> dict:
    """Query the collection for the top-k most similar chunks.

    Returns the ChromaDB query result dict (ids, distances, metadatas, documents).
    """
    coll = collection or get_or_create_collection()
    k = top_k or cfg.TOP_K
    return coll.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )