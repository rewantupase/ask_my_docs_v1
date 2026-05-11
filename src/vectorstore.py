"""
src/vectorstore.py — ChromaDB operations.

ChromaDB is a local, SQLite-backed vector database.
All data lives in ./chroma_db/ — no cloud, no API key, no network calls.

Why ChromaDB over Pinecone/Weaviate?
  - Runs fully local: no account, no billing, no data leaving your machine
  - Persists to disk as SQLite: restart your script, data is still there
  - Supports metadata filtering: filter by source, page, date, etc.
  - Fast enough for millions of vectors on a laptop

Architecture:
  ChromaDB stores:
    - The embedding vector (384 floats)
    - The original text (stored separately from the vector)
    - Arbitrary metadata dict (source, page, chunk_id, preview, etc.)

  On query:
    1. Embed the query (384 floats)
    2. Compute cosine similarity against all stored vectors (HNSW index)
    3. Return top-K documents with their metadata and distance scores
"""

import logging
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.embeddings import get_embedding_model

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_vectorstore = None   # module-level singleton


def get_vectorstore(reset: bool = False):
    """
    Get (or create) the ChromaDB vector store.
    Singleton pattern: only opens the DB connection once per process.
    
    Args:
        reset: if True, wipe the collection and start fresh.
               Use this when re-ingesting all documents from scratch.
    """
    global _vectorstore
    if _vectorstore is not None and not reset:
        return _vectorstore

    from langchain_chroma import Chroma

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embedding_model = get_embedding_model()

    if reset:
        import shutil
        if config.CHROMA_DIR.exists():
            shutil.rmtree(config.CHROMA_DIR)
        logger.info("ChromaDB collection reset.")

    _vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(config.CHROMA_DIR),
    )
    count = _vectorstore._collection.count()
    logger.info(f"ChromaDB ready — {count} chunks indexed.")
    return _vectorstore


def add_chunks(chunks: List[Document], reset: bool = False) -> int:
    """
    Embed and store a list of chunk Documents in ChromaDB.

    Uses chunk_id as the document ID, so re-ingesting the same document
    is idempotent — duplicate vectors are overwritten, not duplicated.

    Returns the number of chunks stored.
    """
    if not chunks:
        logger.warning("No chunks to add.")
        return 0

    store = get_vectorstore(reset=reset)
    
    # ChromaDB requires unique string IDs per document
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    
    logger.info(f"Embedding and storing {len(chunks)} chunks...")
    store.add_documents(documents=chunks, ids=ids)
    
    logger.info(f"Stored {len(chunks)} chunks in ChromaDB.")
    return len(chunks)


def similarity_search(
    query: str, k: int = None
) -> List[Tuple[Document, float]]:
    """
    Vector similarity search.
    Returns list of (Document, distance) tuples, sorted by relevance.
    Distance is cosine distance (lower = more similar) but ChromaDB
    can return it as similarity score depending on version.

    We return raw results — the hybrid retriever combines these with BM25.
    """
    k = k or config.VECTOR_TOP_K
    store = get_vectorstore()
    
    results = store.similarity_search_with_relevance_scores(query, k=k)
    # Returns List[(Document, score)] where score is cosine similarity (0-1)
    # Higher score = more relevant
    return results


def get_all_chunks() -> List[Document]:
    """
    Retrieve all stored chunks. Used to rebuild the BM25 index after restart.
    ChromaDB stores the text alongside the vector, so we can retrieve it
    without re-reading the original documents.
    """
    store = get_vectorstore()
    
    # Use the underlying chromadb client to get all documents
    collection = store._collection
    result = collection.get(include=["documents", "metadatas"])
    
    docs = []
    for text, metadata in zip(result["documents"], result["metadatas"]):
        doc = Document(page_content=text, metadata=metadata or {})
        docs.append(doc)
    
    logger.info(f"Retrieved {len(docs)} chunks from ChromaDB.")
    return docs


def get_chunk_count() -> int:
    """Return the total number of chunks indexed."""
    store = get_vectorstore()
    return store._collection.count()
