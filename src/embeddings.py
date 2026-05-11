"""
src/embeddings.py — Embedding model wrapper.

Uses sentence-transformers/all-MiniLM-L6-v2:
  - 384-dimensional output vectors
  - ~80MB download (cached after first use in ~/.cache/huggingface/)
  - Runs on CPU — no GPU required
  - Trained on 1B+ sentence pairs
  - Within 5% of OpenAI text-embedding-ada-002 on MTEB retrieval benchmarks
  - Completely free forever

How embeddings work:
  The model encodes text into a 384-float vector where semantically similar
  sentences are geometrically close (high cosine similarity).
  "cardiac arrest" and "heart failure" land near each other even though
  they share no words. This enables semantic search beyond keyword matching.

Why normalize embeddings?
  Setting normalize_embeddings=True makes all vectors unit-length (L2 norm = 1).
  This means dot product == cosine similarity, which ChromaDB uses for search.
  Without normalization, longer texts get higher dot products just from length.
"""

import logging
from functools import lru_cache
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the embedding model once and cache it for the process lifetime.
    lru_cache(maxsize=1) ensures we never load it twice — loading takes ~2s
    and uses ~300MB RAM.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,   # cosine similarity via dot product
            "batch_size": 32,               # process 32 chunks at once (speed)
        },
    )
    logger.info("Embedding model loaded.")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings.
    Returns a list of 384-dimensional float vectors.
    Used directly when you need raw vectors (e.g. for BM25 fallback).
    """
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    Uses embed_query (vs embed_documents) — some models have asymmetric
    encoding for queries vs passages; this respects that distinction.
    """
    model = get_embedding_model()
    return model.embed_query(query)
