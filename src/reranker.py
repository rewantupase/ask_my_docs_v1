"""
src/reranker.py — Cross-encoder reranking.

This is the precision layer that runs AFTER hybrid retrieval.

Two types of retrieval encoders:

  Bi-encoder (what all-MiniLM-L6-v2 is):
    - Encodes query → q_vector
    - Encodes document → d_vector
    - Score = dot_product(q_vector, d_vector)
    - Fast: pre-compute d_vectors at index time
    - Limitation: query and document never "see" each other during encoding.
      The model can't capture interactions like:
        "The query asks for X, but this chunk says X is WRONG"

  Cross-encoder (what ms-marco-MiniLM-L-6-v2 is):
    - Takes [CLS] QUERY [SEP] DOCUMENT [SEP] as input
    - Outputs a single relevance score in one forward pass
    - The query and document INTERACT through self-attention at every layer
    - Captures nuanced relationships: negation, contrast, specificity
    - Limitation: can't pre-compute — must run at query time
    - Solution: run it only on the TOP-K candidates from bi-encoder

Two-stage retrieval strategy:
  Stage 1 — Recall: bi-encoder + BM25 retrieve top 20
    → Cheap, fast, casts wide net
    → Goal: ensure the relevant chunk is IN the candidate set (recall)

  Stage 2 — Precision: cross-encoder reranks top 20 → top 5
    → Expensive but only for 20 pairs
    → Goal: put the BEST chunk at rank 1 (precision)

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO: 500K query-passage pairs from Bing search logs
  - 6-layer MiniLM (tiny but very accurate)
  - ~70MB download, runs on CPU
  - Returns a raw logit (unbounded float); higher = more relevant
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder model once and cache it."""
    from sentence_transformers import CrossEncoder

    logger.info(f"Loading reranker: {config.RERANKER_MODEL}")
    model = CrossEncoder(
        config.RERANKER_MODEL,
        max_length=512,           # truncate very long chunks (rare)
        device=config.EMBEDDING_DEVICE,
    )
    logger.info("Reranker loaded.")
    return model


def rerank(
    query: str,
    candidates: List[Document],
    top_k: int = None,
) -> List[Tuple[Document, float]]:
    """
    Rerank candidate chunks using the cross-encoder.

    Args:
        query:      The user's question
        candidates: Documents from hybrid retrieval (typically 20)
        top_k:      How many to return after reranking (typically 5)

    Returns:
        List of (Document, relevance_score) tuples, sorted best-first.
        relevance_score is the raw cross-encoder logit (higher = more relevant).
    """
    top_k = top_k or config.RERANK_TOP_K

    if not candidates:
        return []

    reranker = get_reranker()

    # Each pair is [query, chunk_text] — the cross-encoder scores each pair
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)   # shape: (len(candidates),)

    # Zip documents with scores and sort by score descending
    ranked = sorted(
        zip(candidates, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    top = ranked[:top_k]
    logger.info(
        f"Reranked {len(candidates)} → top {len(top)} "
        f"(top score: {top[0][1]:.3f}, bottom score: {top[-1][1]:.3f})"
    )
    return top
