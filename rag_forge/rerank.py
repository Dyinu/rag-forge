"""Reranking: cross-encoder or nothing.

Cross-encoder reranking is the single biggest accuracy boost you can add to a
RAG pipeline. At Observe.AI it took us from ~72% to ~84% relevance on our
internal eval set. The tradeoff is latency — about 50-100ms per query.
"""

from __future__ import annotations

_reranker_model = None


def cross_encoder_rerank(
    query: str,
    chunks: list[str],
    chunk_indices: list[int],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Rerank retrieved chunks using a cross-encoder model.

    Uses ms-marco-MiniLM-L-6-v2 — small, fast, and good enough. The larger
    models give maybe 1-2% better ranking but 3x the latency.
    """
    global _reranker_model

    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [[query, chunk] for chunk in chunks]
    scores = _reranker_model.predict(pairs)

    scored = list(zip(chunk_indices, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(idx, float(score)) for idx, score in scored[:top_k]]


def no_rerank(
    query: str,
    chunks: list[str],
    chunk_indices: list[int],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Passthrough — just return the chunks as-is with their original order."""
    return [(idx, 1.0 - i * 0.01) for i, idx in enumerate(chunk_indices[:top_k])]


RERANKERS = {
    "cross-encoder": cross_encoder_rerank,
    "none": no_rerank,
}
