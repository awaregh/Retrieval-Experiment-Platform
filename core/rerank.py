"""
Reranking module using cross-encoder models.
"""
from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np

from core.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a cross-encoder model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: Optional[int] = None):
        self.model_name = model_name
        self.top_n = top_n
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers is required for cross-encoder reranking.")
        return self._model

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        if not results:
            return results

        model = self._load_model()
        pairs = [(query, r.text) for r in results]
        scores = model.predict(pairs)

        for result, score in zip(results, scores):
            result.score = float(score)

        reranked = sorted(results, key=lambda r: r.score, reverse=True)
        top_n = self.top_n or len(reranked)
        reranked = reranked[:top_n]

        for rank, r in enumerate(reranked):
            r.rank = rank + 1

        return reranked


class ScoreNormalizationReranker:
    """
    Normalizes scores to [0, 1] using min-max normalization.
    """

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        if not results:
            return results

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        for r in results:
            r.score = (r.score - min_score) / score_range if score_range > 0 else 1.0

        return sorted(results, key=lambda r: r.score, reverse=True)


def get_reranker(strategy: str, **kwargs):
    """Factory to get a reranker."""
    if strategy == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    elif strategy == "score_normalization":
        return ScoreNormalizationReranker()
    elif strategy == "none" or strategy is None:
        return None
    else:
        raise ValueError(f"Unknown reranker strategy: {strategy}")
