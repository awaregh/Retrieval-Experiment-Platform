"""
Retrieval strategies: vector similarity, BM25, and hybrid.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved document with its score."""
    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    rank: int = 0


class VectorRetriever:
    """
    Dense vector retrieval using cosine similarity via ChromaDB.
    """

    def __init__(self, vector_store, embedder, top_k: int = 10, similarity_threshold: float = 0.0):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str, filters: Optional[Dict] = None) -> List[RetrievalResult]:
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.top_k,
            filters=filters,
        )
        ranked = []
        for rank, r in enumerate(results):
            if r.score >= self.similarity_threshold:
                r.rank = rank + 1
                ranked.append(r)
        return ranked


class BM25Retriever:
    """
    Sparse BM25 retrieval using rank-bm25.
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self._bm25 = None
        self._corpus: List[Dict] = []

    def index(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents for BM25 retrieval. Each doc needs 'text', 'chunk_id', 'document_id'."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 is required. Install with: pip install rank-bm25")

        self._corpus = documents
        tokenized = [doc["text"].lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, filters: Optional[Dict] = None) -> List[RetrievalResult]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call index() first.")

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:self.top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            doc = self._corpus[idx]
            results.append(RetrievalResult(
                document_id=doc.get("document_id", ""),
                chunk_id=doc.get("chunk_id", ""),
                text=doc["text"],
                score=float(scores[idx]),
                metadata=doc.get("metadata", {}),
                rank=rank + 1,
            ))
        return results


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (sparse) + dense vector scores via RRF (Reciprocal Rank Fusion).
    """

    def __init__(self, vector_store, embedder, top_k: int = 10, alpha: float = 0.5, rrf_k: int = 60):
        """
        Args:
            alpha: weight for dense scores (1-alpha for BM25)
            rrf_k: RRF constant
        """
        self.vector_retriever = VectorRetriever(vector_store, embedder, top_k=top_k * 2)
        self.bm25_retriever = BM25Retriever(top_k=top_k * 2)
        self.top_k = top_k
        self.alpha = alpha
        self.rrf_k = rrf_k

    def index_bm25(self, documents: List[Dict]) -> None:
        self.bm25_retriever.index(documents)

    def retrieve(self, query: str, filters: Optional[Dict] = None) -> List[RetrievalResult]:
        dense_results = self.vector_retriever.retrieve(query, filters)
        try:
            bm25_results = self.bm25_retriever.retrieve(query, filters)
        except RuntimeError:
            # BM25 not indexed, fall back to dense only
            logger.warning("BM25 not indexed, falling back to dense-only retrieval.")
            return dense_results[:self.top_k]

        # Build RRF score map
        scores: Dict[str, float] = {}
        doc_map: Dict[str, RetrievalResult] = {}

        for r in dense_results:
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + self.alpha / (self.rrf_k + r.rank)
            doc_map[r.chunk_id] = r

        for r in bm25_results:
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + (1 - self.alpha) / (self.rrf_k + r.rank)
            if r.chunk_id not in doc_map:
                doc_map[r.chunk_id] = r

        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:self.top_k]
        results = []
        for rank, cid in enumerate(sorted_ids):
            r = doc_map[cid]
            r.score = scores[cid]
            r.rank = rank + 1
            results.append(r)

        return results


def get_retriever(strategy: str, vector_store=None, embedder=None, top_k: int = 10, **kwargs):
    """Factory to get retriever by strategy."""
    if strategy == "vector":
        return VectorRetriever(vector_store, embedder, top_k=top_k, **kwargs)
    elif strategy == "bm25":
        return BM25Retriever(top_k=top_k)
    elif strategy == "hybrid":
        return HybridRetriever(vector_store, embedder, top_k=top_k, **kwargs)
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}. Choose from: vector, bm25, hybrid")
