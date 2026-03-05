"""
Ranking evaluation metrics: Precision@k, Recall@k, MRR, nDCG.
"""
from __future__ import annotations
import math
from typing import List, Set, Dict, Optional
import numpy as np


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@k = |retrieved[:k] ∩ relevant| / k
    """
    if k <= 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|
    """
    if not relevant:
        return 0.0
    retrieved_at_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Reciprocal Rank = 1 / rank_of_first_relevant_document
    Returns 0 if no relevant document is found.
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(queries_results: List[Dict]) -> float:
    """
    MRR = mean of reciprocal ranks over all queries.
    queries_results: list of {"retrieved": [...], "relevant": {...}}
    """
    if not queries_results:
        return 0.0
    rr_sum = 0.0
    for qr in queries_results:
        rr_sum += reciprocal_rank(qr["retrieved"], set(qr["relevant"]))
    return rr_sum / len(queries_results)


def dcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Discounted Cumulative Gain @ k
    DCG = sum_i (rel_i / log2(i+2))
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / math.log2(i + 2)
    return dcg


def ideal_dcg_at_k(relevant: Set[str], k: int) -> float:
    """
    Ideal DCG: all relevant docs placed at the top.
    """
    num_relevant = min(len(relevant), k)
    idcg = 0.0
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)
    return idcg


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Normalized DCG @ k = DCG@k / IDCG@k
    """
    idcg = ideal_dcg_at_k(relevant, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / idcg


def f1_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """F1@k = harmonic mean of Precision@k and Recall@k."""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_all_metrics(retrieved: List[str], relevant: Set[str], k_values: List[int] = None) -> Dict:
    """
    Compute all metrics for a single query.
    Returns a dict with metric names as keys.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics = {}
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
        metrics[f"f1@{k}"] = f1_at_k(retrieved, relevant, k)

    metrics["mrr"] = reciprocal_rank(retrieved, relevant)
    return metrics
