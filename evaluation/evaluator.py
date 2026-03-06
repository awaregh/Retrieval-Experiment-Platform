"""
Evaluation orchestrator: runs queries through a retrieval pipeline and computes metrics.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from evaluation.metrics import compute_all_metrics, mean_reciprocal_rank

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    query: str
    retrieved_doc_ids: List[str]
    relevant_doc_ids: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    retrieved_texts: List[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across all queries."""
    query_results: List[QueryResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    num_queries: int = 0
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])

    def to_dict(self) -> Dict:
        return {
            "num_queries": self.num_queries,
            "k_values": self.k_values,
            "aggregate_metrics": self.aggregate_metrics,
            "per_query_results": [
                {
                    "query": qr.query,
                    "retrieved_doc_ids": qr.retrieved_doc_ids,
                    "relevant_doc_ids": qr.relevant_doc_ids,
                    "metrics": qr.metrics,
                }
                for qr in self.query_results
            ],
        }


class Evaluator:
    """
    Evaluates a retrieval pipeline on a query dataset.
    """

    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate(
        self,
        queries: List[Dict[str, Any]],
        retriever,
        reranker=None,
    ) -> EvaluationReport:
        """
        Run evaluation over a list of queries.

        Args:
            queries: list of {"question": str, "relevant_documents": [doc_id, ...]}
            retriever: retriever instance with .retrieve(query) method
            reranker: optional reranker with .rerank(query, results) method

        Returns:
            EvaluationReport
        """
        report = EvaluationReport(k_values=self.k_values)
        all_query_results_for_mrr = []

        for query_data in queries:
            query_text = query_data.get("question", query_data.get("query", ""))
            relevant_ids = set(query_data.get("relevant_documents", []))

            try:
                results = retriever.retrieve(query_text)
            except Exception as e:
                logger.error(f"Retrieval failed for query '{query_text}': {e}")
                continue

            if reranker is not None:
                try:
                    results = reranker.rerank(query_text, results)
                except Exception as e:
                    logger.error(f"Reranking failed for query '{query_text}': {e}")

            retrieved_ids = [r.document_id for r in results]
            retrieved_texts = [r.text for r in results]

            metrics = compute_all_metrics(retrieved_ids, relevant_ids, self.k_values)

            query_result = QueryResult(
                query=query_text,
                retrieved_doc_ids=retrieved_ids,
                relevant_doc_ids=list(relevant_ids),
                metrics=metrics,
                retrieved_texts=retrieved_texts,
            )
            report.query_results.append(query_result)
            all_query_results_for_mrr.append({"retrieved": retrieved_ids, "relevant": relevant_ids})

        # Aggregate metrics
        if report.query_results:
            all_metric_keys = report.query_results[0].metrics.keys()
            for key in all_metric_keys:
                values = [qr.metrics.get(key, 0.0) for qr in report.query_results]
                report.aggregate_metrics[key] = sum(values) / len(values)

            report.aggregate_metrics["mrr"] = mean_reciprocal_rank(all_query_results_for_mrr)

        report.num_queries = len(report.query_results)
        return report
