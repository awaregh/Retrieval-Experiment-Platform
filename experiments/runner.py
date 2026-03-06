"""
Experiment runner: orchestrates the full retrieval experiment pipeline.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

from config import settings
from core.chunking import get_chunker
from core.embeddings import get_embedder
from core.retrieval import get_retriever, VectorRetriever, HybridRetriever
from core.rerank import get_reranker
from datasets.manager import DatasetManager
from evaluation.evaluator import Evaluator, EvaluationReport
from experiments.tracker import ExperimentTracker
from vector_store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Orchestrates the full experiment: ingest -> embed -> retrieve -> evaluate -> track.
    """

    def __init__(
        self,
        tracker: Optional[ExperimentTracker] = None,
        dataset_manager: Optional[DatasetManager] = None,
    ):
        self.tracker = tracker or ExperimentTracker(settings.database_url)
        self.dataset_manager = dataset_manager or DatasetManager()

    def run(
        self,
        experiment_name: str,
        dataset_id: str,
        chunking_strategy: str = "fixed",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieval_strategy: str = "vector",
        top_k: int = 10,
        reranker_strategy: Optional[str] = None,
        similarity_threshold: float = 0.0,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run a full experiment and return results.

        Args:
            experiment_name: Human-readable name for this experiment
            dataset_id: ID of a previously loaded dataset

        Returns:
            dict with experiment_id and evaluation report
        """
        exp_id = self.tracker.create_experiment(
            name=experiment_name,
            dataset_id=dataset_id,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            retrieval_strategy=retrieval_strategy,
            top_k=top_k,
            reranker=reranker_strategy,
            config={
                "similarity_threshold": similarity_threshold,
                "k_values": k_values or [1, 3, 5, 10],
            },
        )

        try:
            # 1. Chunk the dataset
            logger.info(
                f"[{exp_id}] Chunking with strategy='{chunking_strategy}', "
                f"size={chunk_size}, overlap={chunk_overlap}"
            )
            chunks = self.dataset_manager.chunk_dataset(
                dataset_id,
                strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # 2. Build embeddings
            logger.info(f"[{exp_id}] Generating embeddings with model='{embedding_model}'")
            embedder = get_embedder(embedding_model)
            texts = [c.text for c in chunks]
            embeddings = embedder.embed(texts, batch_size=settings.embedding_batch_size)

            # 3. Index in vector store
            collection_name = f"exp_{exp_id[:8]}"
            vector_store = ChromaVectorStore(collection_name=collection_name)
            metadatas = [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in chunks
            ]
            ids = [c.chunk_id for c in chunks]
            vector_store.add(embeddings, texts, metadatas, ids=ids)

            # 4. Build retriever
            retriever = get_retriever(
                retrieval_strategy,
                vector_store=vector_store,
                embedder=embedder,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

            # If hybrid, also index BM25
            if retrieval_strategy == "hybrid":
                bm25_docs = [
                    {
                        "text": c.text,
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ]
                retriever.index_bm25(bm25_docs)

            # 5. Set up reranker
            reranker = get_reranker(reranker_strategy or "none") if reranker_strategy else None

            # 6. Evaluate
            queries = self.dataset_manager.get_queries(dataset_id)
            evaluator = Evaluator(k_values=k_values or [1, 3, 5, 10])
            logger.info(f"[{exp_id}] Evaluating on {len(queries)} queries")
            report = evaluator.evaluate(queries, retriever, reranker)

            # 7. Save metrics
            self.tracker.update_metrics(exp_id, report.aggregate_metrics)
            logger.info(f"[{exp_id}] Experiment complete. Metrics: {report.aggregate_metrics}")

            return {
                "experiment_id": exp_id,
                "report": report.to_dict(),
                "aggregate_metrics": report.aggregate_metrics,
            }

        except Exception as e:
            logger.error(f"[{exp_id}] Experiment failed: {e}", exc_info=True)
            self.tracker.fail_experiment(exp_id, str(e))
            raise
