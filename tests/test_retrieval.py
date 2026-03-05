import pytest
from core.retrieval import BM25Retriever, RetrievalResult


SAMPLE_DOCS = [
    {"text": "Kubernetes is a container orchestration platform", "chunk_id": "doc_1_c0", "document_id": "doc_1", "metadata": {}},
    {"text": "Docker containers package applications with their dependencies", "chunk_id": "doc_2_c0", "document_id": "doc_2", "metadata": {}},
    {"text": "Redis is an in-memory data structure store used as a cache", "chunk_id": "doc_3_c0", "document_id": "doc_3", "metadata": {}},
    {"text": "PostgreSQL is a relational database management system", "chunk_id": "doc_4_c0", "document_id": "doc_4", "metadata": {}},
    {"text": "Prometheus monitors Kubernetes clusters and microservices", "chunk_id": "doc_5_c0", "document_id": "doc_5", "metadata": {}},
    {"text": "Apache Kafka is a distributed event streaming platform", "chunk_id": "doc_6_c0", "document_id": "doc_6", "metadata": {}},
    {"text": "Elasticsearch provides full-text search and analytics", "chunk_id": "doc_7_c0", "document_id": "doc_7", "metadata": {}},
    {"text": "Grafana visualizes metrics and logs from Prometheus", "chunk_id": "doc_8_c0", "document_id": "doc_8", "metadata": {}},
]


class TestBM25Retriever:
    def test_index_and_retrieve(self):
        retriever = BM25Retriever(top_k=3)
        retriever.index(SAMPLE_DOCS)
        results = retriever.retrieve("Kubernetes container orchestration")
        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)

    def test_top_k_respected(self):
        retriever = BM25Retriever(top_k=3)
        retriever.index(SAMPLE_DOCS)
        results = retriever.retrieve("database")
        assert len(results) <= 3

    def test_retrieval_not_indexed_raises(self):
        retriever = BM25Retriever(top_k=3)
        with pytest.raises(RuntimeError, match="BM25 index not built"):
            retriever.retrieve("query")

    def test_relevant_doc_ranked_higher(self):
        retriever = BM25Retriever(top_k=5)
        retriever.index(SAMPLE_DOCS)
        results = retriever.retrieve("Kubernetes")
        # doc_1 is about Kubernetes and should appear in top results
        top_doc_ids = [r.document_id for r in results]
        assert "doc_1" in top_doc_ids

    def test_cache_doc_ranked_higher(self):
        retriever = BM25Retriever(top_k=5)
        retriever.index(SAMPLE_DOCS)
        results = retriever.retrieve("Redis cache memory")
        top_doc_ids = [r.document_id for r in results]
        assert "doc_3" in top_doc_ids

    def test_results_have_required_fields(self):
        retriever = BM25Retriever(top_k=3)
        retriever.index(SAMPLE_DOCS)
        results = retriever.retrieve("platform")
        for r in results:
            assert hasattr(r, "document_id")
            assert hasattr(r, "chunk_id")
            assert hasattr(r, "text")
            assert hasattr(r, "score")
            assert hasattr(r, "rank")

    def test_results_are_ranked_by_score(self):
        retriever = BM25Retriever(top_k=5)
        retriever.index(SAMPLE_DOCS)
        results = retriever.retrieve("database")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_values_sequential(self):
        retriever = BM25Retriever(top_k=5)
        retriever.index(SAMPLE_DOCS)
        results = retriever.retrieve("platform")
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_empty_query(self):
        retriever = BM25Retriever(top_k=3)
        retriever.index(SAMPLE_DOCS)
        results = retriever.retrieve("")
        # Should not raise; returns results (scores may all be 0)
        assert isinstance(results, list)

    def test_reindex_replaces_corpus(self):
        retriever = BM25Retriever(top_k=3)
        retriever.index(SAMPLE_DOCS)
        new_docs = [
            {"text": "completely new content about neural networks", "chunk_id": "new_1", "document_id": "new_doc_1", "metadata": {}},
        ]
        retriever.index(new_docs)
        results = retriever.retrieve("neural networks")
        assert results[0].document_id == "new_doc_1"

    def test_metadata_preserved(self):
        docs_with_meta = [
            {"text": "test document", "chunk_id": "c1", "document_id": "d1", "metadata": {"source": "test"}},
        ]
        retriever = BM25Retriever(top_k=1)
        retriever.index(docs_with_meta)
        results = retriever.retrieve("test document")
        assert results[0].metadata == {"source": "test"}

    def test_top_k_larger_than_corpus(self):
        retriever = BM25Retriever(top_k=100)
        retriever.index(SAMPLE_DOCS)
        results = retriever.retrieve("platform")
        assert len(results) <= len(SAMPLE_DOCS)
