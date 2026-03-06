import pytest
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    dcg_at_k,
    ideal_dcg_at_k,
    f1_at_k,
    compute_all_metrics,
)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1", "doc_2", "doc_3"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_zero_precision(self):
        retrieved = ["doc_4", "doc_5", "doc_6"]
        relevant = {"doc_1", "doc_2", "doc_3"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_precision(self):
        retrieved = ["doc_1", "doc_4", "doc_2"]
        relevant = {"doc_1", "doc_2"}
        assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)

    def test_k_zero_returns_zero(self):
        retrieved = ["doc_1"]
        relevant = {"doc_1"}
        assert precision_at_k(retrieved, relevant, 0) == 0.0

    def test_k_larger_than_retrieved(self):
        retrieved = ["doc_1", "doc_2"]
        relevant = {"doc_1", "doc_2"}
        # k=5 but only 2 retrieved — numerator is 2, denominator is 5
        assert precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)

    def test_precision_at_1(self):
        retrieved = ["doc_1", "doc_2"]
        relevant = {"doc_1"}
        assert precision_at_k(retrieved, relevant, 1) == 1.0

    def test_precision_at_1_miss(self):
        retrieved = ["doc_2", "doc_1"]
        relevant = {"doc_1"}
        assert precision_at_k(retrieved, relevant, 1) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1", "doc_2"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_zero_recall(self):
        retrieved = ["doc_4", "doc_5"]
        relevant = {"doc_1", "doc_2"}
        assert recall_at_k(retrieved, relevant, 2) == 0.0

    def test_partial_recall(self):
        retrieved = ["doc_1", "doc_4", "doc_5"]
        relevant = {"doc_1", "doc_2"}
        assert recall_at_k(retrieved, relevant, 3) == pytest.approx(0.5)

    def test_empty_relevant_returns_zero(self):
        retrieved = ["doc_1"]
        relevant = set()
        assert recall_at_k(retrieved, relevant, 1) == 0.0

    def test_recall_improves_with_k(self):
        retrieved = ["doc_4", "doc_1", "doc_2"]
        relevant = {"doc_1", "doc_2"}
        assert recall_at_k(retrieved, relevant, 1) < recall_at_k(retrieved, relevant, 3)


class TestReciprocalRank:
    def test_first_result_relevant(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1"}
        assert reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_result_relevant(self):
        retrieved = ["doc_4", "doc_1", "doc_3"]
        relevant = {"doc_1"}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)

    def test_third_result_relevant(self):
        retrieved = ["doc_4", "doc_5", "doc_1"]
        relevant = {"doc_1"}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant_returns_zero(self):
        retrieved = ["doc_4", "doc_5", "doc_6"]
        relevant = {"doc_1"}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_empty_retrieved_returns_zero(self):
        retrieved = []
        relevant = {"doc_1"}
        assert reciprocal_rank(retrieved, relevant) == 0.0


class TestMeanReciprocalRank:
    def test_perfect_mrr(self):
        queries = [
            {"retrieved": ["doc_1"], "relevant": {"doc_1"}},
            {"retrieved": ["doc_2"], "relevant": {"doc_2"}},
        ]
        assert mean_reciprocal_rank(queries) == 1.0

    def test_empty_queries_returns_zero(self):
        assert mean_reciprocal_rank([]) == 0.0

    def test_mrr_averaging(self):
        queries = [
            {"retrieved": ["doc_1", "doc_2"], "relevant": {"doc_1"}},  # RR = 1.0
            {"retrieved": ["doc_3", "doc_1"], "relevant": {"doc_1"}},  # RR = 0.5
        ]
        assert mean_reciprocal_rank(queries) == pytest.approx(0.75)


class TestNDCGAtK:
    def test_perfect_ndcg(self):
        retrieved = ["doc_1", "doc_2"]
        relevant = {"doc_1", "doc_2"}
        assert ndcg_at_k(retrieved, relevant, 2) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        retrieved = ["doc_3", "doc_4"]
        relevant = {"doc_1", "doc_2"}
        assert ndcg_at_k(retrieved, relevant, 2) == 0.0

    def test_empty_relevant_returns_zero(self):
        retrieved = ["doc_1", "doc_2"]
        relevant = set()
        assert ndcg_at_k(retrieved, relevant, 2) == 0.0

    def test_order_matters(self):
        # Best order: relevant first
        retrieved_good = ["doc_1", "doc_3"]
        retrieved_bad = ["doc_3", "doc_1"]
        relevant = {"doc_1"}
        assert ndcg_at_k(retrieved_good, relevant, 2) > ndcg_at_k(retrieved_bad, relevant, 2)

    def test_ndcg_at_1_perfect(self):
        retrieved = ["doc_1"]
        relevant = {"doc_1"}
        assert ndcg_at_k(retrieved, relevant, 1) == pytest.approx(1.0)

    def test_ndcg_at_1_miss(self):
        retrieved = ["doc_2"]
        relevant = {"doc_1"}
        assert ndcg_at_k(retrieved, relevant, 1) == 0.0


class TestIdealDCG:
    def test_single_relevant(self):
        relevant = {"doc_1"}
        # IDCG@1 = 1/log2(2) = 1.0
        assert ideal_dcg_at_k(relevant, 1) == pytest.approx(1.0)

    def test_k_zero(self):
        relevant = {"doc_1"}
        assert ideal_dcg_at_k(relevant, 0) == 0.0

    def test_more_relevant_than_k(self):
        relevant = {"doc_1", "doc_2", "doc_3", "doc_4", "doc_5"}
        # Only k=2 positions matter
        idcg_k2 = ideal_dcg_at_k(relevant, 2)
        idcg_k5 = ideal_dcg_at_k(relevant, 5)
        assert idcg_k5 > idcg_k2


class TestF1AtK:
    def test_perfect_f1(self):
        retrieved = ["doc_1"]
        relevant = {"doc_1"}
        assert f1_at_k(retrieved, relevant, 1) == pytest.approx(1.0)

    def test_zero_f1(self):
        retrieved = ["doc_2"]
        relevant = {"doc_1"}
        assert f1_at_k(retrieved, relevant, 1) == 0.0

    def test_harmonic_mean(self):
        # P@2 = 0.5, R@2 = 1.0 → F1 = 2*0.5*1.0/(0.5+1.0) = 2/3
        retrieved = ["doc_1", "doc_3"]
        relevant = {"doc_1"}
        p = precision_at_k(retrieved, relevant, 2)
        r = recall_at_k(retrieved, relevant, 2)
        expected = 2 * p * r / (p + r)
        assert f1_at_k(retrieved, relevant, 2) == pytest.approx(expected)


class TestComputeAllMetrics:
    def test_returns_all_metric_keys(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1"}
        k_values = [1, 3]
        metrics = compute_all_metrics(retrieved, relevant, k_values)
        for k in k_values:
            assert f"precision@{k}" in metrics
            assert f"recall@{k}" in metrics
            assert f"ndcg@{k}" in metrics
            assert f"f1@{k}" in metrics
        assert "mrr" in metrics

    def test_default_k_values(self):
        retrieved = ["doc_1"]
        relevant = {"doc_1"}
        metrics = compute_all_metrics(retrieved, relevant)
        for k in [1, 3, 5, 10]:
            assert f"precision@{k}" in metrics

    def test_metrics_in_range(self):
        retrieved = ["doc_1", "doc_3", "doc_2"]
        relevant = {"doc_1", "doc_2"}
        metrics = compute_all_metrics(retrieved, relevant, [1, 3, 5])
        for val in metrics.values():
            assert 0.0 <= val <= 1.0
