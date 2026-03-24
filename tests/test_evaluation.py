"""Tests for evaluation metrics.

Each test class verifies a single metric with edge cases:
- Perfect scores (ideal ranking)
- Zero scores (completely wrong ranking)
- Partial scores (realistic middle ground)
- Empty inputs where applicable
"""
import pytest
import numpy as np
from src.evaluation import (
    ndcg_at_k, map_at_k, hit_rate_at_k, catalog_coverage, novelty,
    precision_at_k, recall_at_k, mrr,
)


class TestNDCG:
    def test_perfect_ranking(self):
        y_true = [[1, 2, 3]]
        y_pred = [[1, 2, 3]]
        assert ndcg_at_k(y_true, y_pred, k=3) == 1.0

    def test_worst_ranking(self):
        y_true = [[1, 2, 3]]
        y_pred = [[4, 5, 6]]
        assert ndcg_at_k(y_true, y_pred, k=3) == 0.0

    def test_partial_hit(self):
        y_true = [[1, 2, 3]]
        y_pred = [[3, 4, 1]]
        score = ndcg_at_k(y_true, y_pred, k=3)
        assert 0.0 < score < 1.0


class TestMAP:
    def test_perfect(self):
        y_true = [[1, 2]]
        y_pred = [[1, 2, 3]]
        assert map_at_k(y_true, y_pred, k=3) == 1.0

    def test_zero(self):
        y_true = [[1, 2]]
        y_pred = [[3, 4, 5]]
        assert map_at_k(y_true, y_pred, k=3) == 0.0


class TestHitRate:
    def test_all_hits(self):
        y_true = [[1, 2], [3, 4]]
        y_pred = [[1, 2, 5], [3, 6, 7]]
        assert hit_rate_at_k(y_true, y_pred, k=3) == 1.0

    def test_no_hits(self):
        y_true = [[1, 2], [3, 4]]
        y_pred = [[5, 6, 7], [8, 9, 10]]
        assert hit_rate_at_k(y_true, y_pred, k=3) == 0.0


class TestCoverage:
    def test_full_coverage(self):
        recs = {1: [1, 2, 3], 2: [4, 5, 6]}
        assert catalog_coverage(recs, n_items=6) == 1.0

    def test_partial_coverage(self):
        recs = {1: [1, 2, 3], 2: [1, 2, 4]}
        cov = catalog_coverage(recs, n_items=10)
        assert 0.0 < cov < 1.0


class TestPrecision:
    def test_perfect_precision(self):
        y_true = [[1, 2, 3]]
        y_pred = [[1, 2, 3, 4, 5]]
        assert precision_at_k(y_true, y_pred, k=3) == 1.0

    def test_zero_precision(self):
        y_true = [[1, 2, 3]]
        y_pred = [[4, 5, 6]]
        assert precision_at_k(y_true, y_pred, k=3) == 0.0

    def test_partial_precision(self):
        y_true = [[1, 2, 3]]
        y_pred = [[1, 4, 5]]
        # 1 hit out of 3
        assert abs(precision_at_k(y_true, y_pred, k=3) - 1 / 3) < 1e-6

    def test_multiple_users(self):
        y_true = [[1, 2], [3, 4]]
        y_pred = [[1, 3], [3, 5]]
        # User 1: 1/2, User 2: 1/2 → avg 0.5
        assert abs(precision_at_k(y_true, y_pred, k=2) - 0.5) < 1e-6


class TestRecall:
    def test_perfect_recall(self):
        y_true = [[1, 2]]
        y_pred = [[1, 2, 3, 4, 5]]
        assert recall_at_k(y_true, y_pred, k=5) == 1.0

    def test_zero_recall(self):
        y_true = [[1, 2]]
        y_pred = [[3, 4, 5]]
        assert recall_at_k(y_true, y_pred, k=3) == 0.0

    def test_partial_recall(self):
        y_true = [[1, 2, 3, 4]]
        y_pred = [[1, 5, 6]]
        # 1 hit out of 4 relevant
        assert abs(recall_at_k(y_true, y_pred, k=3) - 0.25) < 1e-6

    def test_empty_true(self):
        """Users with no relevant items are skipped."""
        y_true = [[]]
        y_pred = [[1, 2, 3]]
        assert recall_at_k(y_true, y_pred, k=3) == 0.0


class TestMRR:
    def test_first_position(self):
        y_true = [[1, 2, 3]]
        y_pred = [[1, 4, 5]]
        assert mrr(y_true, y_pred) == 1.0

    def test_second_position(self):
        y_true = [[1]]
        y_pred = [[4, 1, 5]]
        assert mrr(y_true, y_pred) == 0.5

    def test_no_hit(self):
        y_true = [[1, 2]]
        y_pred = [[3, 4, 5]]
        assert mrr(y_true, y_pred) == 0.0

    def test_multiple_users(self):
        y_true = [[1], [2]]
        y_pred = [[1, 3, 4], [4, 5, 2]]
        # User 1: RR=1.0, User 2: RR=1/3 → mean = 2/3
        expected = (1.0 + 1 / 3) / 2
        assert abs(mrr(y_true, y_pred) - expected) < 1e-6


class TestNovelty:
    def test_returns_positive(self):
        recs = {1: [1, 2, 3]}
        pop = {1: 100, 2: 50, 3: 10}
        nov = novelty(recs, pop, n_total=1000)
        assert nov > 0.0

    def test_rare_items_more_novel(self):
        """Rare items should produce higher novelty than popular items."""
        popular_recs = {1: [1]}
        rare_recs = {1: [2]}
        pop = {1: 900, 2: 10}
        n_total = 1000
        nov_popular = novelty(popular_recs, pop, n_total)
        nov_rare = novelty(rare_recs, pop, n_total)
        assert nov_rare > nov_popular
