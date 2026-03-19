"""Tests for evaluation metrics."""
import pytest
import numpy as np
from src.evaluation import ndcg_at_k, map_at_k, hit_rate_at_k, catalog_coverage, novelty


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


class TestNovelty:
    def test_returns_positive(self):
        recs = {1: [1, 2, 3]}
        pop = {1: 100, 2: 50, 3: 10}
        nov = novelty(recs, pop, n_total=1000)
        assert nov > 0.0
