"""Tests for hybrid recommender."""
import pytest
import numpy as np
from unittest.mock import MagicMock
from src.models.hybrid import HybridRecommender


@pytest.fixture
def hybrid():
    return HybridRecommender()


class MockRecommender:
    def __init__(self, name):
        self.name = name

    def recommend(self, *args, **kwargs):
        n = kwargs.get("n", 10)
        return [{"movieId": i, "score": round(0.9 - i * 0.05, 3)} for i in range(1, n + 1)]

    def recommend_history(self, *args, **kwargs):
        return self.recommend(*args, **kwargs)


class TestHybridRecommender:
    def test_empty_returns_empty(self, hybrid):
        assert hybrid.recommend(user_id=1) == []

    def test_single_recommender(self, hybrid):
        hybrid.add_recommender("content", MockRecommender("content"), weight=1.0)
        recs = hybrid.recommend(user_history=[1, 2], n=5)
        assert len(recs) > 0
        assert "movieId" in recs[0]
        assert "score" in recs[0]

    def test_multiple_recommenders(self, hybrid):
        hybrid.add_recommender("cf", MockRecommender("cf"), weight=0.5)
        hybrid.add_recommender("content", MockRecommender("content"), weight=0.5)
        recs = hybrid.recommend(user_id=1, user_history=[1, 2], n=5)
        assert len(recs) > 0

    def test_cold_start(self, hybrid):
        hybrid.add_recommender("content", MockRecommender("content"), weight=1.0)
        recs = hybrid.cold_start_recommend({"example_movies": [1, 2]}, n=5)
        assert isinstance(recs, list)

    def test_rerank(self, hybrid):
        recs = [{"movieId": i, "score": round(0.9 - i * 0.05, 3)} for i in range(1, 21)]
        np.random.seed(42)
        features = np.random.randn(100, 50)
        reranked = hybrid.rerank(recs, item_features=features, n=5)
        assert len(reranked) == 5
