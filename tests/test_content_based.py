"""Tests for content-based filtering."""
import pytest
from src.models.content_based import ContentBasedRecommender


@pytest.fixture
def cb_recommender():
    return ContentBasedRecommender()


class TestContentBased:
    def test_fit(self, cb_recommender, movies_df):
        cb_recommender.fit(movies_df)
        assert cb_recommender._is_fitted

    def test_recommend(self, cb_recommender, movies_df):
        cb_recommender.fit(movies_df)
        recs = cb_recommender.recommend([1, 2, 3], n=5)
        assert len(recs) <= 5
        assert "movieId" in recs[0]
        assert "score" in recs[0]

    def test_similar_items(self, cb_recommender, movies_df):
        cb_recommender.fit(movies_df)
        item_id = movies_df["movieId"].iloc[0]
        sims = cb_recommender.similar_items(item_id, n=5)
        assert len(sims) <= 5

    def test_explain(self, cb_recommender, movies_df):
        cb_recommender.fit(movies_df)
        item_id = movies_df["movieId"].iloc[0]
        explanation = cb_recommender.explain_recommendation(item_id, [1, 2, 3])
        assert isinstance(explanation, dict)
        assert "item_genres" in explanation

    def test_empty_history(self, cb_recommender, movies_df):
        cb_recommender.fit(movies_df)
        recs = cb_recommender.recommend([], n=5)
        assert isinstance(recs, list)
