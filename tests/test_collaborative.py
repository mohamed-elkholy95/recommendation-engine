"""Tests for collaborative filtering."""
import pytest
from src.models.collaborative import MatrixFactorization


@pytest.fixture
def mf_model():
    return MatrixFactorization(n_factors=10, random_state=42)


class TestMatrixFactorization:
    def test_fit(self, mf_model, ratings_df):
        mf_model.fit(ratings_df)
        assert mf_model._is_fitted

    def test_predict_shape(self, mf_model, ratings_df):
        mf_model.fit(ratings_df)
        user_id = ratings_df["userId"].iloc[0]
        item_id = ratings_df["movieId"].iloc[0]
        pred = mf_model.predict(user_id, item_id)
        assert 0.5 <= pred <= 5.0

    def test_recommend(self, mf_model, ratings_df):
        mf_model.fit(ratings_df)
        user_id = ratings_df["userId"].iloc[0]
        recs = mf_model.recommend(user_id, n=5)
        assert len(recs) <= 5
        assert "movieId" in recs[0]
        assert "score" in recs[0]

    def test_similar_items(self, mf_model, ratings_df):
        mf_model.fit(ratings_df)
        item_id = ratings_df["movieId"].iloc[0]
        sims = mf_model.similar_items(item_id, n=5)
        assert len(sims) <= 5
        assert "movieId" in sims[0]
        assert "similarity" in sims[0]

    def test_unknown_user(self, mf_model, ratings_df):
        mf_model.fit(ratings_df)
        pred = mf_model.predict(999999, 999999)
        assert isinstance(pred, float)

    def test_unfitted_raises(self, mf_model):
        with pytest.raises(RuntimeError):
            mf_model.predict(1, 1)
