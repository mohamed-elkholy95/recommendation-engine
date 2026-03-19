"""Tests for NCF model."""
import pytest
import torch
import numpy as np
from src.models.neural_cf import NCFModel, train_ncf


@pytest.fixture
def ncf_model():
    return NCFModel(n_users=100, n_items=200, embedding_dim=16, mlp_layers=[32, 16])


class TestNCFModel:
    def test_forward_shape(self, ncf_model):
        users = torch.randint(0, 100, (32,))
        items = torch.randint(0, 200, (32,))
        output = ncf_model(users, items)
        assert output.shape == (32,)
        assert (output >= 0).all() and (output <= 1).all()

    def test_forward_single(self, ncf_model):
        output = ncf_model(torch.tensor([0]), torch.tensor([0]))
        assert output.shape == (1,)
        assert 0.0 <= output.item() <= 1.0

    def test_recommend(self, ncf_model):
        recs = ncf_model.recommend(user_idx=0, n_items=200, n=10)
        assert len(recs) <= 10
        assert len(recs[0]) == 2  # (idx, score)

    def test_recommend_exclude(self, ncf_model):
        recs = ncf_model.recommend(user_idx=0, n_items=200, exclude={0, 1}, n=5)
        indices = [r[0] for r in recs]
        assert 0 not in indices
        assert 1 not in indices

    def test_train_ncf(self, ncf_model, ratings_df):
        user_map = {uid: i for i, uid in enumerate(ratings_df["userId"].unique())}
        item_map = {mid: i for i, mid in enumerate(ratings_df["movieId"].unique())}
        history = train_ncf(
            ncf_model, ratings_df, user_map, item_map,
            n_items=len(item_map), n_epochs=2, batch_size=256,
        )
        assert "train_loss" in history
        assert len(history["train_loss"]) == 2
        # Loss should generally decrease (or at least be finite)
        assert all(np.isfinite(l) for l in history["train_loss"])
