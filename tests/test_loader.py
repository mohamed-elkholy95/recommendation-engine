"""Tests for data loading."""
import pytest
import pandas as pd
from src.data.loader import generate_synthetic_movielens, load_movielens, get_stats


class TestGenerateSynthetic:
    def test_keys(self):
        data = generate_synthetic_movielens()
        assert "ratings" in data
        assert "movies" in data

    def test_shapes(self):
        data = generate_synthetic_movielens(n_ratings=1000)
        assert len(data["ratings"]) == 1000
        assert len(data["movies"]) > 0

    def test_rating_range(self):
        data = generate_synthetic_movielens()
        assert data["ratings"]["rating"].min() >= 0.5
        assert data["ratings"]["rating"].max() <= 5.5


class TestGetStats:
    def test_returns_dict(self, movielens_data):
        stats = get_stats(movielens_data["ratings"], movielens_data["movies"])
        assert isinstance(stats, dict)
        assert "n_ratings" in stats
        assert "n_users" in stats
        assert "avg_rating" in stats
