"""Tests for preprocessing."""
import pytest
from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def preprocessor():
    return DataPreprocessor(min_user_ratings=5, min_item_ratings=5)


class TestDataPreprocessor:
    def test_clean_data(self, preprocessor, movielens_data):
        ratings, movies = preprocessor.clean_data(
            movielens_data["ratings"], movielens_data["movies"]
        )
        assert isinstance(ratings, pd.DataFrame)
        assert len(ratings) > 0

    def test_interaction_matrix(self, preprocessor, movielens_data):
        ratings, _ = preprocessor.clean_data(
            movielens_data["ratings"], movielens_data["movies"]
        )
        matrix = preprocessor.create_interaction_matrix(ratings)
        assert matrix.shape[0] > 0
        assert matrix.nnz > 0

    def test_train_test_split(self, preprocessor, movielens_data):
        ratings, _ = preprocessor.clean_data(
            movielens_data["ratings"], movielens_data["movies"]
        )
        train, test = preprocessor.train_test_split_time(ratings, test_ratio=0.2)
        assert len(train) + len(test) == len(ratings)
        assert len(test) > 0

    def test_split_preserves_users(self, preprocessor, movielens_data):
        ratings, _ = preprocessor.clean_data(
            movielens_data["ratings"], movielens_data["movies"]
        )
        train, test = preprocessor.train_test_split_time(ratings, test_ratio=0.2)
        train_users = set(train["userId"].unique())
        test_users = set(test["userId"].unique())
        assert train_users == test_users


import pandas as pd  # needed for fixture
