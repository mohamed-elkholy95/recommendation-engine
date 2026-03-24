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


import numpy as np
import pandas as pd


class TestDataValidation:
    """Tests for the validate() data quality check method."""

    def test_valid_data(self, preprocessor, movielens_data):
        result = preprocessor.validate(movielens_data["ratings"])
        assert result["valid"] is True
        assert len(result["issues"]) == 0
        assert result["stats"]["n_ratings"] > 0

    def test_missing_columns(self, preprocessor):
        df = pd.DataFrame({"user": [1], "item": [1]})
        result = preprocessor.validate(df)
        assert result["valid"] is False
        assert any("Missing required columns" in issue for issue in result["issues"])

    def test_null_values_detected(self, preprocessor):
        df = pd.DataFrame({
            "userId": [1, 2, None],
            "movieId": [1, 2, 3],
            "rating": [4.0, 3.5, 5.0],
        })
        result = preprocessor.validate(df)
        assert any("Null" in issue for issue in result["issues"])

    def test_out_of_range_ratings(self, preprocessor):
        df = pd.DataFrame({
            "userId": [1, 2, 3],
            "movieId": [1, 2, 3],
            "rating": [4.0, 6.0, -1.0],  # two out of range
        })
        result = preprocessor.validate(df)
        assert any("outside" in issue for issue in result["issues"])

    def test_duplicate_detection(self, preprocessor):
        df = pd.DataFrame({
            "userId": [1, 1, 2],
            "movieId": [1, 1, 2],
            "rating": [4.0, 3.0, 5.0],
        })
        result = preprocessor.validate(df)
        assert any("duplicate" in issue for issue in result["issues"])

    def test_sparsity_reported(self, preprocessor, movielens_data):
        result = preprocessor.validate(movielens_data["ratings"])
        assert "sparsity" in result["stats"]
        assert 0.0 <= result["stats"]["sparsity"] <= 1.0
