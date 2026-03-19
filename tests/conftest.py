"""Test fixtures."""
import os, sys, pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import generate_synthetic_movielens


@pytest.fixture
def movielens_data():
    return generate_synthetic_movielens(n_users=100, n_movies=50, n_ratings=5000)


@pytest.fixture
def ratings_df(movielens_data):
    return movielens_data["ratings"]


@pytest.fixture
def movies_df(movielens_data):
    return movielens_data["movies"]
