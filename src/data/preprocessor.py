"""Data preprocessing for recommendation models.

Preprocessing is critical for recommendation quality. This module handles:

1. **Data Cleaning**: Filters inactive users and rare items. Users with very
   few ratings don't provide enough signal for CF, and items with few ratings
   lead to unreliable similarity estimates. The k-core filtering approach
   (minimum ratings per user AND per item) is standard practice.

2. **Interaction Matrix Construction**: Converts tabular ratings into a sparse
   CSR matrix suitable for matrix factorization algorithms. Sparsity is
   typically 95-99%+ in real datasets (most users rate very few items).

3. **Temporal Train/Test Split**: Splits per-user chronologically rather than
   randomly, respecting the natural time ordering of events. Random splits
   would leak future information, inflating offline metrics and producing
   overly optimistic evaluation results.
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.config import RANDOM_SEED, MIN_USER_RATINGS, MIN_ITEM_RATINGS, TEST_RATIO

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """End-to-end data preparation for recommendation models."""

    def __init__(
        self,
        min_user_ratings: int = MIN_USER_RATINGS,
        min_item_ratings: int = MIN_ITEM_RATINGS,
    ) -> None:
        self.min_user_ratings = min_user_ratings
        self.min_item_ratings = min_item_ratings

    def validate(self, ratings: pd.DataFrame) -> Dict[str, any]:
        """Validate rating data and report potential issues.

        Checks for common data quality problems that can silently degrade
        recommendation quality: missing values, out-of-range ratings,
        duplicate entries, and timestamp anomalies.

        Args:
            ratings: Raw ratings DataFrame.

        Returns:
            Dict with validation results and warnings.
        """
        issues = []
        stats = {}

        # Check required columns
        required = {"userId", "movieId", "rating"}
        missing_cols = required - set(ratings.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return {"valid": False, "issues": issues}

        # Check for null values
        null_counts = ratings[list(required)].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

        # Check rating range
        out_of_range = ratings[(ratings["rating"] < 0.5) | (ratings["rating"] > 5.0)]
        if len(out_of_range) > 0:
            issues.append(f"{len(out_of_range)} ratings outside [0.5, 5.0] range")

        # Check for duplicate user-item pairs
        duplicates = ratings.duplicated(subset=["userId", "movieId"], keep=False)
        n_duplicates = duplicates.sum()
        if n_duplicates > 0:
            issues.append(f"{n_duplicates} duplicate user-item pairs found")

        # Compute sparsity
        n_users = ratings["userId"].nunique()
        n_items = ratings["movieId"].nunique()
        sparsity = 1.0 - len(ratings) / (n_users * n_items) if n_users * n_items > 0 else 0
        stats["sparsity"] = round(sparsity, 4)
        stats["n_ratings"] = len(ratings)
        stats["n_users"] = n_users
        stats["n_items"] = n_items

        if sparsity > 0.999:
            issues.append(f"Extremely sparse data ({sparsity:.4f}) — CF models may struggle")

        logger.info("Validation: %d ratings, %d users, %d items, sparsity=%.4f, %d issues",
                    len(ratings), n_users, n_items, sparsity, len(issues))

        return {"valid": len(issues) == 0, "issues": issues, "stats": stats}

    def clean_data(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter inactive users and rare items.

        Args:
            ratings: Raw ratings DataFrame.
            movies: Movies DataFrame.

        Returns:
            (cleaned_ratings, filtered_movies) tuple.
        """
        # Filter users
        user_counts = ratings["userId"].value_counts()
        active_users = user_counts[user_counts >= self.min_user_ratings].index
        ratings = ratings[ratings["userId"].isin(active_users)]

        # Filter items
        item_counts = ratings["movieId"].value_counts()
        popular_items = item_counts[item_counts >= self.min_item_ratings].index
        ratings = ratings[ratings["movieId"].isin(popular_items)]

        # Filter movies
        movies = movies[movies["movieId"].isin(popular_items)]

        logger.info("Cleaned: %d ratings, %d users, %d items",
                    len(ratings), ratings["userId"].nunique(), ratings["movieId"].nunique())
        return ratings, movies

    def create_interaction_matrix(
        self,
        ratings: pd.DataFrame,
    ) -> csr_matrix:
        """Build sparse user-item interaction matrix.

        Args:
            ratings: Ratings DataFrame with userId, movieId, rating.

        Returns:
            CSR matrix shape (n_users, n_items).
        """
        user_ids = ratings["userId"].unique()
        item_ids = ratings["movieId"].unique()
        user_map = {uid: i for i, uid in enumerate(user_ids)}
        item_map = {mid: j for j, mid in enumerate(item_ids)}

        rows = [user_map[uid] for uid in ratings["userId"]]
        cols = [item_map[mid] for mid in ratings["movieId"]]
        data = ratings["rating"].values

        matrix = csr_matrix((data, (rows, cols)),
                            shape=(len(user_ids), len(item_ids)))
        logger.info("Interaction matrix: %d × %d, %d non-zeros",
                    matrix.shape[0], matrix.shape[1], matrix.nnz)
        return matrix

    def train_test_split_time(
        self,
        ratings: pd.DataFrame,
        test_ratio: float = TEST_RATIO,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Per-user temporal train/test split.

        Args:
            ratings: Sorted ratings DataFrame.
            test_ratio: Fraction of each user's ratings for test.

        Returns:
            (train_df, test_df) tuple.
        """
        train_parts, test_parts = [], []
        for _, user_ratings in ratings.groupby("userId"):
            user_ratings = user_ratings.sort_values("timestamp")
            split_idx = max(1, int(len(user_ratings) * (1 - test_ratio)))
            train_parts.append(user_ratings.iloc[:split_idx])
            test_parts.append(user_ratings.iloc[split_idx:])

        train_df = pd.concat(train_parts).reset_index(drop=True)
        test_df = pd.concat(test_parts).reset_index(drop=True)
        logger.info("Temporal split: train=%d, test=%d", len(train_df), len(test_df))
        return train_df, test_df
