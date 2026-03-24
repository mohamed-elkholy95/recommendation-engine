"""Collaborative filtering models.

Collaborative filtering (CF) is one of the two classical approaches to
recommendation (the other being content-based filtering). CF works on the
principle that users who agreed in the past will agree in the future — it
discovers latent patterns in user–item interactions without needing any
metadata about users or items.

This module implements matrix factorization via truncated SVD, which is the
foundation behind Netflix Prize-era recommender systems. For a neural
approach, see ``neural_cf.py`` which learns non-linear user–item interactions.

Key Concepts:
    - **User-Item Matrix**: Sparse matrix R where R[u,i] = rating user u gave item i.
    - **Matrix Factorization**: Approximate R ≈ U · Σ · Vᵀ where U captures user
      preferences and V captures item characteristics in a shared latent space.
    - **Cold Start Problem**: CF cannot recommend items with no interactions — see
      content-based filtering (``content_based.py``) for handling new items.

References:
    - Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization
      Techniques for Recommender Systems. IEEE Computer.
    - Rendle, S. et al. (2022). Revisiting the Performance of iALS on
      Item Recommendation Benchmarks. RecSys.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


class MatrixFactorization:
    """Matrix factorization via truncated SVD.

    Decomposes the user-item interaction matrix R into latent factor
    matrices: R ≈ U · Σ · Vᵀ, where:
        - U (n_users × k) encodes user preferences in latent space
        - Σ (k × k) diagonal matrix of singular values (importance weights)
        - V (n_items × k) encodes item characteristics in the same latent space

    The predicted rating for user u on item i is:
        r̂(u, i) = U[u] · V[i]ᵀ + global_mean

    Attributes:
        n_factors: Number of latent dimensions (k). Higher values capture
            more nuanced patterns but risk overfitting on sparse data.
        user_factors: Learned user latent vectors (U · Σ).
        item_factors: Learned item latent vectors (Vᵀ).
        global_mean: Dataset-wide average rating for bias correction.
    """

    def __init__(
        self, n_factors: int = 50, random_state: int = RANDOM_SEED,
    ) -> None:
        self.n_factors = n_factors
        self.random_state = random_state
        self._is_fitted = False
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_mean: float = 0.0
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[int, int] = {}
        self.global_mean: float = 0.0

    def fit(self, ratings: pd.DataFrame) -> "MatrixFactorization":
        """Train matrix factorization via SVD.

        Args:
            ratings: DataFrame with userId, movieId, rating columns.

        Returns:
            Self.
        """
        self.global_mean = ratings["rating"].mean()
        users = ratings["userId"].unique()
        items = ratings["movieId"].unique()
        self.user_map = {uid: i for i, uid in enumerate(users)}
        self.item_map = {mid: j for j, mid in enumerate(items)}

        matrix = np.zeros((len(users), len(items)))
        for _, row in ratings.iterrows():
            i = self.user_map[row["userId"]]
            j = self.item_map[row["movieId"]]
            matrix[i, j] = row["rating"]

        # Center ratings per user to remove individual bias.
        # Without centering, a generous rater (avg 4.5) and a harsh rater
        # (avg 2.0) who like the same movies would look dissimilar.
        # Centering ensures SVD captures relative preferences, not absolute scales.
        user_mean = matrix.mean(axis=1, keepdims=True)
        user_mean[np.isnan(user_mean)] = 0
        matrix_centered = matrix - user_mean

        k = min(self.n_factors, min(matrix_centered.shape) - 1)
        if k < 1:
            k = 1

        # Truncated SVD: keep only the top-k singular values.
        # This is the key dimensionality reduction step — it forces the model
        # to learn a compressed representation that generalizes across users/items
        # rather than memorizing the sparse interaction matrix.
        U, sigma, Vt = svds(csr_matrix(matrix_centered), k=k)

        # Fold singular values into user factors: U_weighted = U · diag(Σ)
        # This way predictions are simply dot(user_factors[u], item_factors[i])
        self.user_factors = U * sigma
        self.item_factors = Vt.T
        self.user_mean = float(user_mean.mean())

        self._is_fitted = True
        logger.info("MF fitted: %d users, %d items, %d factors",
                    len(users), len(items), k)
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair.

        Args:
            user_id: User identifier.
            item_id: Item identifier.

        Returns:
            Predicted rating (clipped to [0.5, 5.0]).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        if user_id not in self.user_map or item_id not in self.item_map:
            return self.global_mean

        u = self.user_factors[self.user_map[user_id]]
        i = self.item_factors[self.item_map[item_id]]
        pred = float(np.dot(u, i)) + self.global_mean
        return np.clip(pred, 0.5, 5.0)

    def recommend(
        self,
        user_id: int,
        rated_items: Optional[set] = None,
        n: int = 10,
    ) -> List[Dict]:
        """Get top-N recommendations for a user.

        Args:
            user_id: User identifier.
            rated_items: Set of already-rated item IDs to exclude.
            n: Number of recommendations.

        Returns:
            List of dicts with movieId and score.
        """
        if not self._is_fitted or user_id not in self.user_map:
            return []

        u = self.user_factors[self.user_map[user_id]]
        scores = self.item_factors @ u + self.global_mean

        if rated_items:
            for mid in rated_items:
                if mid in self.item_map:
                    scores[self.item_map[mid]] = -np.inf

        top_indices = np.argsort(scores)[-n:][::-1]
        reverse_item_map = {v: k for k, v in self.item_map.items()}

        return [
            {"movieId": int(reverse_item_map[idx]), "score": round(float(scores[idx]), 3)}
            for idx in top_indices
        ]

    def similar_items(self, item_id: int, n: int = 10) -> List[Dict]:
        """Find similar items using latent factor cosine similarity.

        Args:
            item_id: Item identifier.
            n: Number of similar items.

        Returns:
            List of dicts with movieId and similarity score.
        """
        if not self._is_fitted or item_id not in self.item_map:
            return []

        idx = self.item_map[item_id]
        item_vec = self.item_factors[idx:idx+1]
        sims = cosine_similarity(item_vec, self.item_factors)[0]

        sims[idx] = -1.0  # Exclude self
        top_indices = np.argsort(sims)[-n:][::-1]
        reverse_item_map = {v: k for k, v in self.item_map.items()}

        return [
            {"movieId": int(reverse_item_map[i]), "similarity": round(float(sims[i]), 3)}
            for i in top_indices if sims[i] > 0
        ][:n]
