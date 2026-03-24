"""Content-based filtering recommender.

Content-based filtering recommends items similar to what the user has liked
before, using item *features* (metadata) rather than other users' behavior.
This is the complement to collaborative filtering — it doesn't need other
users but does need rich item descriptions.

Approach:
    1. **Feature Extraction**: Convert raw metadata (genres, descriptions)
       into numerical vectors using TF-IDF and multi-hot encoding.
    2. **User Profile**: Average the feature vectors of items the user liked
       to create a user preference vector.
    3. **Similarity**: Rank all items by cosine similarity to the user profile.

Advantages over CF:
    - No cold start for *new items* (only needs metadata, not ratings)
    - Explainable ("recommended because you liked similar genres")
    - Works with a single user's history

Limitations:
    - Suffers from *over-specialization* (filter bubble) — only recommends
      items similar to what the user already knows
    - Cannot capture serendipitous cross-genre discoveries
    - Quality depends heavily on feature engineering

References:
    - Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based
      Recommender Systems: State of the Art and Trends. Recommender Systems
      Handbook, Springer.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import FEATURE_WEIGHTS, RANDOM_SEED

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """Content-based recommender using movie metadata features.

    Combines genre, description, cast, and director signals
    into a weighted feature space for similarity-based recommendations.
    """

    def __init__(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.feature_weights = feature_weights or FEATURE_WEIGHTS
        self._is_fitted = False
        self._feature_matrix: Optional[np.ndarray] = None
        self._similarity_matrix: Optional[np.ndarray] = None
        self._movie_ids: List[int] = []
        self._movie_map: Dict[int, int] = {}
        self._mlb: Optional[MultiLabelBinarizer] = None
        self._genre_matrix: Optional[np.ndarray] = None
        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[np.ndarray] = None

    def fit(self, movies: pd.DataFrame) -> "ContentBasedRecommender":
        """Build feature matrices from movie metadata.

        Args:
            movies: DataFrame with movieId, title, genres columns.
                Optional: overview, cast, director columns.

        Returns:
            Self.
        """
        self._movie_ids = movies["movieId"].tolist()
        self._movie_map = {mid: i for i, mid in enumerate(self._movie_ids)}

        # Genre features (multi-hot)
        genres = movies["genres"].fillna("").str.split("|")
        self._mlb = MultiLabelBinarizer()
        genre_mat = self._mlb.fit_transform(genres).astype(float)
        if hasattr(self._mlb, 'classes_'):
            genre_mat = genre_mat[:, :len(self._mlb.classes_)]
        self._genre_matrix = genre_mat

        # Description features (TF-IDF on titles if no overview)
        text_col = "overview" if "overview" in movies.columns else "title"
        texts = movies[text_col].fillna("").tolist()
        self._tfidf = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), stop_words="english",
        )
        self._tfidf_matrix = self._tfidf.fit_transform(texts).astype(float)

        # Combine features
        genre_weight = self.feature_weights.get("genre", 0.4)
        desc_weight = self.feature_weights.get("description", 0.3)

        # Normalize to same scale — use genre as primary, TF-IDF as supplement
        from sklearn.preprocessing import normalize as sk_normalize
        from sklearn.decomposition import TruncatedSVD
        genre_norm = sk_normalize(genre_mat, norm='l2')
        
        # Reduce TF-IDF to same dimension as genre features for combination
        n_components = min(genre_norm.shape[1], self._tfidf_matrix.shape[1])
        if n_components < self._tfidf_matrix.shape[1]:
            svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
            tfidf_reduced = svd.fit_transform(self._tfidf_matrix)
            tfidf_norm = sk_normalize(tfidf_reduced, norm='l2')
        else:
            tfidf_norm = sk_normalize(
                self._tfidf_matrix.toarray() if hasattr(self._tfidf_matrix, 'toarray') else self._tfidf_matrix,
                norm='l2')

        self._feature_matrix = genre_weight * genre_norm + desc_weight * tfidf_norm

        # Pre-compute similarity matrix
        self._similarity_matrix = cosine_similarity(self._feature_matrix)

        self._is_fitted = True
        logger.info("Content-based fitted: %d movies, %d genre features, %d tfidf features",
                    len(self._movie_ids), genre_mat.shape[1], self._tfidf_matrix.shape[1])
        return self

    def recommend(
        self,
        user_history: List[int],
        n: int = 10,
        exclude_seen: bool = True,
    ) -> List[Dict]:
        """Recommend items based on user history.

        Args:
            user_history: List of movie IDs the user has rated.
            n: Number of recommendations.
            exclude_seen: Exclude already-seen items.

        Returns:
            List of dicts with movieId and score.
        """
        if not self._is_fitted or not user_history:
            return self._popular_items(n)

        valid = [mid for mid in user_history if mid in self._movie_map]
        if not valid:
            return self._popular_items(n)

        # Build user profile as average of seen item features
        indices = [self._movie_map[mid] for mid in valid]
        user_profile = self._feature_matrix[indices].mean(axis=0)

        # Compute similarity to all items
        sims = cosine_similarity(user_profile.reshape(1, -1), self._feature_matrix)[0]

        # Exclude seen
        if exclude_seen:
            seen_indices = set(self._movie_map.get(mid, -1) for mid in user_history)
            for idx in seen_indices:
                if 0 <= idx < len(sims):
                    sims[idx] = -1.0

        top_indices = np.argsort(sims)[-n:][::-1]
        return [
            {"movieId": int(self._movie_ids[i]), "score": round(float(sims[i]), 3)}
            for i in top_indices if sims[i] > 0
        ][:n]

    def similar_items(self, item_id: int, n: int = 10) -> List[Dict]:
        """Find content-similar movies.

        Args:
            item_id: Movie identifier.
            n: Number of similar items.

        Returns:
            List of dicts with movieId and similarity score.
        """
        if not self._is_fitted or item_id not in self._movie_map:
            return []

        idx = self._movie_map[item_id]
        sims = self._similarity_matrix[idx].copy()
        sims[idx] = -1.0

        top_indices = np.argsort(sims)[-n:][::-1]
        return [
            {"movieId": int(self._movie_ids[i]), "similarity": round(float(sims[i]), 3)}
            for i in top_indices if sims[i] > 0
        ][:n]

    def explain_recommendation(
        self,
        item_id: int,
        user_history: List[int],
    ) -> Dict:
        """Explain why an item was recommended.

        Args:
            item_id: Recommended movie ID.
            user_history: User's rated movie IDs.

        Returns:
            Dict with explanation details.
        """
        if not self._is_fitted or item_id not in self._movie_map:
            return {"reason": "Cold start — no explanation available"}

        idx = self._movie_map[item_id]
        item_genre = self._genre_matrix[idx] > 0
        genre_names = []
        if hasattr(self._mlb, 'classes_'):
            genre_names = list(self._mlb.classes_[item_genre])

        # Find most similar items in history
        history_sims = []
        for mid in user_history:
            if mid in self._movie_map and mid != item_id:
                hidx = self._movie_map[mid]
                sim = float(self._similarity_matrix[idx, hidx])
                history_sims.append({"movieId": int(mid), "similarity": round(sim, 3)})
        history_sims.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "item_genres": genre_names,
            "similar_to": history_sims[:3],
        }

    def _popular_items(self, n: int) -> List[Dict]:
        """Fallback: return items with highest average feature norm."""
        norms = np.linalg.norm(self._feature_matrix, axis=1)
        top = np.argsort(norms)[-n:][::-1]
        return [
            {"movieId": int(self._movie_ids[i]), "score": round(float(norms[i]), 3)}
            for i in top
        ]
