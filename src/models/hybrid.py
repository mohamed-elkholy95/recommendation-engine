"""Hybrid recommender combining multiple approaches.

Hybrid recommendation combines multiple recommendation strategies to
overcome each individual method's weaknesses:

    - **CF** excels at capturing collaborative patterns but suffers from
      cold start (new users/items with no interactions).
    - **Content-based** handles new items well but tends toward
      over-specialization (filter bubbles).
    - **Neural CF** can model non-linear interactions but requires more
      training data and compute.

This module uses **weighted score fusion**: each recommender produces
candidate scores independently, then scores are combined via a weighted
average. This is simple, interpretable, and effective.

For diversity, we apply **Maximal Marginal Relevance (MMR)** re-ranking,
which balances relevance with novelty by penalizing items too similar
to already-selected recommendations.

Taxonomy of hybrid approaches (Burke, 2002):
    1. Weighted — linear combination of scores (used here)
    2. Switching — pick one model based on context
    3. Mixed — present results from multiple models side-by-side
    4. Feature combination — use one model's output as input to another
    5. Cascade — sequential refinement
    6. Feature augmentation — use one model to generate features for another
    7. Meta-level — one model produces a learned representation for another

References:
    - Burke, R. (2002). Hybrid Recommender Systems: Survey and Experiments.
      User Modeling and User-Adapted Interaction.
    - Carbonell, J. & Goldstein, J. (1998). The Use of MMR, Diversity-Based
      Reranking for Reordering Documents. SIGIR.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import ENSEMBLE_WEIGHTS, MMR_DIVERSITY_WEIGHT

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Weighted ensemble recommender with diversity re-ranking.

    Combines CF, content-based, and neural CF scores,
    then applies Maximum Marginal Relevance for diversity.
    """

    def __init__(self) -> None:
        self._recommenders: Dict[str, Any] = {}
        self._weights: Dict[str, float] = ENSEMBLE_WEIGHTS.copy()

    def add_recommender(self, name: str, recommender: Any, weight: float = 1.0) -> None:
        """Register a recommender component.

        Args:
            name: Recommender identifier.
            recommender: Recommender object with recommend() method.
            weight: Weight in ensemble.
        """
        self._recommenders[name] = recommender
        self._weights[name] = weight
        logger.info("Added recommender '%s' (weight=%.2f)", name, weight)

    def recommend(
        self,
        user_id: int = None,
        user_history: Optional[List[int]] = None,
        n: int = 10,
    ) -> List[Dict]:
        """Get ensemble recommendations.

        Args:
            user_id: User identifier (for CF/NCF).
            user_history: List of rated movie IDs.
            n: Number of recommendations.

        Returns:
            Top-N recommendations with per-model scores.
        """
        if not self._recommenders:
            return []

        candidate_scores: Dict[int, Dict[str, float]] = {}

        for name, rec in self._recommenders.items():
            try:
                weight = self._weights.get(name, 1.0)
                if name == "cf" and user_id is not None:
                    results = rec.recommend(user_id, rated_items=set(user_history or []), n=n * 3)
                elif name == "content":
                    results = rec.recommend(user_history or [], n=n * 3)
                elif name == "ncf" and user_id is not None:
                    results = rec.recommend(user_id, n_items=1000, n=n * 3)
                else:
                    continue

                score_key = "score" if "score" in (results[0] if results else {}) else "similarity"
                for r in results:
                    mid = r.get("movieId", r.get("item_id"))
                    if mid is None:
                        continue
                    if mid not in candidate_scores:
                        candidate_scores[mid] = {}
                    candidate_scores[mid][name] = float(r.get(score_key, 0))

            except Exception as exc:
                logger.warning("Recommender '%s' failed: %s", name, exc)

        # Weighted fusion
        fused = []
        total_weight = sum(self._weights.get(n, 1.0) for n in self._recommenders)
        for mid, scores in candidate_scores.items():
            total = sum(
                scores.get(n, 0) * self._weights.get(n, 1.0)
                for n in self._recommenders
            ) / total_weight
            fused.append({"movieId": mid, "score": round(total, 3), "details": scores})

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:n]

    def rerank(
        self,
        recommendations: List[Dict],
        item_features: Optional[np.ndarray] = None,
        diversity_weight: float = MMR_DIVERSITY_WEIGHT,
        n: int = 10,
    ) -> List[Dict]:
        """MMR re-ranking for diversity.

        Args:
            recommendations: Initial ranked recommendations.
            item_features: Feature matrix for similarity computation.
            diversity_weight: 0=pure relevance, 1=pure diversity.
            n: Number of items to return.

        Returns:
            Re-ranked list.
        """
        if not recommendations or len(recommendations) <= n:
            return recommendations

        if item_features is None:
            return recommendations[:n]

        selected = [recommendations[0]]
        remaining = list(recommendations[1:])

        while len(selected) < min(n, len(recommendations)) and remaining:
            best_score = -np.inf
            best_idx = 0
            sel_feat = item_features[selected[-1]["movieId"] % len(item_features)]

            for i, rec in enumerate(remaining):
                relevance = rec["score"]
                cand_feat = item_features[rec["movieId"] % len(item_features)]
                # Simulated similarity
                sim = abs(float(np.dot(sel_feat, cand_feat) /
                               (np.linalg.norm(sel_feat) * np.linalg.norm(cand_feat) + 1e-9)))
                mmr = (1 - diversity_weight) * relevance - diversity_weight * sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def cold_start_recommend(self, user_profile: Dict, n: int = 10) -> List[Dict]:
        """Recommendations for new users with no history.

        Args:
            user_profile: Dict with 'genres' and 'example_movies'.
            n: Number of recommendations.

        Returns:
            List of recommendations based on stated preferences.
        """
        if "content" in self._recommenders:
            # Use example movies as pseudo-history
            history = user_profile.get("example_movies", [])
            return self._recommenders["content"].recommend(history, n=n)

        return []
