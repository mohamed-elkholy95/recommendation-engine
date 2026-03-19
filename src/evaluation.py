"""Evaluation metrics for recommendation systems."""
import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def ndcg_at_k(y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Args:
        y_true: Ground truth relevant items per user.
        y_pred: Predicted ranked items per user.
        k: Cutoff.

    Returns:
        Average NDCG@K across all users.
    """
    ndcg_values = []
    for true_items, pred_items in zip(y_true, y_pred):
        true_set = set(true_items)
        dcg = 0.0
        for i, item in enumerate(pred_items[:k]):
            if item in true_set:
                dcg += 1.0 / np.log2(i + 2)

        # Ideal DCG
        ideal_hits = min(len(true_set), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_values)) if ndcg_values else 0.0


def map_at_k(y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
    """Mean Average Precision at K.

    Args:
        y_true: Ground truth relevant items per user.
        y_pred: Predicted ranked items per user.
        k: Cutoff.

    Returns:
        Average MAP@K.
    """
    ap_values = []
    for true_items, pred_items in zip(y_true, y_pred):
        true_set = set(true_items)
        hits = 0
        precision_sum = 0.0
        for i, item in enumerate(pred_items[:k]):
            if item in true_set:
                hits += 1
                precision_sum += hits / (i + 1)
        ap_values.append(precision_sum / min(len(true_set), k) if true_set else 0.0)

    return float(np.mean(ap_values)) if ap_values else 0.0


def hit_rate_at_k(y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
    """Fraction of users with at least one hit in top-K.

    Args:
        y_true: Ground truth relevant items.
        y_pred: Predictions.
        k: Cutoff.

    Returns:
        Hit rate.
    """
    hits = 0
    for true_items, pred_items in zip(y_true, y_pred):
        if set(true_items) & set(pred_items[:k]):
            hits += 1
    return hits / len(y_true) if y_true else 0.0


def catalog_coverage(
    recommendations: Dict[int, List[int]],
    n_items: int,
) -> float:
    """Fraction of catalog recommended to at least one user.

    Args:
        recommendations: Dict mapping user_id to recommended item list.
        n_items: Total number of items in catalog.

    Returns:
        Coverage ratio in [0, 1].
    """
    all_recommended = set()
    for items in recommendations.values():
        all_recommended.update(items)
    return len(all_recommended) / n_items if n_items > 0 else 0.0


def intra_list_diversity(
    recommendations: Dict[int, List[int]],
    item_features: np.ndarray,
) -> float:
    """Average pairwise distance within recommendation lists.

    Args:
        recommendations: User to recommended items mapping.
        item_features: Feature matrix (n_items, n_features).

    Returns:
        Mean diversity score.
    """
    diversities = []
    for items in recommendations.values():
        if len(items) < 2:
            continue
        indices = [i % len(item_features) for i in items]
        feats = item_features[indices]
        # Pairwise cosine distance
        normed = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        sim_matrix = normed @ normed.T
        n = len(items)
        # Upper triangle average (1 - similarity = distance)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        distances = 1.0 - sim_matrix[mask]
        diversities.append(float(np.mean(distances)))

    return float(np.mean(diversities)) if diversities else 0.0


def novelty(
    recommendations: Dict[int, List[int]],
    item_popularity: Dict[int, int],
    n_total: int,
) -> float:
    """Average self-information of recommendations.

    Args:
        recommendations: User to items mapping.
        item_popularity: Item ID to frequency count.
        n_total: Total number of interactions.

    Returns:
        Mean novelty score.
    """
    novelties = []
    for items in recommendations.values():
        for item in items:
            prob = item_popularity.get(item, 1) / n_total
            novelties.append(-np.log2(prob))
    return float(np.mean(novelties)) if novelties else 0.0
