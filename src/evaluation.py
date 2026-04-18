"""Ranking metrics with the binary-relevance conventions used throughout the project.

Each metric operates on a single user's ``recommended`` ranking plus the set of
``relevant`` items. Callers take the mean across users to get batch metrics
(e.g. MAP@k is the mean of ``average_precision_at_k`` over the test users).

All three metrics return 0 when ``relevant`` is empty — the convention that
spares callers from handling the "no ground truth" branch everywhere.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.data.ids import ItemIdx


def hit_rate_at_k(recommended: Sequence[ItemIdx], relevant: set[ItemIdx], k: int) -> float:
    """1.0 if any of the top-``k`` recommended items is in ``relevant``, else 0.0."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not relevant:
        return 0.0
    for item in recommended[:k]:
        if item in relevant:
            return 1.0
    return 0.0


def ndcg_at_k(recommended: Sequence[ItemIdx], relevant: set[ItemIdx], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance at cutoff ``k``.

    ``DCG = Σ rel(i) / log2(i + 1)`` for ranks ``i ∈ [1..k]``; ``IDCG`` is the
    DCG of the ideal ordering that places every relevant item at the top.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: Sequence[ItemIdx], relevant: set[ItemIdx], k: int) -> float:
    """Average precision at cutoff ``k`` — the per-user term inside MAP@k.

    ``AP@k = (Σ P(i) · rel(i)) / min(k, |relevant|)`` for ranks ``i ∈ [1..k]``,
    where ``P(i)`` is precision at rank ``i``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank

    denom = min(k, len(relevant))
    return precision_sum / denom if denom > 0 else 0.0
