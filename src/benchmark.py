"""Evaluation harness — compare any model's ``recommend`` against a test split.

``evaluate_ranking`` is deliberately provider-agnostic. It takes any callable
matching ``recommend(user_idx, *, n) -> list[(item_idx, score)]`` so the same
harness works for ``SvdModel``, ``AlsModel``, ``ContentModel``, ``NcfModel``,
``HybridModel``, or a post-``LlmReranker`` stack.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.data.ids import ItemIdx, UserIdx
from src.evaluation import average_precision_at_k, hit_rate_at_k, ndcg_at_k

RecommendFn = Callable[..., list[tuple[ItemIdx, float]]]


def evaluate_ranking(
    recommend_fn: RecommendFn,
    test: pd.DataFrame,
    *,
    k: int,
) -> dict[str, float]:
    """Mean NDCG@k, hit-rate@k, and MAP@k across every user in ``test``.

    Args:
        recommend_fn: Any callable ``fn(user_idx, *, n) → list[(item, score)]``.
        test: Test split — columns ``user_idx`` / ``item_idx`` are read.
        k: Ranking cutoff.

    Returns:
        A dict with keys ``"ndcg@<k>"``, ``"hit_rate@<k>"``, ``"map@<k>"``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if len(test) == 0:
        return {f"ndcg@{k}": 0.0, f"hit_rate@{k}": 0.0, f"map@{k}": 0.0}

    relevant_by_user: dict[UserIdx, set[ItemIdx]] = {}
    for user_raw, item_raw in zip(test["user_idx"], test["item_idx"], strict=True):
        relevant_by_user.setdefault(UserIdx(int(user_raw)), set()).add(ItemIdx(int(item_raw)))

    ndcg_total = 0.0
    hr_total = 0.0
    ap_total = 0.0
    n_users = 0
    for user, relevant in relevant_by_user.items():
        recommended = [item for item, _ in recommend_fn(user, n=k)]
        ndcg_total += ndcg_at_k(recommended, relevant, k)
        hr_total += hit_rate_at_k(recommended, relevant, k)
        ap_total += average_precision_at_k(recommended, relevant, k)
        n_users += 1

    return {
        f"ndcg@{k}": ndcg_total / n_users,
        f"hit_rate@{k}": hr_total / n_users,
        f"map@{k}": ap_total / n_users,
    }
