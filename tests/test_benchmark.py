"""Tests for src.benchmark — mean NDCG / HR / MAP across a test split."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.benchmark import evaluate_ranking
from src.data.ids import ItemIdx, UserIdx


def _test_frame(rows: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_idx": pd.array([r[0] for r in rows], dtype="int32"),
            "item_idx": pd.array([r[1] for r in rows], dtype="int32"),
            "rating": pd.array([5.0] * len(rows), dtype="float32"),
            "timestamp": pd.array([0] * len(rows), dtype="int64"),
        }
    )


def test_evaluate_ranking_averages_hand_computed_per_user_metrics() -> None:
    # Two test users. Hand-computed expected values follow below.
    test = _test_frame([(0, 3), (0, 5), (1, 2)])

    # User 0: relevant={3, 5}. recommend returns [3, 1, 5, 2, 4] — top-3=[3,1,5]
    #   NDCG@3 = (1/log2(2) + 0 + 1/log2(4)) / (1/log2(2) + 1/log2(3))
    #          = 1.5 / (1 + 1/log2(3))
    #   HR@3 = 1.0
    #   AP@3 = (1/1 + 2/3) / min(3, 2) = (1 + 2/3) / 2 = 5/6
    # User 1: relevant={2}. recommend returns [3, 1, 5] — no hit.
    #   NDCG@3 = 0, HR@3 = 0, AP@3 = 0
    def recommend_fn(user: UserIdx, *, n: int) -> list[tuple[ItemIdx, float]]:
        if int(user) == 0:
            ordered = [3, 1, 5, 2, 4]
        else:
            ordered = [3, 1, 5]
        return [(ItemIdx(i), float(len(ordered) - rank)) for rank, i in enumerate(ordered[:n])]

    metrics = evaluate_ranking(recommend_fn, test, k=3)

    user0_ndcg = (1.0 / math.log2(2) + 1.0 / math.log2(4)) / (
        1.0 / math.log2(2) + 1.0 / math.log2(3)
    )
    user0_ap = (1.0 + 2.0 / 3.0) / 2.0
    assert metrics["ndcg@3"] == pytest.approx(user0_ndcg / 2)
    assert metrics["hit_rate@3"] == pytest.approx(0.5)
    assert metrics["map@3"] == pytest.approx(user0_ap / 2)


def test_evaluate_ranking_returns_zero_metrics_on_empty_test() -> None:
    empty = _test_frame([])

    def recommend_fn(user: UserIdx, *, n: int) -> list[tuple[ItemIdx, float]]:
        raise AssertionError("recommend should not be called on empty test")

    metrics = evaluate_ranking(recommend_fn, empty, k=3)
    assert metrics == {"ndcg@3": 0.0, "hit_rate@3": 0.0, "map@3": 0.0}


def test_evaluate_ranking_ignores_users_with_no_relevant_items_for_hit_rate() -> None:
    # A user who has no ratings in test should not appear in the average.
    test = _test_frame([(0, 3), (0, 5)])

    def recommend_fn(user: UserIdx, *, n: int) -> list[tuple[ItemIdx, float]]:
        return [(ItemIdx(3), 1.0), (ItemIdx(5), 0.9), (ItemIdx(1), 0.1)]

    metrics = evaluate_ranking(recommend_fn, test, k=3)
    assert metrics["hit_rate@3"] == pytest.approx(1.0)


def test_evaluate_ranking_rejects_non_positive_k() -> None:
    test = _test_frame([(0, 3)])

    def recommend_fn(user: UserIdx, *, n: int) -> list[tuple[ItemIdx, float]]:
        return []

    with pytest.raises(ValueError, match="k must"):
        evaluate_ranking(recommend_fn, test, k=0)
