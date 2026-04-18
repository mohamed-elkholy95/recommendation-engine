"""Tests for src.models.collaborative — the SVD baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.ids import ItemIdx, UserIdx
from src.models.collaborative import SvdModel


def _frame(
    user_idx: list[int], item_idx: list[int], rating: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_idx": pd.array(user_idx, dtype="int32"),
            "item_idx": pd.array(item_idx, dtype="int32"),
            "rating": pd.array(rating, dtype="float32"),
            "timestamp": pd.array([0] * len(rating), dtype="int64"),
        }
    )


def _rank_one_frame(user_vec: list[float], item_vec: list[float]) -> pd.DataFrame:
    rows = []
    for u, a in enumerate(user_vec):
        for i, b in enumerate(item_vec):
            rows.append((u, i, float(a * b)))
    return _frame(
        [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
    )


def test_svd_reconstructs_rank_one_matrix_within_tolerance() -> None:
    # concept: a true rank-1 matrix is fully recovered by rank-1 SVD.
    user_vec = [1.0, 2.0, 3.0, 4.0, 5.0]
    item_vec = [1.0, 2.0, 3.0, 4.0]
    train = _rank_one_frame(user_vec, item_vec)

    model = SvdModel(n_factors=1, seed=0)
    model.fit(train, n_users=5, n_items=4)

    for u in range(5):
        for i in range(4):
            expected = user_vec[u] * item_vec[i]
            assert abs(model.predict(UserIdx(u), ItemIdx(i)) - expected) < 1e-3


def test_predict_before_fit_raises_runtime_error() -> None:
    model = SvdModel(n_factors=2)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(UserIdx(0), ItemIdx(0))


def test_recommend_before_fit_raises_runtime_error() -> None:
    model = SvdModel(n_factors=2)
    with pytest.raises(RuntimeError, match="fit"):
        model.recommend(UserIdx(0), n=5)


def test_fit_rejects_empty_train() -> None:
    empty = _frame([], [], [])
    model = SvdModel(n_factors=1)
    with pytest.raises(ValueError, match="at least one"):
        model.fit(empty, n_users=3, n_items=3)


def test_fit_rejects_n_factors_too_large() -> None:
    train = _rank_one_frame([1.0, 2.0, 3.0], [1.0, 2.0])  # 3×2 matrix
    model = SvdModel(n_factors=2)  # must be < min(3, 2) = 2
    with pytest.raises(ValueError, match="n_factors"):
        model.fit(train, n_users=3, n_items=2)


def test_recommend_excludes_seen_items_by_default() -> None:
    train = _rank_one_frame([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0])
    model = SvdModel(n_factors=1, seed=0)
    model.fit(train, n_users=5, n_items=4)

    recs = model.recommend(UserIdx(0), n=4)
    # gate: every item was seen during training → no item can be recommended.
    assert recs == []


def test_recommend_includes_seen_items_when_requested() -> None:
    train = _rank_one_frame([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0])
    model = SvdModel(n_factors=1, seed=0)
    model.fit(train, n_users=5, n_items=4)

    recs = model.recommend(UserIdx(0), n=4, exclude_seen=False)
    assert len(recs) == 4
    # rank-1 SVD: for user 0 (scale 1.0), items rank by item_vec → [3, 2, 1, 0].
    assert [int(i) for i, _ in recs] == [3, 2, 1, 0]


def test_recommend_returns_n_items_sorted_descending() -> None:
    user_vec = [1.0, 2.0, 3.0, 4.0, 5.0]
    item_vec = [1.0, 2.0, 3.0, 4.0]
    # concept: drop one rating so there is exactly one unseen item for user 0.
    full = _rank_one_frame(user_vec, item_vec)
    train = full[~((full["user_idx"] == 0) & (full["item_idx"] == 3))].copy()

    model = SvdModel(n_factors=1, seed=0)
    model.fit(train, n_users=5, n_items=4)

    recs = model.recommend(UserIdx(0), n=2)
    assert len(recs) == 1
    assert int(recs[0][0]) == 3


def test_recommend_rejects_non_positive_n() -> None:
    train = _rank_one_frame([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    model = SvdModel(n_factors=1, seed=0)
    model.fit(train, n_users=3, n_items=3)
    with pytest.raises(ValueError, match="n must"):
        model.recommend(UserIdx(0), n=0)


def test_fit_is_deterministic_given_the_same_seed() -> None:
    train = _rank_one_frame([1.0, 2.5, 4.0], [1.0, 3.0, 5.0])
    m1 = SvdModel(n_factors=1, seed=7)
    m2 = SvdModel(n_factors=1, seed=7)
    m1.fit(train, n_users=3, n_items=3)
    m2.fit(train, n_users=3, n_items=3)

    # gate: deterministic SVD — same seed, same data → same factor matrices.
    assert np.allclose(m1._user_factors, m2._user_factors)
    assert np.allclose(m1._item_factors, m2._item_factors)
