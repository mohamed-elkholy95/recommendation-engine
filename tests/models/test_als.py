"""Tests for src.models.collaborative.AlsModel — hand-rolled Koren ALS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.ids import ItemIdx, UserIdx
from src.models.collaborative import AlsModel


def _rank_one_frame(user_vec: list[float], item_vec: list[float]) -> pd.DataFrame:
    rows = [
        (u, i, float(a * b))
        for u, a in enumerate(user_vec)
        for i, b in enumerate(item_vec)
    ]
    return pd.DataFrame(
        {
            "user_idx": pd.array([r[0] for r in rows], dtype="int32"),
            "item_idx": pd.array([r[1] for r in rows], dtype="int32"),
            "rating": pd.array([r[2] for r in rows], dtype="float32"),
            "timestamp": pd.array([0] * len(rows), dtype="int64"),
        }
    )


def test_als_reconstructs_rank_one_matrix_within_tolerance() -> None:
    # concept: ALS with k=1 on a rank-1 dense fixture converges to the true
    # outer product (up to a global sign flip). 30 iterations with weak
    # regularisation lands inside 1e-2 per entry.
    user_vec = [1.0, 2.0, 3.0, 4.0, 5.0]
    item_vec = [1.0, 2.0, 3.0, 4.0]
    train = _rank_one_frame(user_vec, item_vec)

    model = AlsModel(n_factors=1, n_iter=30, reg=1e-3, seed=0)
    model.fit(train, n_users=5, n_items=4)

    for u in range(5):
        for i in range(4):
            expected = user_vec[u] * item_vec[i]
            assert abs(model.predict(UserIdx(u), ItemIdx(i)) - expected) < 1e-2


def test_fit_is_deterministic_given_the_same_seed() -> None:
    train = _rank_one_frame([1.0, 2.5, 4.0], [1.0, 3.0, 5.0])

    m1 = AlsModel(n_factors=2, n_iter=10, seed=11)
    m2 = AlsModel(n_factors=2, n_iter=10, seed=11)
    m1.fit(train, n_users=3, n_items=3)
    m2.fit(train, n_users=3, n_items=3)

    assert np.allclose(m1._user_factors, m2._user_factors)
    assert np.allclose(m1._item_factors, m2._item_factors)


def test_predict_before_fit_raises_runtime_error() -> None:
    model = AlsModel(n_factors=2)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(UserIdx(0), ItemIdx(0))


def test_recommend_before_fit_raises_runtime_error() -> None:
    model = AlsModel(n_factors=2)
    with pytest.raises(RuntimeError, match="fit"):
        model.recommend(UserIdx(0), n=3)


def test_fit_rejects_empty_train() -> None:
    empty = pd.DataFrame(
        {
            "user_idx": pd.array([], dtype="int32"),
            "item_idx": pd.array([], dtype="int32"),
            "rating": pd.array([], dtype="float32"),
            "timestamp": pd.array([], dtype="int64"),
        }
    )
    model = AlsModel(n_factors=1)
    with pytest.raises(ValueError, match="at least one"):
        model.fit(empty, n_users=3, n_items=3)


@pytest.mark.parametrize("bad_param", ["n_iter", "reg"])
def test_fit_rejects_non_positive_hyperparams(bad_param: str) -> None:
    kwargs = {"n_factors": 1, "n_iter": 5, "reg": 0.1, "seed": 0}
    kwargs[bad_param] = 0 if bad_param == "n_iter" else -1.0
    model = AlsModel(**kwargs)  # type: ignore[arg-type]
    train = _rank_one_frame([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        model.fit(train, n_users=3, n_items=2)


def test_recommend_excludes_seen_items_by_default() -> None:
    train = _rank_one_frame([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    model = AlsModel(n_factors=1, n_iter=15, seed=0)
    model.fit(train, n_users=3, n_items=3)

    # Every (user, item) pair is seen → recommend returns empty list.
    assert model.recommend(UserIdx(0), n=3) == []


def test_baselines_absorb_a_constant_rating_matrix() -> None:
    # concept: if every observation is the same rating r, μ = r, both biases
    # are zero, and the latent factors have nothing to fit — predict returns r
    # for every pair.
    users, items = 4, 4
    rows = [(u, i, 4.2) for u in range(users) for i in range(items)]
    train = _rank_one_frame([1.0] * users, [1.0] * items).assign(
        user_idx=[r[0] for r in rows],
        item_idx=[r[1] for r in rows],
        rating=[r[2] for r in rows],
    )
    train = train.astype(
        {"user_idx": "int32", "item_idx": "int32", "rating": "float32", "timestamp": "int64"}
    )

    model = AlsModel(n_factors=2, n_iter=5, seed=0)
    model.fit(train, n_users=users, n_items=items)

    for u in range(users):
        for i in range(items):
            assert model.predict(UserIdx(u), ItemIdx(i)) == pytest.approx(4.2, abs=1e-3)


def test_item_bias_lifts_globally_popular_items() -> None:
    # concept: item 0 is rated 5.0 by everyone; item 1 is rated 1.0 by
    # everyone. Both items are fully observed — the latent factors must
    # reconstruct the raw ratings via the item_bias term.
    rows = [(u, 0, 5.0) for u in range(4)] + [(u, 1, 1.0) for u in range(4)]
    train = pd.DataFrame(
        {
            "user_idx": pd.array([r[0] for r in rows], dtype="int32"),
            "item_idx": pd.array([r[1] for r in rows], dtype="int32"),
            "rating": pd.array([r[2] for r in rows], dtype="float32"),
            "timestamp": pd.array([0] * len(rows), dtype="int64"),
        }
    )

    model = AlsModel(n_factors=1, n_iter=10, seed=0)
    model.fit(train, n_users=4, n_items=2)

    # item_bias[0] should be +2.0 (5 - μ=3), item_bias[1] should be -2.0.
    assert model._item_bias[0] == pytest.approx(2.0, abs=1e-3)
    assert model._item_bias[1] == pytest.approx(-2.0, abs=1e-3)
