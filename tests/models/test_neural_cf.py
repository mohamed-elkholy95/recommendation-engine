"""Tests for src.models.neural_cf — two-tower MLP with implicit feedback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.ids import ItemIdx, UserIdx
from src.models.neural_cf import NcfModel


def _ratings_frame(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_idx": pd.array([r[0] for r in rows], dtype="int32"),
            "item_idx": pd.array([r[1] for r in rows], dtype="int32"),
            "rating": pd.array([r[2] for r in rows], dtype="float32"),
            "timestamp": pd.array([0] * len(rows), dtype="int64"),
        }
    )


def _separable_clusters_frame() -> pd.DataFrame:
    # concept: two disjoint (user, item) clusters sharing no overlap — the
    # model must learn to score same-cluster pairs high and cross-cluster low.
    positives = [
        (u, i, 5.0) for u in range(4) for i in range(4)
    ] + [(u, i, 5.0) for u in range(4, 8) for i in range(4, 8)]
    return _ratings_frame(positives)


def test_ncf_learns_to_rank_same_cluster_above_cross_cluster() -> None:
    train = _separable_clusters_frame()
    model = NcfModel(
        n_factors=8,
        hidden=(16, 8),
        n_epochs=80,
        batch_size=16,
        lr=5e-3,
        negatives_per_positive=3,
        seed=0,
    )
    model.fit(train, n_users=8, n_items=8)

    # user 0 is in cluster A → item 0 (A) should score higher than item 4 (B).
    same_cluster = model.predict(UserIdx(0), ItemIdx(0))
    cross_cluster = model.predict(UserIdx(0), ItemIdx(4))
    assert same_cluster > cross_cluster


def test_fit_is_deterministic_given_the_same_seed() -> None:
    train = _separable_clusters_frame()

    m1 = NcfModel(n_factors=4, hidden=(8,), n_epochs=5, batch_size=8, seed=11)
    m2 = NcfModel(n_factors=4, hidden=(8,), n_epochs=5, batch_size=8, seed=11)
    m1.fit(train, n_users=8, n_items=8)
    m2.fit(train, n_users=8, n_items=8)

    # Predict the full (user, item) grid and compare.
    s1 = np.array(
        [m1.predict(UserIdx(u), ItemIdx(i)) for u in range(8) for i in range(8)]
    )
    s2 = np.array(
        [m2.predict(UserIdx(u), ItemIdx(i)) for u in range(8) for i in range(8)]
    )
    assert np.allclose(s1, s2, atol=1e-6)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        NcfModel().predict(UserIdx(0), ItemIdx(0))


def test_recommend_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        NcfModel().recommend(UserIdx(0), n=3)


def test_fit_rejects_empty_train() -> None:
    with pytest.raises(ValueError, match="at least one"):
        NcfModel(n_epochs=1).fit(_ratings_frame([]), n_users=4, n_items=4)


def test_fit_rejects_train_without_any_positive_above_threshold() -> None:
    train = _ratings_frame([(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)])
    with pytest.raises(ValueError, match="positive"):
        NcfModel(positive_threshold=4.0, n_epochs=1).fit(
            train, n_users=2, n_items=2
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_epochs": 0}, "n_epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"negatives_per_positive": 0}, "negatives"),
        ({"lr": 0.0}, "lr"),
    ],
)
def test_fit_rejects_non_positive_hyperparams(
    kwargs: dict[str, float], match: str
) -> None:
    train = _separable_clusters_frame()
    model = NcfModel(n_factors=4, hidden=(8,), **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=match):
        model.fit(train, n_users=8, n_items=8)


def test_recommend_excludes_seen_items_by_default() -> None:
    train = _separable_clusters_frame()
    model = NcfModel(n_factors=4, hidden=(8,), n_epochs=3, seed=0)
    model.fit(train, n_users=8, n_items=8)

    # concept: user 0 rated items 0-3; exclude_seen should drop those from recs.
    recs = model.recommend(UserIdx(0), n=8)
    recommended_items = {int(i) for i, _ in recs}
    assert recommended_items.isdisjoint({0, 1, 2, 3})


def test_recommend_rejects_non_positive_n() -> None:
    train = _separable_clusters_frame()
    model = NcfModel(n_factors=4, hidden=(8,), n_epochs=1, seed=0)
    model.fit(train, n_users=8, n_items=8)
    with pytest.raises(ValueError, match="n must"):
        model.recommend(UserIdx(0), n=0)


def test_device_auto_picks_cuda_when_available() -> None:
    # concept: "auto" must select CUDA if torch.cuda.is_available(); otherwise
    # fall back to CPU. This is the only cross-device contract callers depend on.
    import torch

    train = _separable_clusters_frame()
    model = NcfModel(n_factors=4, hidden=(8,), n_epochs=1, seed=0, device="auto")
    model.fit(train, n_users=8, n_items=8)

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert model._device.type == expected


def test_device_respects_explicit_cpu_override() -> None:
    train = _separable_clusters_frame()
    model = NcfModel(n_factors=4, hidden=(8,), n_epochs=1, seed=0, device="cpu")
    model.fit(train, n_users=8, n_items=8)

    assert model._device.type == "cpu"
