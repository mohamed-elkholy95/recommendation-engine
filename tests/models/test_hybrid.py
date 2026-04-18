"""Tests for src.models.hybrid — weighted score fusion + MMR re-rank."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.ids import ItemIdx, PreprocessedData, RawMovieId, RawUserId, UserIdx
from src.models.collaborative import SvdModel
from src.models.content_based import ContentModel
from src.models.hybrid import HybridModel, min_max_normalise, mmr_rerank
from src.models.neural_cf import NcfModel


# -----------------------------------------------------------------------------
# min_max_normalise (pure function)
# -----------------------------------------------------------------------------


def test_min_max_spreads_scores_across_zero_one() -> None:
    scores = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    got = min_max_normalise(scores)
    assert np.allclose(got, [0.0, 0.5, 1.0])


def test_min_max_returns_zero_when_all_scores_equal() -> None:
    got = min_max_normalise(np.array([2.0, 2.0, 2.0], dtype=np.float32))
    assert np.allclose(got, [0.0, 0.0, 0.0])


# -----------------------------------------------------------------------------
# mmr_rerank (pure function)
# -----------------------------------------------------------------------------


def test_mmr_lambda_one_keeps_relevance_order() -> None:
    # gate: λ = 1 disables the diversity penalty → MMR == top-n by relevance.
    candidates = [
        (ItemIdx(0), 1.0),
        (ItemIdx(1), 0.8),
        (ItemIdx(2), 0.6),
    ]
    similarity = np.array(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]], dtype=np.float32
    )
    out = mmr_rerank(candidates, similarity=similarity, n=2, mmr_lambda=1.0)
    assert [int(i) for i, _ in out] == [0, 1]


def test_mmr_lambda_half_prefers_diverse_over_similar() -> None:
    # Hand-computed: with λ=0.5 on {A:1.0, B:0.8, C:0.6} and sim(A,B)=0.9,
    # sim(A,C)=0.1, iter 2 scores B = 0.5*0.8 - 0.5*0.9 = -0.05 vs
    # C = 0.5*0.6 - 0.5*0.1 = 0.25 → pick C instead of B.
    candidates = [
        (ItemIdx(0), 1.0),
        (ItemIdx(1), 0.8),
        (ItemIdx(2), 0.6),
    ]
    similarity = np.array(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]], dtype=np.float32
    )
    out = mmr_rerank(candidates, similarity=similarity, n=2, mmr_lambda=0.5)
    assert [int(i) for i, _ in out] == [0, 2]


def test_mmr_returns_empty_for_empty_candidates() -> None:
    out = mmr_rerank([], similarity=np.zeros((0, 0), dtype=np.float32), n=5, mmr_lambda=0.7)
    assert out == []


def test_mmr_caps_at_candidate_pool_size() -> None:
    candidates = [(ItemIdx(0), 1.0), (ItemIdx(1), 0.5)]
    similarity = np.eye(2, dtype=np.float32)
    out = mmr_rerank(candidates, similarity=similarity, n=10, mmr_lambda=1.0)
    assert len(out) == 2


# -----------------------------------------------------------------------------
# HybridModel — end-to-end integration
# -----------------------------------------------------------------------------


def _hybrid_fixtures() -> tuple[pd.DataFrame, PreprocessedData]:
    # concept: 8 users × 8 items with two disjoint co-rating clusters so every
    # component model has a signal to exploit. Cluster A = items 0-3,
    # cluster B = items 4-7. Users 0-3 rate cluster A; users 4-7 rate cluster B.
    rows = [(u, i, 5.0) for u in range(4) for i in range(4)] + [
        (u, i, 5.0) for u in range(4, 8) for i in range(4, 8)
    ]
    train = pd.DataFrame(
        {
            "user_idx": pd.array([r[0] for r in rows], dtype="int32"),
            "item_idx": pd.array([r[1] for r in rows], dtype="int32"),
            "rating": pd.array([r[2] for r in rows], dtype="float32"),
            "timestamp": pd.array([0] * len(rows), dtype="int64"),
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": list(range(100, 108)),
            "title": [f"Movie{i}" for i in range(8)],
            "genres": [
                "Action",
                "Action",
                "Action",
                "Action",
                "Comedy",
                "Comedy",
                "Comedy",
                "Comedy",
            ],
        }
    )
    empty = train.iloc[:0].copy()
    pre = PreprocessedData(
        train=train,
        val=empty,
        test=empty,
        user_map={RawUserId(u): UserIdx(u) for u in range(8)},
        item_map={RawMovieId(100 + i): ItemIdx(i) for i in range(8)},
        n_users=8,
        n_items=8,
    )
    return movies, pre


def _build_fitted_components() -> tuple[SvdModel, ContentModel, NcfModel, PreprocessedData]:
    movies, pre = _hybrid_fixtures()

    svd = SvdModel(n_factors=2, seed=0)
    svd.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)

    content = ContentModel()
    content.fit(movies, pre)

    neural = NcfModel(n_factors=4, hidden=(8,), n_epochs=20, batch_size=8, seed=0)
    neural.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)

    return svd, content, neural, pre


def test_hybrid_recommend_returns_only_unseen_items() -> None:
    svd, content, neural, pre = _build_fitted_components()
    hybrid = HybridModel(
        collaborative=svd,
        content=content,
        neural=neural,
        n_items=pre.n_items,
        weights=(0.4, 0.3, 0.3),
        mmr_lambda=1.0,  # gate: λ=1 so we test fusion without MMR noise.
    )

    recs = hybrid.recommend(UserIdx(0), n=4)
    # User 0 rated items 0-3 → every rec must come from items 4-7.
    assert {int(i) for i, _ in recs}.issubset({4, 5, 6, 7})
    assert len(recs) == 4


def test_hybrid_recommend_rejects_non_positive_n() -> None:
    svd, content, neural, pre = _build_fitted_components()
    hybrid = HybridModel(
        collaborative=svd, content=content, neural=neural, n_items=pre.n_items
    )
    with pytest.raises(ValueError, match="n must"):
        hybrid.recommend(UserIdx(0), n=0)


def test_hybrid_recommend_rejects_weights_not_summing_to_one() -> None:
    svd, content, neural, pre = _build_fitted_components()
    with pytest.raises(ValueError, match="weights"):
        HybridModel(
            collaborative=svd,
            content=content,
            neural=neural,
            n_items=pre.n_items,
            weights=(0.5, 0.5, 0.5),  # sums to 1.5
        )


def test_hybrid_recommend_rejects_mmr_lambda_out_of_range() -> None:
    svd, content, neural, pre = _build_fitted_components()
    for bad_lambda in (-0.1, 1.1):
        with pytest.raises(ValueError, match="mmr_lambda"):
            HybridModel(
                collaborative=svd,
                content=content,
                neural=neural,
                n_items=pre.n_items,
                mmr_lambda=bad_lambda,
            )
