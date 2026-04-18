"""Tests for src.models.content_based — TF-IDF + rating-weighted user profile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.ids import ItemIdx, PreprocessedData, RawMovieId, RawUserId, UserIdx
from src.models.content_based import ContentModel


def _tiny_movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [100, 200, 300],
            "title": ["Alpha", "Bravo", "Charlie"],
            "genres": ["Action", "Comedy", "Action|Comedy"],
        }
    )


def _build_preprocessed(
    train_rows: list[tuple[int, int, float]],
    *,
    n_users: int,
    n_items: int,
) -> PreprocessedData:
    item_map = {
        RawMovieId(100): ItemIdx(0),
        RawMovieId(200): ItemIdx(1),
        RawMovieId(300): ItemIdx(2),
    }
    user_map = {RawUserId(u): UserIdx(u) for u in range(n_users)}
    train = pd.DataFrame(
        {
            "user_idx": pd.array([r[0] for r in train_rows], dtype="int32"),
            "item_idx": pd.array([r[1] for r in train_rows], dtype="int32"),
            "rating": pd.array([r[2] for r in train_rows], dtype="float32"),
            "timestamp": pd.array([0] * len(train_rows), dtype="int64"),
        }
    )
    empty = train.iloc[:0].copy()
    return PreprocessedData(
        train=train,
        val=empty,
        test=empty,
        user_map=user_map,
        item_map=item_map,
        n_users=n_users,
        n_items=n_items,
    )


def test_recommend_ranks_by_shared_vocabulary_with_user_profile() -> None:
    # concept: user 0 rates only "Alpha" (Action) highly. "Charlie" shares the
    # action token, "Bravo" does not — Charlie must rank ahead of Bravo.
    pre = _build_preprocessed([(0, 0, 5.0)], n_users=1, n_items=3)

    model = ContentModel()
    model.fit(_tiny_movies(), pre)

    recs = model.recommend(UserIdx(0), n=3)
    item_ids = [int(i) for i, _ in recs]
    assert item_ids == [2, 1]  # item 0 is seen and excluded


def test_predict_returns_nonneg_cosine_similarity() -> None:
    pre = _build_preprocessed([(0, 0, 5.0)], n_users=1, n_items=3)
    model = ContentModel()
    model.fit(_tiny_movies(), pre)

    # concept: user profile == item 0's embedding ⇒ perfect cosine similarity.
    assert model.predict(UserIdx(0), ItemIdx(0)) == pytest.approx(1.0, abs=1e-4)
    # concept: item 1 shares no tokens with Alpha/Action → cosine 0.
    assert model.predict(UserIdx(0), ItemIdx(1)) == pytest.approx(0.0, abs=1e-4)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        ContentModel().predict(UserIdx(0), ItemIdx(0))


def test_recommend_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        ContentModel().recommend(UserIdx(0), n=3)


def test_recommend_rejects_non_positive_n() -> None:
    pre = _build_preprocessed([(0, 0, 5.0)], n_users=1, n_items=3)
    model = ContentModel()
    model.fit(_tiny_movies(), pre)
    with pytest.raises(ValueError, match="n must"):
        model.recommend(UserIdx(0), n=0)


def test_recommend_excludes_seen_items_by_default() -> None:
    pre = _build_preprocessed([(0, 0, 5.0), (0, 1, 3.0)], n_users=1, n_items=3)
    model = ContentModel()
    model.fit(_tiny_movies(), pre)

    recs = model.recommend(UserIdx(0), n=3)
    # Items 0 and 1 both seen → only item 2 remains.
    assert [int(i) for i, _ in recs] == [2]


def test_user_with_no_ratings_gets_zero_score_profile() -> None:
    # concept: user 1 has no training ratings ⇒ zero user profile, all cosines
    # are zero. We still return n items, just with score 0.
    pre = _build_preprocessed([(0, 0, 5.0)], n_users=2, n_items=3)
    model = ContentModel()
    model.fit(_tiny_movies(), pre)

    recs = model.recommend(UserIdx(1), n=3)
    assert len(recs) == 3
    for _, score in recs:
        assert score == 0.0


def test_fit_ignores_movies_absent_from_item_map() -> None:
    # concept: extra movie rows with IDs outside item_map must not widen the
    # embedding matrix — item_embeddings stays at n_items rows.
    pre = _build_preprocessed([(0, 0, 5.0)], n_users=1, n_items=3)
    extra = pd.concat(
        [
            _tiny_movies(),
            pd.DataFrame(
                {
                    "movieId": [999],
                    "title": ["Ghost"],
                    "genres": ["Drama"],
                }
            ),
        ],
        ignore_index=True,
    )

    model = ContentModel()
    model.fit(extra, pre)
    assert model._item_embeddings.shape[0] == 3


def test_fit_rejects_empty_train() -> None:
    empty = _build_preprocessed([], n_users=1, n_items=3)
    with pytest.raises(ValueError, match="at least one"):
        ContentModel().fit(_tiny_movies(), empty)


def test_recommendation_is_deterministic_for_equal_scores() -> None:
    # concept: if two items tie on cosine similarity the order must be stable
    # (ascending ItemIdx). Users of the model should never see reshuffled ties.
    pre = _build_preprocessed([(0, 0, 5.0)], n_users=1, n_items=3)
    model = ContentModel()
    model.fit(_tiny_movies(), pre)

    recs_a = model.recommend(UserIdx(0), n=3, exclude_seen=False)
    recs_b = model.recommend(UserIdx(0), n=3, exclude_seen=False)
    assert recs_a == recs_b
    assert np.all(np.diff([score for _, score in recs_a]) <= 1e-7)
