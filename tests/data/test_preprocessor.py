"""Tests for src.data.preprocessor — k-core, id maps, per-user temporal split."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.ids import ItemIdx, RawMovieId, RawUserId, UserIdx
from src.data.preprocessor import (
    _apply_id_maps,
    _build_id_maps,
    _k_core_filter,
    _per_user_split,
    preprocess,
)


# -----------------------------------------------------------------------------
# _k_core_filter
# -----------------------------------------------------------------------------


def test_k_core_drops_users_and_items_below_thresholds(tiny_ratings: pd.DataFrame) -> None:
    # gate: defaults k=5, n_test=5, n_val=5 ⇒ k_user=11, k_item=5.
    filtered = _k_core_filter(tiny_ratings, k_user=11, k_item=5)

    assert sorted(filtered["userId"].unique().tolist()) == [0, 1, 2, 3, 4]
    assert sorted(filtered["movieId"].unique().tolist()) == list(range(11))
    assert len(filtered) == 55


def test_k_core_is_idempotent_on_converged_input(tiny_ratings: pd.DataFrame) -> None:
    once = _k_core_filter(tiny_ratings, k_user=11, k_item=5)
    twice = _k_core_filter(once, k_user=11, k_item=5)

    pd.testing.assert_frame_equal(
        once.reset_index(drop=True), twice.reset_index(drop=True)
    )


def test_k_core_empties_dataset_raises() -> None:
    sparse = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 11, 10, 11],
            "rating": [4.0, 3.0, 5.0, 2.0],
            "timestamp": [0, 1, 2, 3],
        }
    )

    with pytest.raises(ValueError, match="0 users"):
        _k_core_filter(sparse, k_user=5, k_item=5)


# -----------------------------------------------------------------------------
# _build_id_maps
# -----------------------------------------------------------------------------


def test_id_maps_are_bijective_and_sorted(tiny_ratings: pd.DataFrame) -> None:
    filtered = _k_core_filter(tiny_ratings, k_user=11, k_item=5)
    user_map, item_map = _build_id_maps(filtered)

    assert user_map == {RawUserId(i): UserIdx(i) for i in range(5)}
    assert item_map == {RawMovieId(i): ItemIdx(i) for i in range(11)}


def test_id_maps_assign_dense_indices_to_non_contiguous_raw_ids() -> None:
    # concept: raw IDs may be sparse; dense indices must be 0..N-1 in sorted order.
    frame = pd.DataFrame(
        {
            "userId": [17, 3, 17, 3, 9, 9],
            "movieId": [200, 100, 100, 200, 100, 200],
            "rating": [4.0] * 6,
            "timestamp": [0] * 6,
        }
    )
    user_map, item_map = _build_id_maps(frame)

    assert user_map == {
        RawUserId(3): UserIdx(0),
        RawUserId(9): UserIdx(1),
        RawUserId(17): UserIdx(2),
    }
    assert item_map == {RawMovieId(100): ItemIdx(0), RawMovieId(200): ItemIdx(1)}


# -----------------------------------------------------------------------------
# _apply_id_maps
# -----------------------------------------------------------------------------


def test_apply_id_maps_replaces_raw_ids() -> None:
    frame = pd.DataFrame(
        {
            "userId": [17, 3, 9],
            "movieId": [200, 100, 200],
            "rating": [4.0, 2.5, 3.5],
            "timestamp": [10, 20, 30],
        }
    )
    user_map = {RawUserId(3): UserIdx(0), RawUserId(9): UserIdx(1), RawUserId(17): UserIdx(2)}
    item_map = {RawMovieId(100): ItemIdx(0), RawMovieId(200): ItemIdx(1)}

    indexed = _apply_id_maps(frame, user_map, item_map)

    expected = pd.DataFrame(
        {
            "user_idx": [2, 0, 1],
            "item_idx": [1, 0, 1],
            "rating": [4.0, 2.5, 3.5],
            "timestamp": [10, 20, 30],
        }
    )
    pd.testing.assert_frame_equal(
        indexed.reset_index(drop=True),
        expected.astype(indexed.dtypes.to_dict()),
    )


# -----------------------------------------------------------------------------
# _per_user_split
# -----------------------------------------------------------------------------


def test_per_user_split_boundaries() -> None:
    # concept: one user, 11 ratings with timestamps 0..10 ⇒ test=[6..10], val=[1..5], train=[0].
    indexed = pd.DataFrame(
        {
            "user_idx": [0] * 11,
            "item_idx": list(range(11)),
            "rating": [3.0] * 11,
            "timestamp": list(range(11)),
        }
    )

    train, val, test = _per_user_split(indexed, n_test=5, n_val=5)

    assert sorted(test["timestamp"].tolist()) == [6, 7, 8, 9, 10]
    assert sorted(val["timestamp"].tolist()) == [1, 2, 3, 4, 5]
    assert sorted(train["timestamp"].tolist()) == [0]


def test_per_user_split_is_timestamp_ordered_not_row_ordered() -> None:
    # concept: the split must sort by timestamp per user, not respect input row order.
    indexed = pd.DataFrame(
        {
            "user_idx": [0] * 7,
            "item_idx": [5, 2, 1, 4, 0, 6, 3],
            "rating": [3.0] * 7,
            "timestamp": [50, 20, 10, 40, 0, 60, 30],
        }
    )

    train, val, test = _per_user_split(indexed, n_test=2, n_val=2)

    assert sorted(test["timestamp"].tolist()) == [50, 60]
    assert sorted(val["timestamp"].tolist()) == [30, 40]
    assert sorted(train["timestamp"].tolist()) == [0, 10, 20]


# -----------------------------------------------------------------------------
# preprocess — end to end
# -----------------------------------------------------------------------------


def test_preprocess_end_to_end_tiny(tiny_ratings: pd.DataFrame) -> None:
    from src.data.ids import RawData

    raw = RawData(ratings=tiny_ratings, movies=pd.DataFrame(columns=["movieId", "title", "genres"]))

    pre = preprocess(raw, k=5, n_test=5, n_val=5)

    assert pre.n_users == 5
    assert pre.n_items == 11
    # gate: per-user leave-last-n-out — 11 ratings ⇒ train=1, val=5, test=5 per user.
    assert len(pre.train) == 5
    assert len(pre.val) == 25
    assert len(pre.test) == 25
    assert pre.user_map == {RawUserId(i): UserIdx(i) for i in range(5)}
    assert pre.item_map == {RawMovieId(i): ItemIdx(i) for i in range(11)}


def test_preprocess_has_no_future_leakage_per_user(tiny_ratings: pd.DataFrame) -> None:
    from src.data.ids import RawData

    raw = RawData(ratings=tiny_ratings, movies=pd.DataFrame(columns=["movieId", "title", "genres"]))
    pre = preprocess(raw, k=5, n_test=5, n_val=5)

    for user_idx in range(pre.n_users):
        user_train = pre.train[pre.train["user_idx"] == user_idx]["timestamp"]
        user_val = pre.val[pre.val["user_idx"] == user_idx]["timestamp"]
        user_test = pre.test[pre.test["user_idx"] == user_idx]["timestamp"]
        if not user_train.empty and not user_val.empty:
            assert user_train.max() < user_val.min()
        if not user_val.empty and not user_test.empty:
            assert user_val.max() < user_test.min()


@pytest.mark.parametrize(
    "k,n_test,n_val",
    [(0, 5, 5), (5, 0, 5), (5, 5, 0), (-1, 5, 5)],
)
def test_preprocess_rejects_non_positive_params(
    tiny_ratings: pd.DataFrame, k: int, n_test: int, n_val: int
) -> None:
    from src.data.ids import RawData

    raw = RawData(ratings=tiny_ratings, movies=pd.DataFrame(columns=["movieId", "title", "genres"]))

    with pytest.raises(ValueError):
        preprocess(raw, k=k, n_test=n_test, n_val=n_val)
