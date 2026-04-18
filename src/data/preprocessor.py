"""k-core filter, id-map construction, and per-user temporal split.

The pipeline is a sequence of pure functions, each testable on its own:

    _k_core_filter → _build_id_maps → _apply_id_maps → _per_user_split

``preprocess`` is the single public entry point. It chooses ``k_user`` high
enough that every surviving user has at least ``2 * max(n_test, n_val) + 1``
ratings, so leave-last-n-out is guaranteed to produce a non-empty train split.
"""

from __future__ import annotations

import pandas as pd

from src.data.ids import (
    ItemIdx,
    PreprocessedData,
    RawData,
    RawMovieId,
    RawUserId,
    UserIdx,
)


def preprocess(
    raw: RawData,
    *,
    k: int = 5,
    n_test: int = 5,
    n_val: int = 5,
) -> PreprocessedData:
    """Turn ``RawData`` into model-ready train / val / test splits and id maps.

    Applies k-core filtering with a user threshold tall enough to guarantee a
    clean per-user leave-last-n-out split, then builds dense id maps and splits
    each user's ratings by timestamp into train / val / test.

    Args:
        raw: MovieLens ratings + movies as returned by ``load_movielens``.
        k: Minimum ratings per item (and a floor on per-user ratings).
        n_test: Number of most-recent ratings per user to hold out for test.
        n_val: Number of ratings per user to hold out for validation, taken
            from the window immediately preceding the test block.

    Returns:
        A ``PreprocessedData`` whose splits use dense ``UserIdx`` / ``ItemIdx``.

    Raises:
        ValueError: If ``k``, ``n_test``, or ``n_val`` is below 1, or if the
            k-core filter would leave zero users.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n_test < 1:
        raise ValueError(f"n_test must be >= 1, got {n_test}")
    if n_val < 1:
        raise ValueError(f"n_val must be >= 1, got {n_val}")

    # gate: unified k-core — every surviving user has enough ratings to split.
    k_user = max(k, 2 * max(n_test, n_val) + 1)
    filtered = _k_core_filter(raw.ratings, k_user=k_user, k_item=k)

    user_map, item_map = _build_id_maps(filtered)
    indexed = _apply_id_maps(filtered, user_map, item_map)
    train, val, test = _per_user_split(indexed, n_test=n_test, n_val=n_val)

    return PreprocessedData(
        train=train,
        val=val,
        test=test,
        user_map=user_map,
        item_map=item_map,
        n_users=len(user_map),
        n_items=len(item_map),
    )


def _k_core_filter(ratings: pd.DataFrame, *, k_user: int, k_item: int) -> pd.DataFrame:
    # concept: iterate user-drop + item-drop until a full pass removes nothing.
    current = ratings.copy()
    while True:
        before = len(current)
        user_counts = current.groupby("userId")["movieId"].count()
        keep_users = user_counts[user_counts >= k_user].index
        current = current[current["userId"].isin(keep_users)]

        item_counts = current.groupby("movieId")["userId"].count()
        keep_items = item_counts[item_counts >= k_item].index
        current = current[current["movieId"].isin(keep_items)]

        if len(current) == before:
            break

    if current["userId"].nunique() == 0:
        raise ValueError("k-core filter left 0 users — loosen k or n thresholds")

    return current.reset_index(drop=True)


def _build_id_maps(
    ratings: pd.DataFrame,
) -> tuple[dict[RawUserId, UserIdx], dict[RawMovieId, ItemIdx]]:
    # concept: sorted enumeration — same input frame ⇒ same dense indices.
    user_map = {
        RawUserId(int(raw_id)): UserIdx(idx)
        for idx, raw_id in enumerate(sorted(ratings["userId"].unique()))
    }
    item_map = {
        RawMovieId(int(raw_id)): ItemIdx(idx)
        for idx, raw_id in enumerate(sorted(ratings["movieId"].unique()))
    }
    return user_map, item_map


def _apply_id_maps(
    ratings: pd.DataFrame,
    user_map: dict[RawUserId, UserIdx],
    item_map: dict[RawMovieId, ItemIdx],
) -> pd.DataFrame:
    indexed = pd.DataFrame(
        {
            "user_idx": ratings["userId"].map(user_map).astype("int32"),
            "item_idx": ratings["movieId"].map(item_map).astype("int32"),
            "rating": ratings["rating"].astype("float32"),
            "timestamp": ratings["timestamp"].astype("int64"),
        }
    )
    return indexed.reset_index(drop=True)


def _per_user_split(
    indexed: pd.DataFrame,
    *,
    n_test: int,
    n_val: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # concept: rank each user's rows by timestamp, slice by rank from the top.
    ordered = indexed.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)
    reverse_rank = ordered.groupby("user_idx").cumcount(ascending=False)

    test_mask = reverse_rank < n_test
    val_mask = (reverse_rank >= n_test) & (reverse_rank < n_test + n_val)
    train_mask = reverse_rank >= n_test + n_val

    return (
        ordered[train_mask].reset_index(drop=True),
        ordered[val_mask].reset_index(drop=True),
        ordered[test_mask].reset_index(drop=True),
    )
