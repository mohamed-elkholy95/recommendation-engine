"""Tests for src.data.ids — typed ID spaces and the two dataclass wrappers."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from src.data.ids import (
    ItemIdx,
    PreprocessedData,
    RawData,
    RawMovieId,
    RawUserId,
    UserIdx,
)


def test_newtype_round_trips_through_int() -> None:
    # concept: NewType is a zero-cost alias — int(UserIdx(3)) must equal 3.
    assert int(UserIdx(3)) == 3
    assert int(ItemIdx(7)) == 7
    assert int(RawUserId(101)) == 101
    assert int(RawMovieId(202)) == 202


def test_raw_data_is_frozen() -> None:
    ratings = pd.DataFrame(
        {"userId": [1], "movieId": [10], "rating": [4.0], "timestamp": [123]}
    )
    movies = pd.DataFrame({"movieId": [10], "title": ["t"], "genres": ["g"]})
    raw = RawData(ratings=ratings, movies=movies)

    with pytest.raises(dataclasses.FrozenInstanceError):
        raw.ratings = movies  # type: ignore[misc]


def test_preprocessed_data_is_frozen() -> None:
    empty = pd.DataFrame(
        {"user_idx": [], "item_idx": [], "rating": [], "timestamp": []}
    )
    pre = PreprocessedData(
        train=empty,
        val=empty,
        test=empty,
        user_map={RawUserId(1): UserIdx(0)},
        item_map={RawMovieId(10): ItemIdx(0)},
        n_users=1,
        n_items=1,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        pre.n_users = 2  # type: ignore[misc]


def test_preprocessed_data_holds_expected_fields() -> None:
    empty = pd.DataFrame(
        {"user_idx": [], "item_idx": [], "rating": [], "timestamp": []}
    )
    user_map = {RawUserId(5): UserIdx(0), RawUserId(9): UserIdx(1)}
    item_map = {RawMovieId(100): ItemIdx(0)}
    pre = PreprocessedData(
        train=empty,
        val=empty,
        test=empty,
        user_map=user_map,
        item_map=item_map,
        n_users=2,
        n_items=1,
    )

    assert pre.n_users == 2
    assert pre.n_items == 1
    assert pre.user_map[RawUserId(9)] == UserIdx(1)
    assert pre.item_map[RawMovieId(100)] == ItemIdx(0)
