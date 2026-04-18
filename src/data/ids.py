"""Typed ID spaces and dataclass wrappers for the data pipeline.

MovieLens raw IDs (``RawUserId``, ``RawMovieId``) and the dense 0-based indices
used by matrix factorisation (``UserIdx``, ``ItemIdx``) are *different spaces*.
Keeping them distinct at the type level stops them silently aliasing — which
is the dominant class of bug in hand-rolled recommender code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

import pandas as pd

UserIdx = NewType("UserIdx", int)
ItemIdx = NewType("ItemIdx", int)
RawUserId = NewType("RawUserId", int)
RawMovieId = NewType("RawMovieId", int)


@dataclass(frozen=True, slots=True)
class RawData:
    """MovieLens ratings and movies, straight off disk.

    Attributes:
        ratings: Columns ``userId``, ``movieId``, ``rating``, ``timestamp``.
        movies: Columns ``movieId``, ``title``, ``genres``.
    """

    ratings: pd.DataFrame
    movies: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PreprocessedData:
    """Model-ready splits plus the id maps that produced them.

    Every downstream model (CF, content, neural CF) consumes this single object,
    so the id maps travel with the splits and can never drift out of sync.

    Attributes:
        train: Training interactions with columns ``user_idx``, ``item_idx``,
            ``rating``, ``timestamp``.
        val: Validation interactions, same schema as ``train``.
        test: Test interactions, same schema as ``train``.
        user_map: Bijection from raw MovieLens ``userId`` to dense ``UserIdx``.
        item_map: Bijection from raw MovieLens ``movieId`` to dense ``ItemIdx``.
        n_users: ``len(user_map)`` — size of the dense user index space.
        n_items: ``len(item_map)`` — size of the dense item index space.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    user_map: dict[RawUserId, UserIdx]
    item_map: dict[RawMovieId, ItemIdx]
    n_users: int
    n_items: int
