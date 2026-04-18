"""Shared fixtures for src.data tests.

The canonical ``tiny_ratings`` fixture is fully hand-specified so every step
of the pipeline (k-core, id maps, per-user split) has a predictable expected
output. See the Phase 1 design doc §5.1 for the full expected pipeline outcome.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _build_tiny_ratings() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    # concept: 5 "warm" users × 11 "hot" items — the only survivors under the
    # default k-core (k_user=11, k_item=5).
    for user_id in range(5):
        for movie_id in range(11):
            rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "rating": 1.0 + ((user_id + movie_id) % 5),
                    "timestamp": user_id * 1000 + movie_id,
                }
            )

    # concept: 4 users rating only 4 "cold" items — both axes fall below the
    # k-core thresholds, so every row here is expected to be dropped.
    for user_id in range(5, 9):
        for offset, movie_id in enumerate(range(11, 15)):
            rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "rating": 1.0 + ((user_id + movie_id) % 5),
                    "timestamp": user_id * 1000 + offset,
                }
            )

    # concept: user 9 rates only 4 hot items — dropped because 4 < k_user=11.
    for offset, movie_id in enumerate(range(4)):
        rows.append(
            {
                "userId": 9,
                "movieId": movie_id,
                "rating": 1.0 + ((9 + movie_id) % 5),
                "timestamp": 9000 + offset,
            }
        )

    return pd.DataFrame(rows).astype(
        {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    )


@pytest.fixture
def tiny_ratings() -> pd.DataFrame:
    """75 ratings across 10 users × 15 items with hand-predictable k-core / split outcomes."""
    return _build_tiny_ratings()
