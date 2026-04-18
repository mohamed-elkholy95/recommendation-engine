"""Shared ranking helpers used by every model's ``recommend`` / ``fit`` path.

Private to ``src/models/`` — nothing outside the package should import these.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data.ids import ItemIdx, UserIdx

FloatArray = npt.NDArray[np.floating]


def top_k_from_scores(
    scores: FloatArray,
    *,
    seen: set[ItemIdx],
    n: int,
    exclude_seen: bool,
) -> list[tuple[ItemIdx, float]]:
    """Return the top-``n`` indices of ``scores`` sorted by descending value.

    Seen items are masked out (score set to ``-inf``) before selection when
    ``exclude_seen`` is ``True``. Stable sort guarantees deterministic ordering
    when two candidates tie on score — ties resolve by ascending index.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    scores = scores.copy()
    if exclude_seen:
        for item in seen:
            scores[int(item)] = -np.inf

    finite = np.isfinite(scores)
    if not finite.any():
        return []

    finite_idx = np.flatnonzero(finite)
    take = min(n, finite_idx.size)
    top = finite_idx[np.argsort(-scores[finite_idx], kind="stable")][:take]
    return [(ItemIdx(int(i)), float(scores[i])) for i in top]


def require_fitted(fitted: bool, model_name: str) -> None:
    if not fitted:
        raise RuntimeError(f"{model_name}.fit() must be called before predict/recommend")


def require_non_empty_train(train: pd.DataFrame) -> None:
    if len(train) == 0:
        raise ValueError("train must contain at least one rating")


def build_seen_dict(train: pd.DataFrame) -> dict[UserIdx, set[ItemIdx]]:
    seen: dict[UserIdx, set[ItemIdx]] = {}
    for user_raw, item_raw in zip(train["user_idx"], train["item_idx"], strict=True):
        seen.setdefault(UserIdx(int(user_raw)), set()).add(ItemIdx(int(item_raw)))
    return seen
