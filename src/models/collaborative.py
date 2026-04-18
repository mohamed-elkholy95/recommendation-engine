"""Collaborative filtering — truncated-SVD baseline on the explicit rating matrix.

The model factorises the user × item rating matrix as ``R ≈ U · diag(s) · Vᵀ``
with a rank ``n_factors``, then predicts unseen entries from the reconstruction.

Mean-centering is deliberately omitted: raw SVD keeps the math inspectable and
the tests hand-computable. This is the conceptual baseline before the ALS and
NCF layers land — it is expected to underperform them on ranking metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from src.data.ids import ItemIdx, UserIdx
from src.utils.seed import set_all_seeds


@dataclass
class SvdModel:
    """Truncated-SVD collaborative filter for explicit ratings.

    Attributes:
        n_factors: Rank of the SVD approximation.
        seed: Seed fed to ``set_all_seeds`` before each ``fit`` call so the
            ARPACK / LOBPCG iteration starts from a reproducible state.
    """

    n_factors: int = 50
    seed: int = 42

    _user_factors: npt.NDArray[np.float32] = field(
        init=False, default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    _item_factors: npt.NDArray[np.float32] = field(
        init=False, default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    _seen: dict[UserIdx, set[ItemIdx]] = field(init=False, default_factory=dict)
    _fitted: bool = field(init=False, default=False)

    def fit(self, train: pd.DataFrame, *, n_users: int, n_items: int) -> None:
        """Compute the rank-``n_factors`` SVD of the sparse rating matrix.

        Folds the singular values into the left factor so ``predict`` becomes a
        plain dot product between the user row and the item row.

        Args:
            train: Columns ``user_idx``, ``item_idx``, ``rating`` (dtypes from
                the preprocessor).
            n_users: Size of the dense user index space.
            n_items: Size of the dense item index space.

        Raises:
            ValueError: If ``train`` is empty, or ``n_factors`` is not strictly
                less than ``min(n_users, n_items)``.
        """
        if len(train) == 0:
            raise ValueError("train must contain at least one rating")
        if self.n_factors >= min(n_users, n_items):
            raise ValueError(
                f"n_factors ({self.n_factors}) must be < min(n_users, n_items) = "
                f"{min(n_users, n_items)}"
            )

        set_all_seeds(self.seed)
        matrix = _build_sparse_matrix(train, n_users=n_users, n_items=n_items)

        # concept: svds returns singular values ascending — flip both factors
        # so position 0 holds the largest component.
        left, singular, right_t = svds(matrix, k=self.n_factors)
        self._user_factors = (left[:, ::-1] * singular[::-1]).astype(np.float32)
        self._item_factors = right_t[::-1, :].T.astype(np.float32)
        self._seen = _build_seen_dict(train)
        self._fitted = True

    def predict(self, user_idx: UserIdx, item_idx: ItemIdx) -> float:
        """Reconstructed rating for a single (user, item) pair."""
        self._check_fitted()
        score = self._user_factors[int(user_idx)] @ self._item_factors[int(item_idx)]
        return float(score)

    def recommend(
        self,
        user_idx: UserIdx,
        *,
        n: int,
        exclude_seen: bool = True,
    ) -> list[tuple[ItemIdx, float]]:
        """Top-``n`` items for a user, sorted by descending reconstructed score.

        Args:
            user_idx: Dense user index.
            n: Maximum number of items to return (``>= 1``).
            exclude_seen: Drop items the user rated in training. Items masked
                out never re-enter the ranking, so the result may be shorter
                than ``n`` if the user has seen most of the catalogue.

        Returns:
            List of ``(item_idx, score)`` tuples, longest-first.

        Raises:
            ValueError: If ``n < 1``.
            RuntimeError: If ``fit`` has not been called.
        """
        self._check_fitted()
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        scores: npt.NDArray[np.float32] = (
            self._user_factors[int(user_idx)] @ self._item_factors.T
        )
        if exclude_seen:
            for seen in self._seen.get(user_idx, set()):
                scores[int(seen)] = -np.inf

        finite = np.isfinite(scores)
        if not finite.any():
            return []

        finite_idx = np.flatnonzero(finite)
        take = min(n, finite_idx.size)
        top_among_finite = finite_idx[np.argsort(-scores[finite_idx])][:take]
        return [(ItemIdx(int(i)), float(scores[i])) for i in top_among_finite]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SvdModel.fit() must be called before predict/recommend")


def _build_sparse_matrix(
    train: pd.DataFrame, *, n_users: int, n_items: int
) -> csr_matrix:
    return csr_matrix(
        (
            train["rating"].to_numpy(dtype=np.float32),
            (train["user_idx"].to_numpy(), train["item_idx"].to_numpy()),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    )


def _build_seen_dict(train: pd.DataFrame) -> dict[UserIdx, set[ItemIdx]]:
    seen: dict[UserIdx, set[ItemIdx]] = {}
    for user_raw, item_raw in zip(train["user_idx"], train["item_idx"], strict=True):
        seen.setdefault(UserIdx(int(user_raw)), set()).add(ItemIdx(int(item_raw)))
    return seen
