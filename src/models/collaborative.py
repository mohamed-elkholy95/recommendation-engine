"""Collaborative filtering — truncated-SVD and Koren-ALS baselines.

Both models factorise the user × item rating matrix into dense ``user_factors``
and ``item_factors``. Once the factors are in hand, ``predict`` and
``recommend`` behave identically for the two algorithms, so those methods
share the module-level ``_predict_from_factors`` / ``_recommend_from_factors``
helpers below.

Mean-centering is deliberately omitted: raw factorisation keeps the math
inspectable and the tests hand-computable. These are the conceptual
baselines before Neural CF lands — they are expected to underperform on
ranking metrics.
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

FloatArray = npt.NDArray[np.float32]


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

    _user_factors: FloatArray = field(
        init=False, default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    _item_factors: FloatArray = field(
        init=False, default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    _seen: dict[UserIdx, set[ItemIdx]] = field(init=False, default_factory=dict)
    _fitted: bool = field(init=False, default=False)

    def fit(self, train: pd.DataFrame, *, n_users: int, n_items: int) -> None:
        """Compute the rank-``n_factors`` SVD of the sparse rating matrix.

        Folds the singular values into the left factor so ``predict`` becomes a
        plain dot product between the user row and the item row.

        Raises:
            ValueError: If ``train`` is empty, or ``n_factors`` is not strictly
                less than ``min(n_users, n_items)``.
        """
        _require_non_empty(train)
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
        _require_fitted(self._fitted, "SvdModel")
        return _predict_from_factors(
            self._user_factors, self._item_factors, user_idx, item_idx
        )

    def recommend(
        self,
        user_idx: UserIdx,
        *,
        n: int,
        exclude_seen: bool = True,
    ) -> list[tuple[ItemIdx, float]]:
        """Top-``n`` items for a user, sorted by descending reconstructed score."""
        _require_fitted(self._fitted, "SvdModel")
        return _recommend_from_factors(
            self._user_factors,
            self._item_factors,
            user_idx=user_idx,
            seen=self._seen.get(user_idx, set()),
            n=n,
            exclude_seen=exclude_seen,
        )


@dataclass
class AlsModel:
    """Koren alternating least squares on observed explicit ratings.

    At each iteration the user factors are solved per-user against the current
    item factors (and vice-versa) over only the ``(u, i)`` pairs actually
    observed in training, with ``reg * I`` added to the normal-equation
    matrix for Tikhonov regularisation.

    Attributes:
        n_factors: Dimensionality of the latent space.
        n_iter: Number of full ALS sweeps (one X update + one Y update).
        reg: Regularisation strength ``λ`` — larger values shrink factors.
        seed: Seed for ``set_all_seeds`` and the per-fit random init.
    """

    n_factors: int = 50
    n_iter: int = 15
    reg: float = 0.1
    seed: int = 42

    _user_factors: FloatArray = field(
        init=False, default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    _item_factors: FloatArray = field(
        init=False, default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    _seen: dict[UserIdx, set[ItemIdx]] = field(init=False, default_factory=dict)
    _fitted: bool = field(init=False, default=False)

    def fit(self, train: pd.DataFrame, *, n_users: int, n_items: int) -> None:
        """Alternate per-user / per-item least-squares solves ``n_iter`` times.

        Raises:
            ValueError: If ``train`` is empty, ``n_iter < 1``, or ``reg < 0``.
        """
        _require_non_empty(train)
        if self.n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {self.n_iter}")
        if self.reg < 0:
            raise ValueError(f"reg must be >= 0, got {self.reg}")

        set_all_seeds(self.seed)
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / float(np.sqrt(self.n_factors))
        user_factors = (
            rng.standard_normal((n_users, self.n_factors)).astype(np.float32) * scale
        )
        item_factors = (
            rng.standard_normal((n_items, self.n_factors)).astype(np.float32) * scale
        )

        matrix_csr = _build_sparse_matrix(train, n_users=n_users, n_items=n_items)
        matrix_csc = matrix_csr.tocsc()
        reg_eye = (self.reg * np.eye(self.n_factors)).astype(np.float32)

        for _ in range(self.n_iter):
            _als_sweep(user_factors, item_factors, matrix_csr, reg_eye)
            _als_sweep(item_factors, user_factors, matrix_csc.T.tocsr(), reg_eye)

        self._user_factors = user_factors
        self._item_factors = item_factors
        self._seen = _build_seen_dict(train)
        self._fitted = True

    def predict(self, user_idx: UserIdx, item_idx: ItemIdx) -> float:
        _require_fitted(self._fitted, "AlsModel")
        return _predict_from_factors(
            self._user_factors, self._item_factors, user_idx, item_idx
        )

    def recommend(
        self,
        user_idx: UserIdx,
        *,
        n: int,
        exclude_seen: bool = True,
    ) -> list[tuple[ItemIdx, float]]:
        _require_fitted(self._fitted, "AlsModel")
        return _recommend_from_factors(
            self._user_factors,
            self._item_factors,
            user_idx=user_idx,
            seen=self._seen.get(user_idx, set()),
            n=n,
            exclude_seen=exclude_seen,
        )


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _als_sweep(
    left: FloatArray,
    right: FloatArray,
    observations_by_left: csr_matrix,
    reg_eye: FloatArray,
) -> None:
    # concept: one side of a Koren ALS step — for each row u of `left`, solve
    # (right_Ou.T @ right_Ou + λI) · x = right_Ou.T · r_u over the observed
    # pairs Ou. Writes results back into `left` in-place.
    indptr = observations_by_left.indptr
    indices = observations_by_left.indices
    data = observations_by_left.data
    for row in range(left.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        if start == end:
            continue
        right_rows = right[indices[start:end]]
        ratings = data[start:end]
        normal = right_rows.T @ right_rows + reg_eye
        rhs = right_rows.T @ ratings
        left[row] = np.linalg.solve(normal, rhs).astype(np.float32)


def _predict_from_factors(
    user_factors: FloatArray,
    item_factors: FloatArray,
    user_idx: UserIdx,
    item_idx: ItemIdx,
) -> float:
    return float(user_factors[int(user_idx)] @ item_factors[int(item_idx)])


def _recommend_from_factors(
    user_factors: FloatArray,
    item_factors: FloatArray,
    *,
    user_idx: UserIdx,
    seen: set[ItemIdx],
    n: int,
    exclude_seen: bool,
) -> list[tuple[ItemIdx, float]]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    scores: FloatArray = user_factors[int(user_idx)] @ item_factors.T
    if exclude_seen:
        for item in seen:
            scores[int(item)] = -np.inf

    finite = np.isfinite(scores)
    if not finite.any():
        return []
    finite_idx = np.flatnonzero(finite)
    take = min(n, finite_idx.size)
    top = finite_idx[np.argsort(-scores[finite_idx])][:take]
    return [(ItemIdx(int(i)), float(scores[i])) for i in top]


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


def _require_non_empty(train: pd.DataFrame) -> None:
    if len(train) == 0:
        raise ValueError("train must contain at least one rating")


def _require_fitted(fitted: bool, model_name: str) -> None:
    if not fitted:
        raise RuntimeError(f"{model_name}.fit() must be called before predict/recommend")
