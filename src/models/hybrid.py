"""Hybrid fusion — normalised weighted mix of CF + content + NCF, then MMR re-rank.

The hybrid is deliberately additive, not multiplicative. Each component model
produces a full per-item score vector for the target user; we min-max each
vector into ``[0, 1]`` to erase per-model scale differences, take a weighted
average, apply the ``exclude_seen`` mask, and pick the top ``pool_size``
candidates. Those candidates then go through MMR against the content-based
item embeddings so the final top-``n`` trades relevance for diversity as
controlled by ``mmr_lambda``.

``HybridModel`` intentionally wraps **already-fitted** component models rather
than fitting them itself — composing pre-trained models gives the caller full
control over component hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.data.ids import ItemIdx, UserIdx
from src.models.collaborative import AlsModel, SvdModel
from src.models.content_based import ContentModel
from src.models.neural_cf import NcfModel

FloatArray = npt.NDArray[np.float32]


@dataclass
class HybridModel:
    """Weighted score fusion across CF, content-based, and neural CF + MMR."""

    collaborative: SvdModel | AlsModel
    content: ContentModel
    neural: NcfModel
    n_items: int
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    mmr_lambda: float = 0.7
    pool_size: int = 50

    def __post_init__(self) -> None:
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError(
                f"weights must sum to 1.0, got {self.weights} (sum={sum(self.weights)})"
            )
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0, 1], got {self.mmr_lambda}")
        if self.pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {self.pool_size}")

    def recommend(
        self,
        user_idx: UserIdx,
        *,
        n: int,
        exclude_seen: bool = True,
    ) -> list[tuple[ItemIdx, float]]:
        """Top-``n`` items by fused score, re-ranked for diversity with MMR.

        Args:
            user_idx: Dense user index.
            n: Final number of items to return (``>= 1``).
            exclude_seen: Drop items the user has rated in any component's
                training set.

        Returns:
            List of ``(item_idx, fused_score)`` tuples of length ``≤ n``.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        cf_scores = _full_score_vector(
            self.collaborative.recommend, user_idx, self.n_items
        )
        content_scores = _full_score_vector(
            self.content.recommend, user_idx, self.n_items
        )
        neural_scores = _full_score_vector(
            self.neural.recommend, user_idx, self.n_items
        )

        fused = (
            self.weights[0] * min_max_normalise(cf_scores)
            + self.weights[1] * min_max_normalise(content_scores)
            + self.weights[2] * min_max_normalise(neural_scores)
        ).astype(np.float32)

        if exclude_seen:
            for item in self.collaborative._seen.get(user_idx, set()):
                fused[int(item)] = -np.inf

        pool = _top_pool(fused, pool_size=self.pool_size)
        if not pool:
            return []

        similarity = _content_similarity_matrix(
            content=self.content, items=[i for i, _ in pool]
        )
        return mmr_rerank(pool, similarity=similarity, n=n, mmr_lambda=self.mmr_lambda)


def min_max_normalise(scores: FloatArray) -> FloatArray:
    """Rescale a score vector into ``[0, 1]``; a flat vector maps to all-zero."""
    minimum = float(scores.min())
    maximum = float(scores.max())
    if maximum == minimum:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - minimum) / (maximum - minimum)).astype(np.float32)


def mmr_rerank(
    candidates: list[tuple[ItemIdx, float]],
    *,
    similarity: FloatArray,
    n: int,
    mmr_lambda: float,
) -> list[tuple[ItemIdx, float]]:
    """Re-rank ``candidates`` by Maximal Marginal Relevance against ``similarity``.

    Iteratively picks the candidate that maximises
    ``λ · relevance(i) − (1 − λ) · max_{j ∈ selected} sim(i, j)``.

    Args:
        candidates: Candidate ``(item_idx, relevance)`` pairs.
        similarity: Square matrix of shape ``(len(candidates), len(candidates))``
            holding pairwise item similarity in candidate-index order.
        n: Maximum number of items to emit.
        mmr_lambda: Relevance / diversity tradeoff — ``1.0`` keeps the relevance
            ranking; ``0.0`` fully maximises diversity.
    """
    if not candidates:
        return []

    relevances = np.array([rel for _, rel in candidates], dtype=np.float32)
    take = min(n, len(candidates))
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    for _ in range(take):
        best_idx = -1
        best_score = -np.inf
        for cand in remaining:
            if selected:
                penalty = float(max(similarity[cand, sel] for sel in selected))
            else:
                penalty = 0.0
            score = mmr_lambda * float(relevances[cand]) - (1.0 - mmr_lambda) * penalty
            if score > best_score:
                best_score = score
                best_idx = cand
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[idx] for idx in selected]


def _full_score_vector(
    recommend_fn: object, user_idx: UserIdx, n_items: int
) -> FloatArray:
    # concept: every model exposes recommend(); asking for n=n_items with
    # exclude_seen=False materialises the full score vector.
    from typing import Callable, cast

    fn = cast(Callable[..., list[tuple[ItemIdx, float]]], recommend_fn)
    ranked = fn(user_idx, n=n_items, exclude_seen=False)
    vec = np.zeros(n_items, dtype=np.float32)
    for item, score in ranked:
        vec[int(item)] = score
    return vec


def _top_pool(scores: FloatArray, *, pool_size: int) -> list[tuple[ItemIdx, float]]:
    finite = np.isfinite(scores)
    if not finite.any():
        return []
    finite_idx = np.flatnonzero(finite)
    take = min(pool_size, finite_idx.size)
    top = finite_idx[np.argsort(-scores[finite_idx], kind="stable")][:take]
    return [(ItemIdx(int(i)), float(scores[i])) for i in top]


def _content_similarity_matrix(
    *, content: ContentModel, items: list[ItemIdx]
) -> FloatArray:
    # concept: content._item_embeddings is L2-normalised, so cosine similarity
    # reduces to a sparse × sparse dot product.
    sub = content._item_embeddings[[int(i) for i in items]]
    similarity: FloatArray = (sub @ sub.T).toarray().astype(np.float32)
    return similarity
