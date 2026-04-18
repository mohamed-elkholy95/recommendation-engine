"""Neural collaborative filtering — a two-tower MLP trained on implicit feedback.

Ratings above ``positive_threshold`` are treated as positive interactions; every
positive pair is paired with ``negatives_per_positive`` items sampled uniformly
from items the user has not seen. The model concatenates per-user and per-item
embeddings and passes them through an MLP; the output is a BCE-trained logit.

Unlike the SVD / ALS baselines this model can fit non-linear interactions
between users and items, at the cost of a training loop and a framework dep.
ONNX export is a serving concern and is deferred to the deployment phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn

from src.data.ids import ItemIdx, UserIdx
from src.models._ranking import (
    build_seen_dict,
    require_fitted,
    require_non_empty_train,
    top_k_from_scores,
)
from src.utils.seed import set_all_seeds

FloatArray = npt.NDArray[np.float32]


class _TwoTower(nn.Module):  # type: ignore[misc]  # torch not in pre-commit's mypy env
    """User + item embeddings concatenated, passed through an MLP head."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)

        layers: list[nn.Module] = []
        in_dim = 2 * n_factors
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, user_idx: torch.Tensor, item_idx: torch.Tensor
    ) -> torch.Tensor:
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        x = torch.cat([u, i], dim=-1)
        logits: torch.Tensor = self.mlp(x).squeeze(-1)
        return logits


@dataclass
class NcfModel:
    """Two-tower MLP recommender trained with BCE on sampled negatives."""

    n_factors: int = 32
    hidden: tuple[int, ...] = (64, 32, 16)
    n_epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    negatives_per_positive: int = 4
    positive_threshold: float = 4.0
    seed: int = 42

    _net: nn.Module | None = field(init=False, default=None)
    _n_items: int = field(init=False, default=0)
    _seen: dict[UserIdx, set[ItemIdx]] = field(init=False, default_factory=dict)
    _fitted: bool = field(init=False, default=False)

    def fit(self, train: pd.DataFrame, *, n_users: int, n_items: int) -> None:
        """Train the two-tower MLP on rating-derived positives + sampled negatives.

        Raises:
            ValueError: If ``train`` is empty, hyperparameters are non-positive,
                or no rating in ``train`` clears ``positive_threshold``.
        """
        require_non_empty_train(train)
        self._validate_hyperparams()

        set_all_seeds(self.seed)
        positives = train[train["rating"] >= self.positive_threshold]
        if len(positives) == 0:
            raise ValueError(
                f"no positive interactions with rating >= {self.positive_threshold}"
            )

        seen = build_seen_dict(train)
        pos_users = positives["user_idx"].to_numpy(dtype=np.int64)
        pos_items = positives["item_idx"].to_numpy(dtype=np.int64)

        net = _TwoTower(n_users, n_items, self.n_factors, tuple(self.hidden))
        optimiser = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_epochs):
            neg_users, neg_items = _sample_negatives(
                pos_users=pos_users,
                seen=seen,
                n_items=n_items,
                negatives_per_positive=self.negatives_per_positive,
                rng=rng,
            )
            users = np.concatenate([pos_users, neg_users])
            items = np.concatenate([pos_items, neg_items])
            labels = np.concatenate(
                [np.ones(len(pos_users), np.float32), np.zeros(len(neg_users), np.float32)]
            )

            perm = rng.permutation(len(users))
            users, items, labels = users[perm], items[perm], labels[perm]

            net.train()
            for start in range(0, len(users), self.batch_size):
                end = start + self.batch_size
                u_batch = torch.from_numpy(users[start:end])
                i_batch = torch.from_numpy(items[start:end])
                y_batch = torch.from_numpy(labels[start:end])

                optimiser.zero_grad()
                logits = net(u_batch, i_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimiser.step()

        self._net = net
        self._n_items = n_items
        self._seen = seen
        self._fitted = True

    def predict(self, user_idx: UserIdx, item_idx: ItemIdx) -> float:
        """Sigmoid-squashed interaction probability for a (user, item) pair."""
        require_fitted(self._fitted, "NcfModel")
        assert self._net is not None
        self._net.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([int(user_idx)], dtype=torch.long)
            item_tensor = torch.tensor([int(item_idx)], dtype=torch.long)
            logit = self._net(user_tensor, item_tensor)
            return float(torch.sigmoid(logit).item())

    def recommend(
        self,
        user_idx: UserIdx,
        *,
        n: int,
        exclude_seen: bool = True,
    ) -> list[tuple[ItemIdx, float]]:
        """Top-``n`` items ranked by predicted interaction probability."""
        require_fitted(self._fitted, "NcfModel")
        assert self._net is not None
        self._net.eval()
        with torch.no_grad():
            user_tensor = torch.full(
                (self._n_items,), int(user_idx), dtype=torch.long
            )
            item_tensor = torch.arange(self._n_items, dtype=torch.long)
            logits = self._net(user_tensor, item_tensor)
            scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        return top_k_from_scores(
            scores,
            seen=self._seen.get(user_idx, set()),
            n=n,
            exclude_seen=exclude_seen,
        )

    def _validate_hyperparams(self) -> None:
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.negatives_per_positive < 1:
            raise ValueError(
                f"negatives_per_positive must be >= 1, got {self.negatives_per_positive}"
            )
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")


def _sample_negatives(
    *,
    pos_users: npt.NDArray[np.int64],
    seen: dict[UserIdx, set[ItemIdx]],
    n_items: int,
    negatives_per_positive: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    # concept: for each positive (u, +) draw k items uniformly at random and
    # reject if the user has already seen them. Loop until the quota is filled
    # — fine for any reasonable catalogue density.
    neg_users = np.repeat(pos_users, negatives_per_positive)
    neg_items = np.empty_like(neg_users)
    for slot, user in enumerate(neg_users):
        seen_set = seen.get(UserIdx(int(user)), set())
        while True:
            candidate = int(rng.integers(0, n_items))
            if ItemIdx(candidate) not in seen_set:
                neg_items[slot] = candidate
                break
    return neg_users, neg_items
