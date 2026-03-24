"""Neural Collaborative Filtering model (PyTorch).

Neural Collaborative Filtering (NCF) replaces the inner product of
traditional matrix factorization with a learned neural network, allowing
it to capture non-linear user–item interaction patterns.

Architecture overview (He et al., WWW 2017):
    1. **GMF branch** — Generalized Matrix Factorization: element-wise product
       of user/item embeddings, analogous to classical MF but with learned weights.
    2. **MLP branch** — Multi-Layer Perceptron: concatenated embeddings passed
       through hidden layers to learn arbitrary interaction functions.
    3. **NeuMF** — Concatenation of GMF and MLP outputs → final sigmoid prediction.

Training uses Binary Cross-Entropy (BCE) loss with implicit feedback:
ratings ≥ 3.0 are treated as positive interactions (label=1), rest as negative.

References:
    - He, X. et al. (2017). Neural Collaborative Filtering. WWW.
    - Rendle, S. et al. (2020). Neural Collaborative Filtering vs. Matrix
      Factorization Revisited. RecSys.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config import RANDOM_SEED, NCF_EMBEDDING_DIM, NCF_BATCH_SIZE, NCF_EPOCHS

logger = logging.getLogger(__name__)


class NCFModel(nn.Module):
    """Neural Collaborative Filtering combining GMF and MLP branches.

    Architecture:
    - GMF branch: user_emb ⊙ item_emb (element-wise product)
    - MLP branch: concat(user_emb, item_emb) → hidden layers → output
    - Combined: concat(GMF, MLP) → sigmoid → interaction probability
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = NCF_EMBEDDING_DIM,
        mlp_layers: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        mlp_layers = mlp_layers or [128, 64, 32]

        # GMF embeddings
        self.user_emb_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_emb_gmf = nn.Embedding(n_items, embedding_dim)

        # MLP embeddings
        self.user_emb_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_emb_mlp = nn.Embedding(n_items, embedding_dim)

        # MLP layers
        mlp_input = embedding_dim * 2
        mlp_dims = [mlp_input] + mlp_layers
        self.mlp_layers = nn.ModuleList()
        for i in range(len(mlp_dims) - 1):
            self.mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))

        # Final prediction layer
        self.prediction = nn.Linear(embedding_dim + mlp_layers[-1], 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            user_ids: Tensor of user indices.
            item_ids: Tensor of item indices.

        Returns:
            Sigmoid output in [0, 1] — interaction probability.
        """
        # GMF branch
        u_gmf = self.user_emb_gmf(user_ids)
        i_gmf = self.item_emb_gmf(item_ids)
        gmf_out = u_gmf * i_gmf

        # MLP branch
        u_mlp = self.user_emb_mlp(user_ids)
        i_mlp = self.item_emb_mlp(item_ids)
        mlp_in = torch.cat([u_mlp, i_mlp], dim=-1)
        x = mlp_in
        for layer in self.mlp_layers:
            x = layer(x)
        mlp_out = x

        # Combine and predict
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        output = torch.sigmoid(self.prediction(combined).squeeze(-1))
        return output

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude: Optional[set] = None,
        n: int = 10,
        device: str = "cpu",
    ) -> List[Tuple[int, float]]:
        """Score all items for a user. Return top-N.

        Args:
            user_idx: Mapped user index.
            n_items: Total number of items.
            exclude: Set of item indices to exclude.
            n: Number of recommendations.
            device: torch device.

        Returns:
            List of (item_idx, score) tuples.
        """
        self.eval()
        with torch.no_grad():
            user_tensor = torch.full((n_items,), user_idx, dtype=torch.long, device=device)
            item_tensor = torch.arange(n_items, dtype=torch.long, device=device)
            scores = self.forward(user_tensor, item_tensor).cpu().numpy()

        if exclude:
            for idx in exclude:
                if 0 <= idx < len(scores):
                    scores[idx] = -1.0

        top_indices = np.argsort(scores)[-n:][::-1]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0][:n]


def _generate_negative_samples(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    n_items: int,
    neg_ratio: int = 4,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate negative samples for implicit feedback training.

    In implicit feedback settings, we only observe positive interactions
    (user watched/rated an item). Negative sampling creates synthetic
    "non-interaction" pairs by randomly pairing users with items they
    haven't interacted with. This teaches the model to discriminate
    between items a user likes vs. random items.

    Args:
        user_indices: Array of user indices from positive interactions.
        item_indices: Array of item indices from positive interactions.
        n_items: Total number of items in the catalog.
        neg_ratio: Number of negative samples per positive sample.
            Higher ratios improve precision but slow training.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (all_users, all_items, all_labels) arrays combining
        positive samples (label=1) and negative samples (label=0).
    """
    rng = np.random.default_rng(seed)

    # Build set of observed interactions for fast lookup
    positive_pairs = set(zip(user_indices, item_indices))
    n_positive = len(user_indices)

    neg_users = []
    neg_items = []

    for idx in range(n_positive):
        uid = user_indices[idx]
        # Sample neg_ratio items that this user hasn't interacted with
        sampled = 0
        while sampled < neg_ratio:
            neg_item = rng.integers(0, n_items)
            if (uid, neg_item) not in positive_pairs:
                neg_users.append(uid)
                neg_items.append(neg_item)
                sampled += 1

    # Combine positive and negative samples
    all_users = np.concatenate([user_indices, np.array(neg_users)])
    all_items = np.concatenate([item_indices, np.array(neg_items)])
    all_labels = np.concatenate([
        np.ones(n_positive, dtype=np.float32),
        np.zeros(len(neg_users), dtype=np.float32),
    ])

    return all_users, all_items, all_labels


def train_ncf(
    model: NCFModel,
    train_df: "pd.DataFrame",
    user_map: Dict[int, int],
    item_map: Dict[int, int],
    n_items: int,
    n_epochs: int = NCF_EPOCHS,
    batch_size: int = NCF_BATCH_SIZE,
    lr: float = 0.001,
    neg_ratio: int = 4,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """Train NCF model with BCE loss and negative sampling.

    Uses implicit feedback: ratings ≥ 3.0 are positive interactions,
    and random unobserved user–item pairs serve as negative examples.
    Negative samples are regenerated each epoch to expose the model
    to diverse non-interactions (dynamic negative sampling).

    Args:
        model: NCFModel instance.
        train_df: Training ratings DataFrame.
        user_map: User ID to index mapping.
        item_map: Item ID to index mapping.
        n_items: Total number of items.
        n_epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        neg_ratio: Negative samples per positive interaction.
        device: torch device.

    Returns:
        Dict with 'train_loss' list per epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Extract positive interactions (implicit feedback threshold)
    positive_mask = train_df["rating"] >= 3.0
    pos_df = train_df[positive_mask]
    pos_user_indices = pos_df["userId"].map(user_map).values
    pos_item_indices = pos_df["movieId"].map(item_map).values

    history: Dict[str, List[float]] = {"train_loss": []}

    for epoch in range(n_epochs):
        model.train()

        # Dynamic negative sampling: resample each epoch for variety
        all_users, all_items, all_labels = _generate_negative_samples(
            pos_user_indices, pos_item_indices, n_items,
            neg_ratio=neg_ratio, seed=RANDOM_SEED + epoch,
        )

        n_samples = len(all_labels)
        indices = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            u = torch.tensor(all_users[batch_idx], dtype=torch.long, device=device)
            i = torch.tensor(all_items[batch_idx], dtype=torch.long, device=device)
            y = torch.tensor(all_labels[batch_idx], dtype=torch.float, device=device)

            optimizer.zero_grad()
            pred = model(u, i)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)
        logger.info("NCF Epoch %d/%d: loss=%.4f (pos=%d, neg=%d)",
                    epoch + 1, n_epochs, avg_loss,
                    len(pos_user_indices), n_samples - len(pos_user_indices))

    return history
