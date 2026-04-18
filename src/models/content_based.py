"""Content-based retrieval — TF-IDF item embeddings, rating-weighted user profile.

Each movie is encoded as a TF-IDF vector over its title + genre tokens. Every
user is represented as the rating-weighted mean of the item vectors they rated
in training, L2-normalised so the score is a cosine similarity.

Unlike the collaborative models, this one never trains on co-rating patterns;
it only reads metadata. It therefore handles cold-start items the collaborative
models cannot — but misses taste signals the collaborative models exploit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.ids import ItemIdx, PreprocessedData, RawMovieId, UserIdx
from src.models._ranking import (
    build_seen_dict,
    require_fitted,
    require_non_empty_train,
    top_k_from_scores,
)

FloatArray = npt.NDArray[np.float32]


@dataclass
class ContentModel:
    """TF-IDF + rating-weighted user profile content-based recommender."""

    _item_embeddings: csr_matrix = field(
        init=False, default_factory=lambda: csr_matrix((0, 0), dtype=np.float32)
    )
    _user_profiles: FloatArray = field(
        init=False, default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    _seen: dict[UserIdx, set[ItemIdx]] = field(init=False, default_factory=dict)
    _fitted: bool = field(init=False, default=False)

    def fit(self, movies: pd.DataFrame, preprocessed: PreprocessedData) -> None:
        """Fit TF-IDF over ``title + genres`` and build a profile per train user.

        Args:
            movies: Columns ``movieId``, ``title``, ``genres`` (as loaded from
                MovieLens). Rows whose ``movieId`` is not in
                ``preprocessed.item_map`` are silently ignored.
            preprocessed: The pipeline output — brings ``item_map``, ``train``,
                and size metadata.

        Raises:
            ValueError: If ``preprocessed.train`` is empty.
        """
        require_non_empty_train(preprocessed.train)

        docs = _build_item_documents(movies, item_map=preprocessed.item_map)
        vectorizer = TfidfVectorizer(lowercase=True)
        item_embeddings = vectorizer.fit_transform(docs).astype(np.float32)
        # TfidfVectorizer already L2-normalises rows with the default ``norm='l2'``.

        self._user_profiles = _build_user_profiles(
            train=preprocessed.train,
            item_embeddings=item_embeddings,
            n_users=preprocessed.n_users,
            n_items=preprocessed.n_items,
        )
        self._item_embeddings = item_embeddings
        self._seen = build_seen_dict(preprocessed.train)
        self._fitted = True

    def predict(self, user_idx: UserIdx, item_idx: ItemIdx) -> float:
        """Cosine similarity between the user's profile and one item embedding."""
        require_fitted(self._fitted, "ContentModel")
        item_row = self._item_embeddings.getrow(int(item_idx)).toarray().ravel()
        return float(self._user_profiles[int(user_idx)] @ item_row)

    def recommend(
        self,
        user_idx: UserIdx,
        *,
        n: int,
        exclude_seen: bool = True,
    ) -> list[tuple[ItemIdx, float]]:
        """Top-``n`` items by cosine similarity with the user's profile."""
        require_fitted(self._fitted, "ContentModel")
        profile = self._user_profiles[int(user_idx)]
        scores: FloatArray = (self._item_embeddings @ profile).astype(np.float32)
        return top_k_from_scores(
            scores,
            seen=self._seen.get(user_idx, set()),
            n=n,
            exclude_seen=exclude_seen,
        )


def _build_item_documents(
    movies: pd.DataFrame, *, item_map: dict[RawMovieId, ItemIdx]
) -> list[str]:
    # concept: one document string per dense ItemIdx — any item missing from
    # ``movies`` gets the empty string (TF-IDF yields a zero vector).
    docs: list[str] = ["" for _ in range(len(item_map))]
    for movie_id, title, genres in zip(
        movies["movieId"].to_numpy(),
        movies["title"].to_numpy(),
        movies["genres"].to_numpy(),
        strict=True,
    ):
        raw_id = RawMovieId(int(movie_id))
        if raw_id not in item_map:
            continue
        docs[int(item_map[raw_id])] = f"{title} {str(genres).replace('|', ' ')}"
    return docs


def _build_user_profiles(
    *,
    train: pd.DataFrame,
    item_embeddings: csr_matrix,
    n_users: int,
    n_items: int,
) -> FloatArray:
    rating_matrix = csr_matrix(
        (
            train["rating"].to_numpy(dtype=np.float32),
            (train["user_idx"].to_numpy(), train["item_idx"].to_numpy()),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    row_sums = np.asarray(rating_matrix.sum(axis=1)).ravel()
    # concept: avoid a div-by-zero for users with no ratings — the weight row
    # stays at zero so their profile ends up as the zero vector.
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    weights = diags(1.0 / safe_sums) @ rating_matrix

    profiles = (weights @ item_embeddings).toarray().astype(np.float32)
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalised: FloatArray = (profiles / norms).astype(np.float32)
    return normalised
