"""Tests for src.api routes — /health, /recommend, /rate."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import InMemoryRatingStore
from src.data.ids import ItemIdx, PreprocessedData, RawMovieId, RawUserId, UserIdx
from src.models.collaborative import SvdModel
from src.models.content_based import ContentModel
from src.models.hybrid import HybridModel
from src.models.neural_cf import NcfModel


def _fitted_hybrid() -> tuple[HybridModel, PreprocessedData]:
    rows = [(u, i, 5.0) for u in range(4) for i in range(4)] + [
        (u, i, 5.0) for u in range(4, 8) for i in range(4, 8)
    ]
    train = pd.DataFrame(
        {
            "user_idx": pd.array([r[0] for r in rows], dtype="int32"),
            "item_idx": pd.array([r[1] for r in rows], dtype="int32"),
            "rating": pd.array([r[2] for r in rows], dtype="float32"),
            "timestamp": pd.array([0] * len(rows), dtype="int64"),
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": list(range(100, 108)),
            "title": [f"Movie{i}" for i in range(8)],
            "genres": (["Action"] * 4) + (["Comedy"] * 4),
        }
    )
    pre = PreprocessedData(
        train=train,
        val=train.iloc[:0].copy(),
        test=train.iloc[:0].copy(),
        user_map={RawUserId(1000 + u): UserIdx(u) for u in range(8)},
        item_map={RawMovieId(100 + i): ItemIdx(i) for i in range(8)},
        n_users=8,
        n_items=8,
    )

    svd = SvdModel(n_factors=2, seed=0)
    svd.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)
    content = ContentModel()
    content.fit(movies, pre)
    neural = NcfModel(n_factors=4, hidden=(8,), n_epochs=5, batch_size=8, seed=0)
    neural.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)

    hybrid = HybridModel(
        collaborative=svd,
        content=content,
        neural=neural,
        n_items=pre.n_items,
        mmr_lambda=1.0,
    )
    return hybrid, pre


@pytest.fixture
def client() -> TestClient:
    hybrid, pre = _fitted_hybrid()
    app = create_app(
        model=hybrid,
        user_map=pre.user_map,
        item_map=pre.item_map,
        rating_store=InMemoryRatingStore(),
    )
    return TestClient(app)


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_returns_unseen_items_for_known_user(client: TestClient) -> None:
    response = client.post("/recommend", json={"user_id": 1000, "n": 4})

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 1000
    assert len(body["items"]) == 4
    # User 1000 rated movies 100-103 → all recs must come from 104-107.
    assert {item["movie_id"] for item in body["items"]}.issubset({104, 105, 106, 107})
    # Each item carries a numeric score.
    for item in body["items"]:
        assert isinstance(item["score"], float)


def test_recommend_unknown_user_returns_404(client: TestClient) -> None:
    response = client.post("/recommend", json={"user_id": 99999, "n": 5})
    assert response.status_code == 404
    assert "user" in response.json()["detail"].lower()


def test_recommend_rejects_non_positive_n(client: TestClient) -> None:
    response = client.post("/recommend", json={"user_id": 1000, "n": 0})
    assert response.status_code == 422


def test_rate_records_interaction(client: TestClient) -> None:
    response = client.post(
        "/rate",
        json={"user_id": 1000, "movie_id": 104, "rating": 4.5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["stored_count"] == 1


def test_rate_unknown_user_returns_404(client: TestClient) -> None:
    response = client.post(
        "/rate",
        json={"user_id": 99999, "movie_id": 104, "rating": 4.0},
    )
    assert response.status_code == 404


def test_rate_unknown_movie_returns_404(client: TestClient) -> None:
    response = client.post(
        "/rate",
        json={"user_id": 1000, "movie_id": 99999, "rating": 4.0},
    )
    assert response.status_code == 404


def test_rate_rejects_out_of_range_rating(client: TestClient) -> None:
    response = client.post(
        "/rate",
        json={"user_id": 1000, "movie_id": 104, "rating": 10.0},
    )
    assert response.status_code == 422


def test_rate_stores_multiple_interactions_and_counts_them(client: TestClient) -> None:
    for movie_id in (104, 105, 106):
        response = client.post(
            "/rate",
            json={"user_id": 1000, "movie_id": movie_id, "rating": 4.0},
        )
        assert response.status_code == 200
    body = response.json()
    assert body["stored_count"] == 3
