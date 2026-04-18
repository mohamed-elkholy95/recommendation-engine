"""End-to-end test — fit every component on real ml-latest-small, spin up the
FastAPI app via ``TestClient``, drive the full request path for every route.

These tests are ``@pytest.mark.integration``-marked and are skipped in the
default CI lane. Run them explicitly with::

    pytest tests/test_e2e.py -m integration

Fitting the three components on ml-latest-small costs ~5 seconds on GPU
(cu132 + RTX 5080) and ~30 seconds on CPU. Both fits are cached at module
scope so every test in this file reuses the same fitted app.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import InMemoryRatingStore
from src.data.ids import PreprocessedData
from src.data.loader import load_movielens
from src.data.preprocessor import preprocess
from src.models.collaborative import SvdModel
from src.models.content_based import ContentModel
from src.models.hybrid import HybridModel
from src.models.neural_cf import NcfModel

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def fitted_app() -> Iterable[tuple[TestClient, PreprocessedData]]:
    raw = load_movielens("ml-latest-small", data_dir=Path("data/raw"))
    pre = preprocess(raw)

    svd = SvdModel(n_factors=32, seed=42)
    svd.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)

    content = ContentModel()
    content.fit(raw.movies, pre)

    neural = NcfModel(n_factors=16, n_epochs=2, seed=42)
    neural.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)

    hybrid = HybridModel(
        collaborative=svd,
        content=content,
        neural=neural,
        n_items=pre.n_items,
    )
    app = create_app(
        model=hybrid,
        user_map=pre.user_map,
        item_map=pre.item_map,
        rating_store=InMemoryRatingStore(),
    )
    with TestClient(app) as client:
        yield client, pre


def test_health_endpoint_reports_ok(
    fitted_app: tuple[TestClient, PreprocessedData],
) -> None:
    client, _ = fitted_app
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_returns_n_unseen_items_for_a_real_user(
    fitted_app: tuple[TestClient, PreprocessedData],
) -> None:
    client, pre = fitted_app
    raw_user_id = int(next(iter(pre.user_map)))
    user_idx = int(pre.user_map[next(iter(pre.user_map))])
    known_movie_ids = {int(m) for m in pre.item_map}
    reverse_item_map = {int(v): int(k) for k, v in pre.item_map.items()}

    response = client.post("/recommend", json={"user_id": raw_user_id, "n": 10})

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == raw_user_id
    assert len(body["items"]) == 10

    seen_dense_idxs = {int(i) for i in pre.train[pre.train["user_idx"] == user_idx]["item_idx"]}
    seen_movie_ids = {reverse_item_map[i] for i in seen_dense_idxs}
    for item in body["items"]:
        assert item["movie_id"] in known_movie_ids
        assert isinstance(item["score"], float)
        # gate: seen items must be excluded by default.
        assert item["movie_id"] not in seen_movie_ids


def test_recommend_unknown_user_returns_404(
    fitted_app: tuple[TestClient, PreprocessedData],
) -> None:
    client, _ = fitted_app
    response = client.post("/recommend", json={"user_id": 99999, "n": 5})
    assert response.status_code == 404
    assert "user" in response.json()["detail"].lower()


def test_recommend_validates_n(fitted_app: tuple[TestClient, PreprocessedData]) -> None:
    client, _ = fitted_app
    response = client.post("/recommend", json={"user_id": 1, "n": 0})
    assert response.status_code == 422


def test_rate_then_recommend_workflow(
    fitted_app: tuple[TestClient, PreprocessedData],
) -> None:
    client, pre = fitted_app
    user_id = int(next(iter(pre.user_map)))
    movie_id = int(next(iter(pre.item_map)))

    rate_response = client.post(
        "/rate",
        json={"user_id": user_id, "movie_id": movie_id, "rating": 4.5},
    )
    assert rate_response.status_code == 200
    body = rate_response.json()
    assert body["accepted"] is True
    assert body["stored_count"] >= 1

    rec_response = client.post("/recommend", json={"user_id": user_id, "n": 5})
    assert rec_response.status_code == 200
    assert len(rec_response.json()["items"]) == 5


def test_rate_rejects_unknown_movie(
    fitted_app: tuple[TestClient, PreprocessedData],
) -> None:
    client, pre = fitted_app
    user_id = int(next(iter(pre.user_map)))
    response = client.post(
        "/rate",
        json={"user_id": user_id, "movie_id": 99999, "rating": 4.0},
    )
    assert response.status_code == 404


def test_rate_rejects_out_of_range_rating(
    fitted_app: tuple[TestClient, PreprocessedData],
) -> None:
    client, pre = fitted_app
    user_id = int(next(iter(pre.user_map)))
    movie_id = int(next(iter(pre.item_map)))
    response = client.post(
        "/rate",
        json={"user_id": user_id, "movie_id": movie_id, "rating": 10.0},
    )
    assert response.status_code == 422


def test_recommend_scores_are_finite_and_bounded(
    fitted_app: tuple[TestClient, PreprocessedData],
) -> None:
    # concept: MMR re-ranks the top-`pool_size` candidates — the *order* is
    # MMR-optimised, not score-sorted, so we cannot assert monotonicity on the
    # response. What we CAN assert is that every emitted score is finite and
    # inside the min-max-normalised [0, 1] range (the sum of three weighted
    # normalised scores that themselves sit in [0, 1]).
    import math

    client, pre = fitted_app
    user_id = int(next(iter(pre.user_map)))
    response = client.post("/recommend", json={"user_id": user_id, "n": 20})

    assert response.status_code == 200
    for item in response.json()["items"]:
        assert math.isfinite(item["score"])
        assert 0.0 <= item["score"] <= 1.0 + 1e-6


def test_recommend_items_are_unique(
    fitted_app: tuple[TestClient, PreprocessedData],
) -> None:
    client, pre = fitted_app
    user_id = int(next(iter(pre.user_map)))
    response = client.post("/recommend", json={"user_id": user_id, "n": 20})

    movie_ids = [item["movie_id"] for item in response.json()["items"]]
    assert len(movie_ids) == len(set(movie_ids))
