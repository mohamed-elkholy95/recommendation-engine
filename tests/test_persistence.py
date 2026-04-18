"""Tests for src.persistence — pickle round-trips for every fitted model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.ids import ItemIdx, PreprocessedData, RawMovieId, RawUserId, UserIdx
from src.models.collaborative import AlsModel, SvdModel
from src.models.content_based import ContentModel
from src.models.neural_cf import NcfModel
from src.persistence import load_model, save_model


def _train_frame() -> pd.DataFrame:
    rows = [(u, i, 5.0) for u in range(4) for i in range(4)] + [
        (u, i, 5.0) for u in range(4, 8) for i in range(4, 8)
    ]
    return pd.DataFrame(
        {
            "user_idx": pd.array([r[0] for r in rows], dtype="int32"),
            "item_idx": pd.array([r[1] for r in rows], dtype="int32"),
            "rating": pd.array([r[2] for r in rows], dtype="float32"),
            "timestamp": pd.array([0] * len(rows), dtype="int64"),
        }
    )


def test_svd_model_round_trips_through_pickle(tmp_path: Path) -> None:
    model = SvdModel(n_factors=2, seed=0)
    model.fit(_train_frame(), n_users=8, n_items=8)

    before = [model.predict(UserIdx(u), ItemIdx(i)) for u in range(8) for i in range(8)]

    path = tmp_path / "svd.pkl"
    save_model(model, path)
    restored: SvdModel = load_model(path)

    after = [restored.predict(UserIdx(u), ItemIdx(i)) for u in range(8) for i in range(8)]
    assert before == after


def test_als_model_round_trips_through_pickle(tmp_path: Path) -> None:
    model = AlsModel(n_factors=2, n_iter=5, seed=0)
    model.fit(_train_frame(), n_users=8, n_items=8)

    before = [model.predict(UserIdx(u), ItemIdx(i)) for u in range(8) for i in range(8)]

    path = tmp_path / "als.pkl"
    save_model(model, path)
    restored: AlsModel = load_model(path)

    after = [restored.predict(UserIdx(u), ItemIdx(i)) for u in range(8) for i in range(8)]
    assert before == after


def test_content_model_round_trips_through_pickle(tmp_path: Path) -> None:
    movies = pd.DataFrame(
        {
            "movieId": list(range(100, 108)),
            "title": [f"Movie{i}" for i in range(8)],
            "genres": (["Action"] * 4) + (["Comedy"] * 4),
        }
    )
    train = _train_frame()
    pre = PreprocessedData(
        train=train,
        val=train.iloc[:0].copy(),
        test=train.iloc[:0].copy(),
        user_map={RawUserId(u): UserIdx(u) for u in range(8)},
        item_map={RawMovieId(100 + i): ItemIdx(i) for i in range(8)},
        n_users=8,
        n_items=8,
    )

    model = ContentModel()
    model.fit(movies, pre)

    before = model.recommend(UserIdx(0), n=5, exclude_seen=False)

    path = tmp_path / "content.pkl"
    save_model(model, path)
    restored: ContentModel = load_model(path)

    after = restored.recommend(UserIdx(0), n=5, exclude_seen=False)
    assert before == after


def test_ncf_model_round_trips_through_pickle(tmp_path: Path) -> None:
    model = NcfModel(n_factors=4, hidden=(8,), n_epochs=3, batch_size=8, seed=0)
    model.fit(_train_frame(), n_users=8, n_items=8)

    before = [model.predict(UserIdx(u), ItemIdx(i)) for u in range(8) for i in range(8)]

    path = tmp_path / "ncf.pkl"
    save_model(model, path)
    restored: NcfModel = load_model(path)

    after = [restored.predict(UserIdx(u), ItemIdx(i)) for u in range(8) for i in range(8)]
    # concept: tiny float drift from the torch device-move inside save_model
    # (gpu → cpu → gpu) is expected; we accept a tight absolute tolerance.
    for b, a in zip(before, after, strict=True):
        assert abs(b - a) < 1e-5


def test_load_model_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.pkl")


def test_save_model_creates_parent_directory(tmp_path: Path) -> None:
    model = SvdModel(n_factors=2, seed=0)
    model.fit(_train_frame(), n_users=8, n_items=8)

    nested = tmp_path / "a" / "b" / "svd.pkl"
    save_model(model, nested)
    assert nested.exists()
