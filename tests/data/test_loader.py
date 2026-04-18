"""Tests for src.data.loader — MovieLens download + verify + read."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import pytest

from src.data.loader import CHECKSUMS, load_movielens


def _write_tiny_csvs(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n1,10,4.0,1000\n2,10,3.5,1100\n"
    )
    (dataset_dir / "movies.csv").write_text(
        'movieId,title,genres\n10,"Toy Story (1995)",Animation\n'
    )


def test_unknown_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        load_movielens("ml-42m", data_dir=tmp_path)  # type: ignore[arg-type]


def test_reads_extracted_csvs_without_network(tmp_path: Path) -> None:
    _write_tiny_csvs(tmp_path / "ml-latest-small")

    raw = load_movielens("ml-latest-small", data_dir=tmp_path)

    assert list(raw.ratings.columns) == ["userId", "movieId", "rating", "timestamp"]
    assert raw.ratings["userId"].dtype == "int32"
    assert raw.ratings["movieId"].dtype == "int32"
    assert raw.ratings["rating"].dtype == "float32"
    assert raw.ratings["timestamp"].dtype == "int64"
    assert len(raw.ratings) == 2
    assert raw.movies["title"].iloc[0] == "Toy Story (1995)"


def test_data_dir_is_a_file_raises(tmp_path: Path) -> None:
    collision = tmp_path / "not-a-dir"
    collision.write_text("I'm a file, not a directory")

    with pytest.raises(NotADirectoryError):
        load_movielens("ml-latest-small", data_dir=collision)


def _fake_urlopen_returning(payload: bytes) -> Any:
    def _opener(url: str, *args: object, **kwargs: object) -> io.BytesIO:
        return io.BytesIO(payload)

    return _opener


def test_checksum_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bogus_zip = io.BytesIO()
    with zipfile.ZipFile(bogus_zip, "w") as zf:
        zf.writestr("ml-latest-small/ratings.csv", "userId,movieId,rating,timestamp\n")

    monkeypatch.setattr(
        "src.data.loader.urlopen", _fake_urlopen_returning(bogus_zip.getvalue())
    )

    with pytest.raises(RuntimeError, match="checksum"):
        load_movielens("ml-latest-small", data_dir=tmp_path)


def test_happy_path_download_extracts_and_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # concept: build a zip whose payload matches the registered SHA-256 so the
    # loader walks the full download → verify → extract → read path.
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "ml-latest-small/ratings.csv",
            "userId,movieId,rating,timestamp\n1,10,4.0,1000\n",
        )
        zf.writestr(
            "ml-latest-small/movies.csv",
            'movieId,title,genres\n10,"Toy Story (1995)",Animation\n',
        )
    payload = archive.getvalue()

    import hashlib

    fingerprint = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(CHECKSUMS, "ml-latest-small", fingerprint)
    monkeypatch.setattr("src.data.loader.urlopen", _fake_urlopen_returning(payload))

    raw = load_movielens("ml-latest-small", data_dir=tmp_path)

    assert len(raw.ratings) == 1
    assert (tmp_path / "ml-latest-small" / "ratings.csv").exists()
    # gate: the zip is deleted after extraction to keep data/raw tidy.
    assert not (tmp_path / "ml-latest-small.zip").exists()


@pytest.mark.integration
def test_integration_small_real_download(tmp_path: Path) -> None:
    # concept: skipped in default CI — exercises the real network + checksum
    # path against the published MovieLens fixture.
    raw = load_movielens("ml-latest-small", data_dir=tmp_path)

    assert len(raw.ratings) > 0
    assert len(raw.movies) > 0
    assert raw.ratings["userId"].dtype == "int32"


def test_checksums_dict_covers_supported_names() -> None:
    assert set(CHECKSUMS.keys()) == {"ml-latest-small", "ml-25m"}
    for digest in CHECKSUMS.values():
        assert len(digest) == 64
        int(digest, 16)  # valid hex
