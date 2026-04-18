"""MovieLens loader — downloads the published zip, verifies SHA-256, and reads CSVs.

The loader is intentionally minimal: any pandas / numpy work belongs in the
preprocessor. ``load_movielens`` returns a ``RawData`` with explicit dtypes so
downstream code never has to guess column types.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

import pandas as pd

from src.data.ids import RawData

DatasetName = Literal["ml-latest-small", "ml-25m"]

BASE_URL = "https://files.grouplens.org/datasets/movielens"

# gate: SHA-256 of the published MovieLens zips. Update in a dedicated commit
# whenever GroupLens re-uploads the archives.
CHECKSUMS: dict[str, str] = {
    "ml-latest-small": "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436",
    "ml-25m": "8b21cfb7eb1706b4ec0aac894368d90acf26ebdfb6aced3ebd4ad5bd1eb9c6aa",
}

_RATINGS_DTYPES: Mapping[Hashable, str] = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}

_MOVIES_DTYPES: Mapping[Hashable, str] = {
    "movieId": "int32",
    "title": "string",
    "genres": "string",
}


def load_movielens(
    name: DatasetName,
    data_dir: Path = Path("data/raw"),
) -> RawData:
    """Load a MovieLens dataset by name, downloading and verifying it if absent.

    Args:
        name: Dataset identifier — one of the keys in ``CHECKSUMS``.
        data_dir: Root directory under which ``<name>/`` will be extracted.

    Returns:
        A ``RawData`` with typed ``ratings`` and ``movies`` DataFrames.

    Raises:
        ValueError: If ``name`` is not a registered dataset.
        NotADirectoryError: If ``data_dir`` exists and is not a directory.
        RuntimeError: If the downloaded zip's SHA-256 does not match.
    """
    if name not in CHECKSUMS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(CHECKSUMS)}")
    if data_dir.exists() and not data_dir.is_dir():
        raise NotADirectoryError(f"data_dir exists and is not a directory: {data_dir}")

    dataset_dir = data_dir / name
    _ensure_extracted(name, data_dir=data_dir, dataset_dir=dataset_dir)

    ratings = pd.read_csv(dataset_dir / "ratings.csv", dtype=_RATINGS_DTYPES)
    movies = pd.read_csv(dataset_dir / "movies.csv", dtype=_MOVIES_DTYPES)
    return RawData(ratings=ratings, movies=movies)


def _ensure_extracted(name: str, *, data_dir: Path, dataset_dir: Path) -> None:
    if (dataset_dir / "ratings.csv").exists():
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"{name}.zip"
    _download(url=f"{BASE_URL}/{name}.zip", dest=zip_path)
    _verify_sha256(zip_path, expected=CHECKSUMS[name])

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)

    zip_path.unlink()


def _download(url: str, dest: Path) -> None:
    # concept: stream to disk so large archives (ml-25m is ~260 MB) never sit in RAM.
    with urlopen(url) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def _verify_sha256(path: Path, *, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {path.name}: expected {expected}, got {actual}")
