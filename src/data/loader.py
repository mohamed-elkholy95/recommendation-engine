"""Data loading and preprocessing for MovieLens recommendation engine."""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, RAW_DIR

logger = logging.getLogger(__name__)


def generate_synthetic_movielens(
    n_users: int = 1000, n_movies: int = 500, n_ratings: int = 50000,
    seed: int = RANDOM_SEED,
) -> Dict[str, pd.DataFrame]:
    """Generate synthetic MovieLens-style data.

    Args:
        n_users: Number of users.
        n_movies: Number of movies.
        n_ratings: Number of ratings.
        seed: Random seed.

    Returns:
        Dict with 'ratings', 'movies', 'tags' DataFrames.
    """
    rng = np.random.default_rng(seed)

    # Movies
    genres = ["Action", "Adventure", "Animation", "Comedy", "Crime",
              "Documentary", "Drama", "Fantasy", "Horror", "Mystery",
              "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    titles = [f"Movie {i}: The {rng.choice(genres)} Adventure" for i in range(1, n_movies + 1)]
    movie_genres = ["|".join(rng.choice(genres, size=rng.integers(1, 4), replace=False))
                    for _ in range(n_movies)]
    years = rng.integers(1970, 2025, n_movies)

    movies = pd.DataFrame({
        "movieId": range(1, n_movies + 1),
        "title": titles,
        "genres": movie_genres,
        "year": years,
    })

    # Ratings
    user_ids = rng.integers(1, n_users + 1, n_ratings)
    movie_ids = rng.integers(1, n_movies + 1, n_ratings)
    # Power-law rating distribution: some users rate many, most rate few
    ratings = rng.integers(1, 6, n_ratings).astype(float)
    # Add bias: popular movies get higher ratings
    popularity = pd.Series(movie_ids).value_counts()
    for mid in popularity.head(50).index:
        mask = movie_ids == mid
        ratings[mask] = np.clip(ratings[mask] + rng.normal(0.5, 0.5, mask.sum()), 1, 5)
    timestamps = rng.integers(1_500_000_000, 1_700_000_000, n_ratings)

    # Remove duplicates (keep last)
    ratings_df = pd.DataFrame({
        "userId": user_ids, "movieId": movie_ids,
        "rating": ratings, "timestamp": timestamps,
    }).drop_duplicates(subset=["userId", "movieId"], keep="last")

    # Tags
    n_tags = n_ratings // 5
    tag_data = pd.DataFrame({
        "userId": rng.integers(1, n_users + 1, n_tags),
        "movieId": rng.integers(1, n_movies + 1, n_tags),
        "tag": [f"tag_{i}" for i in rng.integers(0, 100, n_tags)],
        "timestamp": rng.integers(1_500_000_000, 1_700_000_000, n_tags),
    }).drop_duplicates(subset=["userId", "movieId", "tag"], keep="last")

    logger.info("Generated synthetic data: %d ratings, %d movies, %d users",
                len(ratings_df), n_movies, n_users)
    return {"ratings": ratings_df, "movies": movies, "tags": tag_data}


def load_movielens(data_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Load MovieLens CSV files or generate synthetic data.

    Args:
        data_dir: Path to data directory. If None, uses RAW_DIR.

    Returns:
        Dict with 'ratings', 'movies', 'tags' DataFrames.
    """
    path = Path(data_dir) if data_dir else RAW_DIR

    ratings_path = path / "ratings.csv"
    movies_path = path / "movies.csv"

    if ratings_path.exists() and movies_path.exists():
        logger.info("Loading MovieLens from %s", path)
        ratings = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)
        tags_path = path / "tags.csv"
        tags = pd.read_csv(tags_path) if tags_path.exists() else pd.DataFrame()
    else:
        logger.info("No MovieLens files found — generating synthetic data")
        data = generate_synthetic_movielens()
        data["ratings"].to_csv(ratings_path, index=False)
        data["movies"].to_csv(movies_path, index=False)
        if "tags" in data and not data["tags"].empty:
            data["tags"].to_csv(path / "tags.csv", index=False)
        ratings = data["ratings"]
        movies = data["movies"]
        tags = data.get("tags", pd.DataFrame())

    return {"ratings": ratings, "movies": movies, "tags": tags}


def get_stats(ratings: pd.DataFrame, movies: pd.DataFrame) -> Dict:
    """Compute dataset statistics."""
    return {
        "n_ratings": len(ratings),
        "n_users": ratings["userId"].nunique(),
        "n_movies": ratings["movieId"].nunique(),
        "avg_rating": round(float(ratings["rating"].mean()), 2),
        "rating_std": round(float(ratings["rating"].std()), 2),
        "sparsity": round(1.0 - len(ratings) / (ratings["userId"].nunique() * ratings["movieId"].nunique()), 6),
        "genre_counts": movies["genres"].str.split("|").explode().value_counts().head(10).to_dict(),
    }
