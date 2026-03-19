# Project Plan: Hybrid Recommendation Engine

**Project:** 14-recommendation-engine  
**Category:** Recommendation Systems / Deep Learning  
**Difficulty:** Advanced (⭐⭐⭐⭐)  
**Timeline:** 12 days  
**Status:** 🔨 In Progress

---

## Overview

Build a production-grade hybrid recommendation system that combines three complementary approaches:
1. **Collaborative Filtering** — "Users like you liked this"
2. **Content-Based Filtering** — "This matches your taste profile"
3. **Neural Collaborative Filtering** — "Deep patterns in user-item interactions"

The system is built on MovieLens 25M (25M ratings) and augmented with TMDB movie metadata for rich content features. The final hybrid ensemble provides better coverage, accuracy, and diversity than any single approach.

---

## Phase 1: Data Pipeline (Days 1-2)

### Datasets
- **MovieLens 25M:** ratings.csv, movies.csv, tags.csv, links.csv
  - 25,000,095 ratings | 62,423 movies | 162,541 users
- **TMDB API:** poster images, descriptions, cast, director, budget, revenue

### Data Exploration
- Rating distribution (scale: 0.5–5.0 stars, mean ~3.5)
- Genre popularity (Drama, Comedy, Thriller most common)
- User activity distribution (power law — few users rate thousands)
- Item activity distribution (long tail — most movies have few ratings)
- Sparsity: 25M / (162K × 62K) = 0.25% density

### File: `src/data/loader.py`

```python
def load_movielens(path: str) -> dict[str, pd.DataFrame]:
    """
    Load all MovieLens files.
    Returns: {
        "ratings": DataFrame[userId, movieId, rating, timestamp],
        "movies": DataFrame[movieId, title, genres],
        "tags": DataFrame[userId, movieId, tag, timestamp],
        "links": DataFrame[movieId, imdbId, tmdbId]
    }
    """
    ...

def fetch_tmdb_metadata(
    movie_ids: list[int],
    api_key: str,
    cache_dir: str = "data/tmdb_cache"
) -> pd.DataFrame:
    """
    Fetch movie metadata from TMDB API.
    Respects rate limit (40 req/10s).
    Caches results to avoid re-fetching.
    Returns: DataFrame[tmdbId, title, overview, genres, cast, director,
                        release_date, runtime, vote_average, poster_path]
    """
    ...
```

### File: `src/data/preprocessor.py`

```python
class DataPreprocessor:
    """
    End-to-end data preparation for recommendation models.
    """
    
    def clean_data(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        min_user_ratings: int = 20,
        min_item_ratings: int = 10
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove users with < min_user_ratings.
        Remove movies with < min_item_ratings.
        Re-index user and item IDs to contiguous integers.
        Returns: (clean_ratings, clean_movies)
        """
        ...
    
    def create_user_features(
        self,
        ratings: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate rating history per user.
        Features: avg_rating, rating_count, std_rating,
                  top_genres (one-hot), activity_period (days)
        """
        ...
    
    def create_item_features(
        self,
        movies: pd.DataFrame,
        tmdb_metadata: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Create item feature vectors.
        Features: genres (multi-hot), release_year, decade,
                  description_tfidf (if TMDB available),
                  cast_tfidf, director
        """
        ...
    
    def create_interaction_matrix(
        self,
        ratings: pd.DataFrame
    ) -> scipy.sparse.csr_matrix:
        """
        Build user-item sparse interaction matrix.
        Returns: CSR matrix shape (n_users, n_items)
        """
        ...
    
    def train_test_split_time(
        self,
        ratings: pd.DataFrame,
        test_ratio: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Per-user temporal split: last test_ratio% of each user's
        ratings → test. Remaining → train.
        Ensures test items were rated after train items.
        """
        ...
```

---

## Phase 2: Collaborative Filtering (Days 2-4)

### Approach 1: Memory-Based CF
- Compute user-user or item-item similarity from interaction matrix
- Predict ratings as weighted average of neighbor ratings
- Fast for small datasets; doesn't scale well to 25M

### Approach 2: Matrix Factorization
- Decompose interaction matrix into user and item latent factors
- SVD (Surprise), NMF, ALS (implicit)
- More scalable, better generalization

### File: `src/models/collaborative.py`

```python
from surprise import SVD, NMF, KNNBasic, Dataset, Reader

class UserBasedCF:
    """
    User-based collaborative filtering using cosine similarity.
    """
    
    def fit(self, ratings: pd.DataFrame) -> "UserBasedCF":
        """Build user similarity matrix."""
        ...
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair."""
        ...
    
    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Top-N item recommendations for user."""
        ...


class ItemBasedCF:
    """
    Item-based collaborative filtering using cosine similarity.
    Better for large user bases (item space is smaller).
    """
    
    def fit(self, ratings: pd.DataFrame) -> "ItemBasedCF":
        ...
    
    def predict(self, user_id: int, item_id: int) -> float:
        ...
    
    def similar_items(self, item_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return top-N most similar items."""
        ...


class MatrixFactorization:
    """
    Matrix factorization via SVD (Surprise library).
    
    Parameters:
    - n_factors: number of latent dimensions (default 100)
    - n_epochs: training epochs (default 20)
    - lr: learning rate (default 0.005)
    - reg: L2 regularization (default 0.02)
    """
    
    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        algorithm: str = "svd"  # svd | nmf | als
    ):
        ...
    
    def fit(self, train_data: pd.DataFrame) -> "MatrixFactorization":
        """Train matrix factorization model."""
        ...
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair."""
        ...
    
    def recommend(self, user_id: int, n: int = 10) -> list[dict]:
        """
        Top-N recommendations with predicted ratings.
        Returns: [{"movie_id": id, "score": float, "title": str}]
        """
        ...
    
    def similar_items(self, item_id: int, n: int = 10) -> list[dict]:
        """
        Find most similar items using item latent vectors.
        Cosine similarity in latent space.
        """
        ...
    
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Return user latent factor vector."""
        ...
    
    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Return item latent factor vector."""
        ...
```

---

## Phase 3: Content-Based Filtering (Days 4-5)

### Feature Construction
- **Genres:** Multi-hot encoding (19 MovieLens genre categories)
- **Description:** TF-IDF on TMDB overview (5000 terms, bigrams)
- **Cast:** TF-IDF on top-3 actors (weighted by billing order)
- **Director:** One-hot (frequency-filtered to top 500 directors)
- **Year:** Normalized decade feature

### Weighted Feature Vector
`genre×0.4 + description×0.3 + cast×0.2 + director×0.1`

### File: `src/models/content_based.py`

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    """
    Content-based recommender using movie metadata.
    
    Similarity is computed in feature space:
    item_matrix shape: (n_movies, n_features)
    Item-item cosine similarity for recommendations.
    User profile = weighted average of liked items.
    """
    
    def __init__(
        self,
        feature_weights: dict = None
    ):
        self.feature_weights = feature_weights or {
            "genre": 0.4,
            "description": 0.3,
            "cast": 0.2,
            "director": 0.1
        }
    
    def fit(self, movies_metadata: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build TF-IDF matrices for text features.
        Combine into weighted feature matrix.
        Pre-compute item-item similarity matrix (or use approximate NN).
        """
        ...
    
    def recommend(
        self,
        user_history: list[int],
        n: int = 10,
        exclude_seen: bool = True
    ) -> list[dict]:
        """
        Build user profile from history (average feature vector).
        Find top-N similar unseen items.
        Returns: [{"movie_id", "score", "title", "genres"}]
        """
        ...
    
    def similar_items(self, item_id: int, n: int = 10) -> list[dict]:
        """Return top-N most content-similar movies."""
        ...
    
    def explain_recommendation(
        self,
        user_id: int,
        item_id: int,
        user_history: list[int]
    ) -> dict:
        """
        Why was this item recommended?
        Returns: {
            "matching_genres": list[str],
            "similar_to": list[str],  # movies in history it resembles
            "feature_overlap": dict  # feature-by-feature similarity
        }
        """
        ...
```

---

## Phase 4: Neural Collaborative Filtering (Days 5-7)

### Architecture: Two-Tower NCF
- **User Tower:** embedding(user_id, dim=64) → MLP
- **Item Tower:** embedding(item_id, dim=64) → MLP
- **Interaction:** dot product + element-wise product → MLP → sigmoid
- **Training:** Binary cross-entropy (implicit feedback: rated=1, not-rated=0)
- **Negative Sampling:** 4 negatives per positive (popularity-weighted)

### File: `src/models/neural_cf.py`

```python
import torch
import torch.nn as nn

class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering combining GMF and MLP.
    
    Architecture:
    - GMF branch: user_emb ⊙ item_emb (generalized matrix factorization)
    - MLP branch: concat(user_emb, item_emb) → MLP layers
    - NeuMF: concat(GMF_out, MLP_out) → sigmoid
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        mlp_layers: list[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        mlp_layers = mlp_layers or [128, 64, 32]
        
        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)
        
        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        ...
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        Returns: sigmoid output in [0,1] (interaction probability)
        """
        ...
    
    def recommend(
        self,
        user_id: int,
        all_items: torch.Tensor,
        n: int = 10,
        exclude_seen: set = None
    ) -> list[tuple[int, float]]:
        """
        Score all items for user. Return top-N.
        Batched inference for efficiency.
        """
        ...


def train_ncf(
    model: NCFModel,
    train_data: pd.DataFrame,
    n_epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 0.001,
    patience: int = 5
) -> NCFModel:
    """
    Train NCF with BCE loss, Adam optimizer, early stopping on NDCG@10.
    Negative sampling: 4 negatives per positive (popularity-weighted).
    """
    ...
```

---

## Phase 5: Hybrid System (Days 7-8)

### Fusion Strategy
- Normalize all model scores to [0, 1]
- Weighted sum with learned or heuristic weights
- Default: CF=0.4, Content=0.3, NCF=0.3
- Re-rank for diversity using Maximum Marginal Relevance (MMR)

### File: `src/models/hybrid.py`

```python
class HybridRecommender:
    """
    Ensemble recommender combining multiple approaches.
    
    Score fusion: weighted sum of normalized scores.
    Diversity re-ranking: Maximum Marginal Relevance (MMR).
    """
    
    def __init__(self):
        self.recommenders: dict[str, Any] = {}
        self.weights: dict[str, float] = {}
    
    def add_recommender(
        self,
        name: str,
        recommender: Any,
        weight: float
    ) -> None:
        """Register a recommender component."""
        ...
    
    def recommend(
        self,
        user_id: int,
        n: int = 20,
        exclude_seen: bool = True
    ) -> list[dict]:
        """
        Get ensemble recommendations.
        1. Collect top candidates from each component
        2. Normalize scores to [0,1]
        3. Weighted sum
        4. Sort by final score
        Returns: top-N with per-model scores for transparency
        """
        ...
    
    def rerank(
        self,
        recommendations: list[dict],
        user_history: list[int],
        diversity_weight: float = 0.3,
        n: int = 10
    ) -> list[dict]:
        """
        MMR re-ranking: balance relevance vs diversity.
        Prevents recommending 10 similar sci-fi films.
        diversity_weight: 0.0 = pure relevance, 1.0 = pure diversity
        """
        ...
    
    def cold_start_recommend(
        self,
        user_profile: dict,
        n: int = 10
    ) -> list[dict]:
        """
        For new users with no rating history.
        Uses: stated preferences (genres, favorite movies) + popularity baseline.
        user_profile: {"genres": ["Action", "Drama"], "example_movies": [296, 318]}
        """
        ...
```

---

## Phase 6: FastAPI (Days 8-9)

### File: `src/api/main.py`

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/recommendations/{user_id}` | Personalized top-N recommendations |
| GET | `/similar/{item_id}` | Content-similar movies |
| GET | `/search` | Search movies by title/genre |
| POST | `/rate` | Submit rating (online learning queue) |
| GET | `/trending` | Trending movies (last 30 days) |
| GET | `/health` | Health check |
| GET | `/model/info` | Model versions and performance |

**Key Implementation Details:**
- Pre-load all models at startup
- Recommendation caching (user_id → recommendations, TTL=1hr)
- Async endpoints where possible
- Rate limiting: 100 req/min per IP

---

## Phase 7: Streamlit Dashboard (Days 9-11)

### File: `src/dashboard/app.py`

**Sections:**

1. **User Profile View**
   - User ID input or demo user selection
   - Rating history visualization (scatter: year vs rating)
   - Favorite genres radar chart

2. **Recommendation Feed**
   - Movie cards with TMDB poster images
   - Genre tags, year, average rating
   - Explanation: "Because you liked [movie] / Because you enjoy [genre]"

3. **Similar Movies Explorer**
   - Select any movie → see content-similar neighbors
   - Similarity breakdown: genre%, description%, cast%

4. **Model Comparison**
   - Show top-10 from CF, Content, NCF, Hybrid side by side
   - Highlight overlaps and differences

5. **Metrics Dashboard**
   - NDCG@10, MAP@10, coverage, diversity
   - Train/validation loss curves (NCF)

6. **Rating Submission**
   - Rate movies → updates recommendation list live

---

## Phase 8: Evaluation (Days 11-12)

### File: `src/evaluation.py`

```python
def ndcg_at_k(
    y_true: list[list[int]],
    y_pred: list[list[int]],
    k: int = 10
) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    ...

def map_at_k(
    y_true: list[list[int]],
    y_pred: list[list[int]],
    k: int = 10
) -> float:
    """Mean Average Precision at K."""
    ...

def hit_rate_at_k(y_true, y_pred, k=10) -> float:
    """Fraction of users with at least 1 hit in top-K."""
    ...

def catalog_coverage(
    recommendations: dict[int, list[int]],
    n_items: int
) -> float:
    """Fraction of catalog recommended to at least 1 user."""
    ...

def intra_list_diversity(
    recommendations: dict[int, list[int]],
    item_features: np.ndarray
) -> float:
    """Average pairwise distance within recommended lists."""
    ...

def novelty(
    recommendations: dict[int, list[int]],
    item_popularity: dict[int, float]
) -> float:
    """Average self-information (log(1/popularity)) of recommendations."""
    ...
```

---

## Testing Plan

| Test File | Coverage |
|---|---|
| `tests/test_loader.py` | load_movielens schema, fetch_tmdb caching |
| `tests/test_preprocessor.py` | clean_data, split_time, interaction matrix |
| `tests/test_collaborative.py` | UserCF, ItemCF, MatrixFactorization predict |
| `tests/test_content_based.py` | fit, recommend, similar_items, explain |
| `tests/test_neural_cf.py` | forward pass shapes, recommend output |
| `tests/test_hybrid.py` | ensemble scoring, rerank diversity |
| `tests/test_evaluation.py` | NDCG, MAP, coverage metrics |
| `tests/test_api.py` | All endpoints with TestClient |

---

## Dependencies

```txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3.0
scikit-surprise>=1.1.3
torch>=2.1.0
fastapi>=0.109.0
uvicorn>=0.27.0
streamlit>=1.30.0
plotly>=5.18.0
requests>=2.31.0
Pillow>=10.0.0
scipy>=1.11.0
pydantic>=2.0.0
pytest>=7.4.0
httpx>=0.25.0
```

---

## Timeline

| Day | Tasks |
|---|---|
| 1-2 | EDA, data pipeline, preprocessing |
| 2-4 | Collaborative filtering (memory + matrix factorization) |
| 4-5 | Content-based filtering |
| 5-7 | Neural collaborative filtering (NCF) |
| 7-8 | Hybrid ensemble, re-ranking |
| 8-9 | FastAPI |
| 9-11 | Streamlit dashboard |
| 11-12 | Evaluation, testing, documentation |

---

## Key Design Decisions

1. **Temporal train/test split** — users' last 20% of ratings as test set (realistic evaluation)
2. **Hybrid over single model** — CF handles collaborative signals, content handles cold-start items
3. **MMR re-ranking** — prevents filter bubbles (all action movies) with diversity penalty
4. **Two-tower NCF** — separates user/item spaces, enables efficient ANN retrieval
5. **TMDB cache** — API rate limits make caching mandatory for large catalogs
6. **Normalized scores** — CF scores (rating estimates ~0-5) and NCF scores (0-1) need normalization before fusion
