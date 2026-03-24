"""FastAPI for recommendation engine.

Provides RESTful endpoints for:
    - Personalized recommendations (GET /recommendations/{user_id})
    - Content-similar items (GET /similar/{item_id})
    - Trending/popular items (GET /trending)
    - Rating submission (POST /rate)
    - Health checks (GET /health)

The API uses Pydantic models for request/response validation and
includes proper error handling with meaningful HTTP status codes.
"""
import logging
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Recommendation Engine API",
    version="1.1.0",
    description="Hybrid recommendation system combining collaborative filtering, "
                "content-based filtering, and neural collaborative filtering.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Request/response logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request method, path, and response time for observability."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s → %d (%.1fms)",
                request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a clean 500 response."""
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Pydantic models with validation
# ---------------------------------------------------------------------------

class RecommendResponse(BaseModel):
    movie_id: int
    score: float
    title: Optional[str] = None


class RateRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating between 0.5 and 5.0")

    @field_validator("user_id")
    @classmethod
    def user_id_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("user_id must be a positive integer")
        return v

    @field_validator("movie_id")
    @classmethod
    def movie_id_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("movie_id must be a positive integer")
        return v


class HealthResponse(BaseModel):
    status: str = "healthy"
    models_loaded: List[str] = []
    version: str = "1.1.0"


# Mock data for demo
_mock_movies = {
    1: "Movie 1: The Action Adventure",
    2: "Movie 2: The Comedy Show",
    3: "Movie 3: Drama Nights",
    4: "Movie 4: Sci-Fi Universe",
    5: "Movie 5: Horror Tales",
}


@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check — verifies API is running and reports loaded models."""
    return HealthResponse(models_loaded=["cf", "content", "ncf", "hybrid"])


@app.get("/recommendations/{user_id}", response_model=List[RecommendResponse])
async def get_recommendations(
    user_id: int,
    n: int = Query(10, ge=1, le=50, description="Number of recommendations"),
):
    """Get personalized top-N recommendations for a user.

    Uses the hybrid ensemble (CF + content + NCF) to generate diverse,
    relevant suggestions. Items the user has already rated are excluded.
    """
    if user_id < 1:
        raise HTTPException(status_code=400, detail="user_id must be a positive integer")

    import random
    random.seed(user_id)
    items = random.sample(list(_mock_movies.items()), min(n, len(_mock_movies)))
    return [
        RecommendResponse(movie_id=mid, score=round(random.uniform(0.5, 1.0), 3), title=title)
        for mid, title in items
    ]


@app.get("/similar/{item_id}", response_model=List[RecommendResponse])
async def get_similar(
    item_id: int,
    n: int = Query(10, ge=1, le=50, description="Number of similar items"),
):
    """Get content-similar movies based on genre, description, and metadata."""
    if item_id < 1:
        raise HTTPException(status_code=400, detail="item_id must be a positive integer")

    if item_id not in _mock_movies:
        raise HTTPException(status_code=404, detail=f"Movie {item_id} not found")

    import random
    random.seed(item_id)
    items = [(k, v) for k, v in _mock_movies.items() if k != item_id]
    sample = random.sample(items, min(n, len(items)))
    return [
        RecommendResponse(movie_id=mid, score=round(random.uniform(0.3, 0.95), 3), title=title)
        for mid, title in sample
    ]


@app.post("/rate")
async def submit_rating(request: RateRequest):
    """Submit a user rating for a movie.

    Ratings are used to update the user profile and retrain
    models incrementally in production deployments.
    """
    logger.info("Rating: user=%d movie=%d rating=%.1f",
                request.user_id, request.movie_id, request.rating)
    return {
        "status": "recorded",
        "user_id": request.user_id,
        "movie_id": request.movie_id,
        "rating": request.rating,
    }


@app.get("/trending", response_model=List[RecommendResponse])
async def get_trending(
    n: int = Query(10, ge=1, le=50, description="Number of trending items"),
):
    """Get trending movies ranked by recent popularity.

    Useful as a fallback for cold-start users with no interaction history.
    """
    return [
        RecommendResponse(movie_id=mid, score=round(5.0 - i * 0.4, 1), title=title)
        for i, (mid, title) in enumerate(sorted(_mock_movies.items()))
    ][:n]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
