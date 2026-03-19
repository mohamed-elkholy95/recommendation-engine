"""FastAPI for recommendation engine."""
import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="Recommendation Engine API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class RecommendResponse(BaseModel):
    movie_id: int
    score: float
    title: Optional[str] = None


class RateRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float = Field(..., ge=0.5, le=5.0)


class HealthResponse(BaseModel):
    status: str = "healthy"
    models_loaded: List[str] = []


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
    return HealthResponse(models_loaded=["cf", "content", "ncf", "hybrid"])


@app.get("/recommendations/{user_id}", response_model=List[RecommendResponse])
async def get_recommendations(user_id: int, n: int = Query(10, ge=1, le=50)):
    """Get personalized top-N recommendations."""
    import random
    random.seed(user_id)
    items = random.sample(list(_mock_movies.items()), min(n, len(_mock_movies)))
    return [
        RecommendResponse(movie_id=mid, score=round(random.uniform(0.5, 1.0), 3), title=title)
        for mid, title in items
    ]


@app.get("/similar/{item_id}", response_model=List[RecommendResponse])
async def get_similar(item_id: int, n: int = Query(10, ge=1, le=50)):
    """Get content-similar movies."""
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
    """Submit a rating."""
    logger.info("Rating: user=%d movie=%d rating=%.1f", request.user_id, request.movie_id, request.rating)
    return {"status": "recorded", "user_id": request.user_id, "movie_id": request.movie_id}


@app.get("/trending", response_model=List[RecommendResponse])
async def get_trending(n: int = Query(10, ge=1, le=50)):
    """Get trending movies."""
    return [
        RecommendResponse(movie_id=mid, score=round(5.0 - i * 0.4, 1), title=title)
        for i, (mid, title) in enumerate(sorted(_mock_movies.items()))
    ][:n]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
