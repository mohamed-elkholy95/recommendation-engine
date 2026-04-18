"""FastAPI app factory for the hybrid recommender.

``create_app`` takes a fitted ``HybridModel`` plus the id maps produced by the
data pipeline. The app is stateless with respect to the caller — everything it
needs is passed in at construction time — so it is trivial to swap a tiny
fitted model in during tests and a ml-25m-scale one in production.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from src.api.deps import InMemoryRatingStore
from src.api.schemas import (
    HealthResponse,
    RateRequest,
    RateResponse,
    RecommendedItem,
    RecommendRequest,
    RecommendResponse,
)
from src.data.ids import ItemIdx, RawMovieId, RawUserId, UserIdx
from src.models.hybrid import HybridModel


def create_app(
    *,
    model: HybridModel,
    user_map: dict[RawUserId, UserIdx],
    item_map: dict[RawMovieId, ItemIdx],
    rating_store: InMemoryRatingStore,
) -> FastAPI:
    """Build a ready-to-serve FastAPI app around a fitted hybrid model."""
    app = FastAPI(title="Hybrid Recommendation Engine")
    app.state.model = model
    app.state.user_map = user_map
    app.state.item_map = item_map
    # concept: tests + responses need the reverse lookup (ItemIdx → movieId).
    app.state.reverse_item_map = {idx: raw for raw, idx in item_map.items()}
    app.state.rating_store = rating_store

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness probe — returns ``{"status": "ok"}`` whenever the app is up."""
        return HealthResponse(status="ok")

    @app.post("/recommend", response_model=RecommendResponse)
    async def recommend(req: RecommendRequest, request: Request) -> RecommendResponse:
        """Top-N items for a known user, fused across every component model."""
        state = request.app.state
        raw_user_id = RawUserId(req.user_id)
        if raw_user_id not in state.user_map:
            raise HTTPException(status_code=404, detail=f"user {req.user_id} not found")

        user_idx = state.user_map[raw_user_id]
        recs = state.model.recommend(user_idx, n=req.n)
        items = [
            RecommendedItem(
                movie_id=int(state.reverse_item_map[item_idx]),
                score=float(score),
            )
            for item_idx, score in recs
        ]
        return RecommendResponse(user_id=req.user_id, items=items)

    @app.post("/rate", response_model=RateResponse)
    async def rate(req: RateRequest, request: Request) -> RateResponse:
        """Accept a rating for a known (user, movie) and append it to the store."""
        state = request.app.state
        if RawUserId(req.user_id) not in state.user_map:
            raise HTTPException(status_code=404, detail=f"user {req.user_id} not found")
        if RawMovieId(req.movie_id) not in state.item_map:
            raise HTTPException(status_code=404, detail=f"movie {req.movie_id} not found")

        state.rating_store.add(req.user_id, req.movie_id, req.rating)
        return RateResponse(accepted=True, stored_count=state.rating_store.count())

    return app
