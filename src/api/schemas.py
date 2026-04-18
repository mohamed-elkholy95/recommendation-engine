"""Pydantic v2 request / response schemas for the HTTP API.

All public payloads pass through these types — pandas / numpy objects never
cross the API boundary (CODE_STYLE §5.6).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class RecommendRequest(BaseModel):
    user_id: int = Field(ge=0, description="Raw MovieLens userId")
    n: int = Field(gt=0, le=200, description="Number of items to recommend")


class RecommendedItem(BaseModel):
    movie_id: int
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    items: list[RecommendedItem]


class RateRequest(BaseModel):
    user_id: int = Field(ge=0)
    movie_id: int = Field(ge=0)
    rating: float = Field(ge=0.5, le=5.0)


class RateResponse(BaseModel):
    accepted: bool
    stored_count: int
