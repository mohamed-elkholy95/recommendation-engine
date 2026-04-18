"""In-memory rating store — the minimum surface needed by the /rate endpoint.

Production deployments swap this for a durable store (e.g. a write-through
Kafka topic or a database). For now we keep a list so /rate is testable in
isolation and /recommend has nothing to wait on.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InMemoryRatingStore:
    """Collects rating submissions during the process lifetime."""

    _records: list[tuple[int, int, float]] = field(default_factory=list)

    def add(self, user_id: int, movie_id: int, rating: float) -> None:
        self._records.append((user_id, movie_id, rating))

    def count(self) -> int:
        return len(self._records)

    def records(self) -> list[tuple[int, int, float]]:
        return list(self._records)
