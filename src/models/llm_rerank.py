"""LLM re-ranker — adds a diversity / explainability layer on top of the hybrid.

The reranker is deliberately isolated from any specific LLM provider: it takes
a ``LlmClient`` protocol implementation, builds a JSON-in / JSON-out prompt,
and parses the response back into ``RerankedItem`` records. If the provider
raises, times out, or returns malformed JSON the reranker **silently falls
back to the original candidate order** — the hybrid recommender must stay
available even when the LLM is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from src.data.ids import ItemIdx


@dataclass(frozen=True, slots=True)
class RerankedItem:
    """A candidate after LLM re-ranking, carrying the explanation from the model."""

    item_idx: ItemIdx
    score: float
    explanation: str


class LlmClient(Protocol):
    """Minimal client interface — a ``complete`` method that returns a string."""

    def complete(self, prompt: str, *, timeout: float) -> str: ...


@dataclass
class LlmReranker:
    """Reorders the hybrid's top-N candidates using an LLM, with safe fallback."""

    client: LlmClient
    timeout: float = 2.0

    def rerank(
        self,
        *,
        candidates: list[tuple[ItemIdx, float]],
        catalogue: dict[ItemIdx, str],
        user_context: str,
    ) -> list[RerankedItem]:
        """Ask the LLM to reorder ``candidates`` based on ``user_context``.

        On any failure (exception, non-JSON response, missing schema) the
        original candidate order is preserved and explanations default to
        empty strings.
        """
        if not candidates:
            return []

        prompt = _build_prompt(
            candidates=candidates, catalogue=catalogue, user_context=user_context
        )
        try:
            raw = self.client.complete(prompt, timeout=self.timeout)
            ranking = _parse_ranking(raw)
        except Exception:
            return _fallback(candidates)

        if not ranking:
            return _fallback(candidates)

        return _merge_ranking_with_candidates(ranking, candidates=candidates)


def _build_prompt(
    *,
    candidates: list[tuple[ItemIdx, float]],
    catalogue: dict[ItemIdx, str],
    user_context: str,
) -> str:
    lines = [
        f"- movie_id {int(item)}: {catalogue.get(item, '(unknown)')}"
        for item, _ in candidates
    ]
    listing = "\n".join(lines)
    return (
        "You are re-ranking a list of candidate movies for a single user.\n\n"
        f"User context: {user_context}\n\n"
        f"Candidates:\n{listing}\n\n"
        "Return JSON of the form "
        '{"ranking": [{"movie_id": <int>, "reason": "<short>"}]}. '
        "Include every candidate exactly once; do not invent new movie_ids."
    )


def _parse_ranking(raw: str) -> list[tuple[int, str]]:
    payload = _extract_json_object(raw)
    if payload is None:
        return []
    ranking_field = payload.get("ranking") if isinstance(payload, dict) else None
    if not isinstance(ranking_field, list):
        return []

    out: list[tuple[int, str]] = []
    for entry in ranking_field:
        if not isinstance(entry, dict):
            continue
        movie_id = entry.get("movie_id")
        reason = entry.get("reason", "")
        if isinstance(movie_id, int) and isinstance(reason, str):
            out.append((movie_id, reason))
    return out


def _extract_json_object(raw: str) -> dict[str, object] | None:
    # concept: small LLMs wrap JSON in markdown fences and surrounding prose;
    # find the first balanced {...} block whose payload parses.
    for candidate in _candidate_json_blocks(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _candidate_json_blocks(raw: str) -> list[str]:
    blocks: list[str] = []

    # Try markdown code fences first (```json ... ``` or ``` ... ```).
    fence_start = raw.find("```")
    while fence_start != -1:
        after_fence = fence_start + 3
        if raw[after_fence:].lstrip().startswith("json"):
            after_fence = raw.index("\n", after_fence) + 1
        fence_end = raw.find("```", after_fence)
        if fence_end == -1:
            break
        blocks.append(raw[after_fence:fence_end].strip())
        fence_start = raw.find("```", fence_end + 3)

    # Then the outermost {...} span as a fallback.
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        blocks.append(raw[first_brace : last_brace + 1])

    return blocks


def _merge_ranking_with_candidates(
    ranking: list[tuple[int, str]],
    *,
    candidates: list[tuple[ItemIdx, float]],
) -> list[RerankedItem]:
    scores = {int(item): score for item, score in candidates}
    ordered: list[RerankedItem] = []
    seen: set[int] = set()
    for movie_id, reason in ranking:
        if movie_id not in scores or movie_id in seen:
            continue
        seen.add(movie_id)
        ordered.append(
            RerankedItem(
                item_idx=ItemIdx(movie_id),
                score=scores[movie_id],
                explanation=reason,
            )
        )

    # gate: every original candidate must be present — append any the LLM
    # dropped at the end in their original relative order.
    for item, score in candidates:
        if int(item) in seen:
            continue
        ordered.append(RerankedItem(item_idx=item, score=score, explanation=""))
    return ordered


def _fallback(candidates: list[tuple[ItemIdx, float]]) -> list[RerankedItem]:
    return [RerankedItem(item_idx=i, score=s, explanation="") for i, s in candidates]
