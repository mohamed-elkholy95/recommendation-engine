"""Tests for src.models.llm_rerank — the LLM re-ranker stays graceful under failure."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from src.data.ids import ItemIdx
from src.models.llm_rerank import LlmClient, LlmReranker, RerankedItem


class _ScriptedClient:
    """Test double that returns a pre-scripted response or raises an error."""

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[str] = []

    def complete(self, prompt: str, *, timeout: float) -> str:
        self.calls.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _catalogue() -> dict[ItemIdx, str]:
    return {
        ItemIdx(0): "Alpha",
        ItemIdx(1): "Bravo",
        ItemIdx(2): "Charlie",
    }


def _candidates() -> list[tuple[ItemIdx, float]]:
    return [
        (ItemIdx(0), 0.9),
        (ItemIdx(1), 0.8),
        (ItemIdx(2), 0.7),
    ]


def test_rerank_returns_items_in_the_order_the_llm_specifies() -> None:
    llm_response = json.dumps(
        {
            "ranking": [
                {"movie_id": 2, "reason": "matches mystery preference"},
                {"movie_id": 0, "reason": "solid fallback pick"},
                {"movie_id": 1, "reason": "lower priority"},
            ]
        }
    )
    reranker = LlmReranker(client=_ScriptedClient(llm_response), timeout=0.5)

    reranked = reranker.rerank(
        candidates=_candidates(),
        catalogue=_catalogue(),
        user_context="Likes mystery films.",
    )

    assert [int(r.item_idx) for r in reranked] == [2, 0, 1]
    assert reranked[0].explanation == "matches mystery preference"


def test_rerank_preserves_scores_from_the_original_candidates() -> None:
    llm_response = json.dumps(
        {"ranking": [{"movie_id": 1, "reason": ""}, {"movie_id": 0, "reason": ""}]}
    )
    reranker = LlmReranker(client=_ScriptedClient(llm_response), timeout=0.5)

    reranked = reranker.rerank(
        candidates=_candidates(), catalogue=_catalogue(), user_context=""
    )

    # gate: the LLM only re-orders; scores must come from the fused hybrid output.
    by_item = {int(r.item_idx): r.score for r in reranked}
    assert by_item[0] == pytest.approx(0.9)
    assert by_item[1] == pytest.approx(0.8)


def test_rerank_falls_back_to_input_order_when_llm_raises() -> None:
    reranker = LlmReranker(
        client=_ScriptedClient(RuntimeError("network down")), timeout=0.5
    )

    reranked = reranker.rerank(
        candidates=_candidates(), catalogue=_catalogue(), user_context=""
    )

    assert [int(r.item_idx) for r in reranked] == [0, 1, 2]
    assert all(r.explanation == "" for r in reranked)


def test_rerank_falls_back_when_llm_returns_non_json() -> None:
    reranker = LlmReranker(client=_ScriptedClient("not json"), timeout=0.5)

    reranked = reranker.rerank(
        candidates=_candidates(), catalogue=_catalogue(), user_context=""
    )

    assert [int(r.item_idx) for r in reranked] == [0, 1, 2]


def test_rerank_extracts_json_from_markdown_code_fence() -> None:
    # concept: small LLMs like to wrap JSON in ```json ... ``` markdown fences.
    # Extraction must find the payload inside the fence.
    llm_response = (
        "Here is the re-ranked list:\n\n"
        "```json\n"
        '{"ranking": [{"movie_id": 2, "reason": "best match"}, {"movie_id": 0, "reason": "runner-up"}]}\n'
        "```\n\nHope this helps!"
    )
    reranker = LlmReranker(client=_ScriptedClient(llm_response), timeout=0.5)

    reranked = reranker.rerank(
        candidates=_candidates(), catalogue=_catalogue(), user_context=""
    )

    assert [int(r.item_idx) for r in reranked] == [2, 0, 1]


def test_rerank_extracts_json_from_prose_wrapped_response() -> None:
    # concept: some models prefix commentary before the JSON and append more
    # after. Extraction must find the outermost {...} span.
    llm_response = (
        "I will sort these: the user seems to like drama.\n"
        '{"ranking": [{"movie_id": 1, "reason": "drama"}]}\n'
        "Let me know if you want another pass."
    )
    reranker = LlmReranker(client=_ScriptedClient(llm_response), timeout=0.5)

    reranked = reranker.rerank(
        candidates=_candidates(), catalogue=_catalogue(), user_context=""
    )

    assert int(reranked[0].item_idx) == 1


def test_rerank_drops_unknown_movie_ids_and_appends_missing_ones() -> None:
    # concept: the LLM can hallucinate IDs not in the candidate set — filter
    # them. Any original candidate the LLM omitted is appended at the end so
    # the caller still gets exactly the candidates it passed in.
    llm_response = json.dumps(
        {
            "ranking": [
                {"movie_id": 999, "reason": "hallucination"},
                {"movie_id": 1, "reason": "good pick"},
            ]
        }
    )
    reranker = LlmReranker(client=_ScriptedClient(llm_response), timeout=0.5)

    reranked = reranker.rerank(
        candidates=_candidates(), catalogue=_catalogue(), user_context=""
    )

    assert [int(r.item_idx) for r in reranked] == [1, 0, 2]


def test_rerank_is_a_noop_for_an_empty_candidate_list() -> None:
    reranker = LlmReranker(client=_ScriptedClient(""), timeout=0.5)
    assert reranker.rerank(candidates=[], catalogue=_catalogue(), user_context="") == []


def test_llm_client_protocol_covers_scripted_double() -> None:
    # concept: scripted double must satisfy the exported Protocol — gives us
    # confidence the real Anthropic/OpenAI adapters will fit the same socket.
    client: LlmClient = _ScriptedClient("{}")
    assert client.complete("prompt", timeout=0.1) == "{}"


def test_reranked_items_are_a_stable_sequence() -> None:
    reranker = LlmReranker(
        client=_ScriptedClient(json.dumps({"ranking": []})), timeout=0.5
    )
    out: Sequence[RerankedItem] = reranker.rerank(
        candidates=_candidates(), catalogue=_catalogue(), user_context=""
    )
    # Simple sanity that we got a real sequence back, not a generator.
    first = out[0]
    assert isinstance(first, RerankedItem)


def test_scripted_client_records_prompt_contains_titles_and_context() -> None:
    client = _ScriptedClient(json.dumps({"ranking": []}))
    reranker = LlmReranker(client=client, timeout=0.5)
    reranker.rerank(
        candidates=_candidates(),
        catalogue=_catalogue(),
        user_context="Loves sci-fi.",
    )

    # Prompt must include every candidate title and the user context so the
    # caller can't accidentally strip this information.
    prompt = client.calls[0]
    assert "Alpha" in prompt and "Bravo" in prompt and "Charlie" in prompt
    assert "Loves sci-fi" in prompt


def test_missing_optional_kwargs_in_protocol_works(monkeypatch: Any) -> None:
    # concept: real clients often take extra kwargs — Protocol must not reject
    # a client that accepts additional keyword arguments beyond what's declared.
    class _ExtendedClient:
        def complete(self, prompt: str, *, timeout: float, extra: int = 0) -> str:
            return json.dumps({"ranking": []})

    reranker = LlmReranker(client=_ExtendedClient(), timeout=0.1)
    out = reranker.rerank(
        candidates=_candidates(), catalogue=_catalogue(), user_context=""
    )
    assert len(out) == 3
