"""Tests for src.models.hf_llm_client — HuggingFaceClient adapter.

Unit tests mock the ``transformers.pipeline`` factory so no real model is ever
downloaded. A slower integration test (``scripts/demo_llm_rerank.py``) runs
the real tiny model end-to-end and is invoked by hand, not by pytest.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from src.models.hf_llm_client import HuggingFaceClient
from src.models.llm_rerank import LlmClient


class _FakePipeline:
    """Stand-in for ``transformers.pipeline('text-generation', ...)``."""

    def __init__(
        self,
        *,
        response: str | list[dict[str, str]] | Exception = "",
    ) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> list[dict[str, Any]]:
        self.calls.append({"messages": messages, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return [{"generated_text": self.response}]


def _pipeline_factory(fake: _FakePipeline) -> Callable[..., _FakePipeline]:
    def _factory(task: str, model: str, **_: object) -> _FakePipeline:
        assert task == "text-generation"
        assert model
        return fake

    return _factory


def test_client_satisfies_the_llm_client_protocol() -> None:
    fake = _FakePipeline(response="{}")
    client: LlmClient = HuggingFaceClient(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        _pipeline_factory=_pipeline_factory(fake),
    )
    assert client.complete("hi", timeout=0.1) == "{}"


def test_complete_forwards_the_prompt_as_a_user_message() -> None:
    fake = _FakePipeline(response='{"ranking": []}')
    client = HuggingFaceClient(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        _pipeline_factory=_pipeline_factory(fake),
    )
    client.complete("re-rank these movies", timeout=1.0)

    assert fake.calls, "pipeline was not invoked"
    call = fake.calls[0]
    assert call["messages"][-1]["role"] == "user"
    assert call["messages"][-1]["content"] == "re-rank these movies"


def test_complete_returns_only_the_generated_tail_after_the_prompt() -> None:
    # Pipelines return the full chat; HuggingFaceClient must strip the prompt.
    fake = _FakePipeline(
        response=[
            {"role": "user", "content": "rerank"},
            {"role": "assistant", "content": '{"ranking": [{"movie_id": 1}]}'},
        ]
    )
    client = HuggingFaceClient(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        _pipeline_factory=_pipeline_factory(fake),
    )
    out = client.complete("rerank", timeout=1.0)
    assert '"movie_id": 1' in out


def test_complete_propagates_exceptions_so_the_reranker_can_fall_back() -> None:
    # gate: LlmReranker's contract is that any raise from the client triggers
    # the safe-fallback path — so HuggingFaceClient must NOT silently swallow.
    fake = _FakePipeline(response=RuntimeError("CUDA OOM"))
    client = HuggingFaceClient(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        _pipeline_factory=_pipeline_factory(fake),
    )
    with pytest.raises(RuntimeError, match="CUDA OOM"):
        client.complete("rerank", timeout=1.0)


def test_pipeline_is_constructed_lazily_and_cached() -> None:
    calls = {"count": 0}
    fake = _FakePipeline(response="{}")

    def counting_factory(task: str, model: str, **_: object) -> _FakePipeline:
        calls["count"] += 1
        return fake

    client = HuggingFaceClient(
        model="tiny", _pipeline_factory=counting_factory
    )
    # Factory not called at construction time (lazy init).
    assert calls["count"] == 0

    client.complete("a", timeout=0.1)
    client.complete("b", timeout=0.1)
    # Called once total — cached across calls.
    assert calls["count"] == 1


def test_model_argument_is_passed_through_to_the_pipeline() -> None:
    observed: dict[str, str] = {}

    def recording_factory(task: str, model: str, **_: object) -> _FakePipeline:
        observed["model"] = model
        return _FakePipeline(response="{}")

    client = HuggingFaceClient(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        _pipeline_factory=recording_factory,
    )
    client.complete("x", timeout=0.1)
    assert observed["model"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_max_new_tokens_is_forwarded_to_the_pipeline_call() -> None:
    fake = _FakePipeline(response="{}")
    client = HuggingFaceClient(
        model="tiny",
        max_new_tokens=42,
        _pipeline_factory=_pipeline_factory(fake),
    )
    client.complete("x", timeout=0.1)
    assert fake.calls[0]["max_new_tokens"] == 42
