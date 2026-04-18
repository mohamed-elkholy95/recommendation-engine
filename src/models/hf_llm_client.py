"""Hugging Face ``transformers`` adapter for the LLM re-ranker.

Satisfies the ``LlmClient`` Protocol from ``src.models.llm_rerank`` by wrapping
``transformers.pipeline('text-generation', model=...)``. Model loading is lazy
(deferred until the first ``complete`` call) so importing this module is free
and the pipeline is cached across subsequent calls.

Install the optional ``llm`` extra to bring ``transformers`` + ``accelerate``
in::

    pip install -e ".[llm]"

Any tiny instruction-tuned model works. On a laptop-class CPU, defaults that
converge quickly: ``Qwen/Qwen2.5-0.5B-Instruct`` (~1 GB, JSON-friendly) or
``TinyLlama/TinyLlama-1.1B-Chat-v1.0`` (~2 GB). On GPU they run in under a
second per re-rank.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

PipelineFactory = Callable[..., Any]


def _default_pipeline_factory(task: str, model: str, **kwargs: object) -> Any:
    # concept: deferred import so that importing this module doesn't pull in
    # the ~1 GB torch+transformers dependency tree when the caller never
    # actually instantiates a HuggingFaceClient.
    from transformers import pipeline  # type: ignore[import-not-found,unused-ignore]

    # transformers.pipeline is overloaded per task; the str-arg path isn't
    # visible to mypy for every overload, so silence the overload-resolution
    # complaint and trust the runtime.
    return pipeline(task, model=model, **kwargs)  # type: ignore[call-overload]


@dataclass
class HuggingFaceClient:
    """``LlmClient`` implementation backed by a local ``transformers`` pipeline.

    Attributes:
        model: HF model id, e.g. ``"Qwen/Qwen2.5-0.5B-Instruct"``.
        max_new_tokens: Cap on generated tokens per call. Guards latency more
            than the ``timeout`` arg (which this adapter ignores — HF pipelines
            don't expose a clean cancel hook; bound generation with this
            instead).
        device: Torch device string passed to the pipeline (``"auto"`` uses
            CUDA when available, falls back to CPU). Ignored when a custom
            ``_pipeline_factory`` is supplied for tests.
    """

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 512
    device: str = "auto"
    _pipeline_factory: PipelineFactory = field(default=_default_pipeline_factory)
    _pipeline: Any | None = field(init=False, default=None)

    def complete(self, prompt: str, *, timeout: float) -> str:
        """Generate a completion for ``prompt``. ``timeout`` is advisory only."""
        del timeout  # HF pipelines don't expose cancel — guard latency via max_new_tokens.
        pipe = self._ensure_pipeline()
        messages = [{"role": "user", "content": prompt}]
        raw = pipe(messages, max_new_tokens=self.max_new_tokens)
        return _extract_generated_text(raw)

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory(
                "text-generation", model=self.model, device_map=self.device
            )
        return self._pipeline


def _extract_generated_text(raw: Any) -> str:
    # transformers returns [{"generated_text": <str>}] for a raw prompt and
    # [{"generated_text": [chat messages]}] when the input is a message list.
    if not isinstance(raw, list) or not raw:
        return ""
    first = raw[0]
    if not isinstance(first, dict):
        return ""
    generated = first.get("generated_text")
    if isinstance(generated, str):
        return generated
    if isinstance(generated, list) and generated:
        last = generated[-1]
        if isinstance(last, dict):
            content = last.get("content")
            if isinstance(content, str):
                return content
    return ""
