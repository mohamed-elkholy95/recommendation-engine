"""End-to-end demo: hybrid recommender + real Hugging Face tiny-model re-ranker.

Loads ml-latest-small, reuses the warm-start model cache under
``$RECO_MODELS_DIR`` (default ``models/``) if present, builds a
HybridModel, pulls the top-``n`` recommendations for a single user, then
re-ranks them with a real Hugging Face tiny model via ``HuggingFaceClient``
and ``LlmReranker``. Prints a side-by-side of the pre-rerank and post-rerank
orderings along with the LLM's explanations.

Usage:
    python scripts/demo_llm_rerank.py
    RECO_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct USER_ID=1 N=10 python scripts/demo_llm_rerank.py

Defaults to ``Qwen/Qwen2.5-0.5B-Instruct`` — small enough to load on CPU or
a laptop GPU in a few seconds. Any instruction-tuned HF model should work.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.data.ids import ItemIdx, RawUserId
from src.data.loader import load_movielens
from src.data.preprocessor import preprocess
from src.models.collaborative import SvdModel
from src.models.content_based import ContentModel
from src.models.hf_llm_client import HuggingFaceClient
from src.models.hybrid import HybridModel
from src.models.llm_rerank import LlmReranker
from src.models.neural_cf import NcfModel
from src.persistence import load_model, save_model


def main() -> int:
    dataset = os.environ.get("RECO_DATASET", "ml-latest-small")
    data_dir = Path(os.environ.get("RECO_DATA_DIR", "data/raw"))
    models_dir = Path(os.environ.get("RECO_MODELS_DIR", "models"))
    model_name = os.environ.get("RECO_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    raw_user_id = int(os.environ.get("USER_ID", "1"))
    n = int(os.environ.get("N", "10"))

    def stage(msg: str) -> None:
        print(f"[demo] {msg}", flush=True)

    stage(f"loading {dataset}")
    raw = load_movielens(dataset, data_dir=data_dir)  # type: ignore[arg-type]
    pre = preprocess(raw)

    svd_path = models_dir / "svd.pkl"
    content_path = models_dir / "content.pkl"
    ncf_path = models_dir / "ncf.pkl"

    if svd_path.exists() and content_path.exists() and ncf_path.exists():
        stage(f"reloading cached models from {models_dir}")
        svd: SvdModel = load_model(svd_path)
        content: ContentModel = load_model(content_path)
        neural: NcfModel = load_model(ncf_path)
    else:
        stage("fitting models (first run)")
        svd = SvdModel(n_factors=32, seed=42)
        svd.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)
        content = ContentModel()
        content.fit(raw.movies, pre)
        neural = NcfModel(n_factors=16, n_epochs=3, seed=42)
        neural.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)
        save_model(svd, svd_path)
        save_model(content, content_path)
        save_model(neural, ncf_path)
        stage(f"cached to {models_dir}")

    hybrid = HybridModel(collaborative=svd, content=content, neural=neural, n_items=pre.n_items)

    if RawUserId(raw_user_id) not in pre.user_map:
        stage(f"user_id {raw_user_id} not in user_map; picking the first available")
        raw_user_id = int(next(iter(pre.user_map)))
    user_idx = pre.user_map[RawUserId(raw_user_id)]

    stage(f"hybrid top-{n} for user_id={raw_user_id}")
    candidates = hybrid.recommend(user_idx, n=n)

    reverse_item_map = {int(v): int(k) for k, v in pre.item_map.items()}
    title_by_raw_id = dict(
        zip(raw.movies["movieId"].astype(int), raw.movies["title"].astype(str), strict=True)
    )
    catalogue: dict[ItemIdx, str] = {
        item: title_by_raw_id.get(reverse_item_map[int(item)], "(unknown)")
        for item, _ in candidates
    }

    print("\nHybrid ranking (before LLM re-rank):")
    for rank, (item, score) in enumerate(candidates, start=1):
        print(f"  {rank:>2}. {catalogue[item]:<60s}  score={score:.4f}")

    stage(f"loading LLM client ({model_name})")
    # concept: tiny models front-load prose before the JSON; 800 tokens leaves
    # room for both a few paragraphs of analysis and a complete JSON block.
    client = HuggingFaceClient(model=model_name, max_new_tokens=800)
    reranker = LlmReranker(client=client, timeout=60.0)

    # concept: give the LLM just enough user context to meaningfully re-rank.
    user_rated = pre.train[pre.train["user_idx"] == int(user_idx)]
    top_seen = (
        user_rated.sort_values("rating", ascending=False)
        .head(5)["item_idx"]
        .map(lambda i: title_by_raw_id.get(reverse_item_map[int(i)], "(unknown)"))
        .tolist()
    )
    user_context = f"The user previously rated these movies highly: {', '.join(top_seen)}."

    stage("running LLM re-rank")
    reranked = reranker.rerank(
        candidates=candidates, catalogue=catalogue, user_context=user_context
    )

    has_any_reason = any(item.explanation for item in reranked)
    print("\nLLM-reranked (with explanations):")
    for rank, item in enumerate(reranked, start=1):
        title = catalogue[item.item_idx]
        if item.explanation:
            reason = item.explanation
        elif has_any_reason:
            reason = "(LLM did not rank this item; preserved in original order)"
        else:
            reason = "(LLM declined or failed — full fallback to hybrid order)"
        print(f"  {rank:>2}. {title:<60s}  score={item.score:.4f}")
        print(f"       → {reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
