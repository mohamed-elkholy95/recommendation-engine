"""Side-by-side NDCG / HR / MAP comparison of every model on ml-latest-small.

Fits SvdModel, AlsModel, ContentModel, and NcfModel on the preprocessor's
train split, then evaluates each (plus a HybridModel that fuses SVD + content
+ NCF) against the test split using ``evaluate_ranking``. Prints a markdown
table so the output can be pasted directly into the README.

Usage:
    python scripts/compare_models.py           # defaults: ml-latest-small, k=10
    python scripts/compare_models.py --k 20    # deeper cutoff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from src.benchmark import evaluate_ranking
from src.data.loader import load_movielens
from src.data.preprocessor import preprocess
from src.models.collaborative import AlsModel, SvdModel
from src.models.content_based import ContentModel
from src.models.hybrid import HybridModel
from src.models.neural_cf import NcfModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="ml-latest-small")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ncf-epochs", type=int, default=5)
    args = parser.parse_args()

    def stage(msg: str) -> None:
        print(f"[compare] {msg}", flush=True)

    stage(f"loading {args.dataset}")
    raw = load_movielens(args.dataset, data_dir=args.data_dir)  # type: ignore[arg-type]
    pre = preprocess(raw)
    stage(
        f"n_users={pre.n_users} n_items={pre.n_items} train={len(pre.train)} test={len(pre.test)}"
    )

    stage("fitting SvdModel")
    svd = SvdModel(n_factors=32, seed=42)
    svd.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)

    stage("fitting AlsModel")
    als = AlsModel(n_factors=32, n_iter=15, reg=0.1, seed=42)
    als.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)

    stage("fitting ContentModel")
    content = ContentModel()
    content.fit(raw.movies, pre)

    stage(f"fitting NcfModel(n_epochs={args.ncf_epochs})")
    ncf = NcfModel(n_factors=16, n_epochs=args.ncf_epochs, seed=42)
    ncf.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)

    stage("assembling HybridModel")
    hybrid = HybridModel(
        collaborative=svd,
        content=content,
        neural=ncf,
        n_items=pre.n_items,
    )

    models = [
        ("SVD", svd.recommend),
        ("ALS", als.recommend),
        ("Content", content.recommend),
        ("NCF", ncf.recommend),
        ("Hybrid", hybrid.recommend),
    ]
    k = args.k

    print(f"\n| Model    | NDCG@{k} | HR@{k} | MAP@{k} | sec |")
    print("|----------|---------|--------|---------|-----|")
    for name, recommend_fn in models:
        start = perf_counter()
        metrics = evaluate_ranking(recommend_fn, pre.test, k=k)
        elapsed = perf_counter() - start
        print(
            f"| {name:<8s} | {metrics[f'ndcg@{k}']:.4f}  | "
            f"{metrics[f'hit_rate@{k}']:.4f} | {metrics[f'map@{k}']:.4f}  | "
            f"{elapsed:>3.1f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
