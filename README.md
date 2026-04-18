# Recommendation Engine

A hybrid movie recommendation engine combining **collaborative filtering**, **content-based retrieval**, and **neural collaborative filtering**, with an optional **LLM re-ranker** on top. Designed as a production-shaped portfolio project — every layer is tested, typed, deterministic, and served behind a FastAPI HTTP interface.

> **Status:** tooling and project scaffolding complete. Model layers are being implemented phase-by-phase.

---

## Why "hybrid"?

Pure collaborative filtering works well for warm users but collapses on the cold-start problem. Pure content-based methods handle new items but miss the taste signal hidden in co-rating patterns. Neural collaborative filtering can capture non-linear interactions but is easy to overfit. None of them alone is enough for a recommender that has to work on day one.

This project fuses three complementary signals and — optionally — asks an LLM to re-rank the top-N for diversity and explainability:

| Signal | Strength | Weakness |
|---|---|---|
| **Collaborative filtering** (matrix factorisation) | Taste signal from co-ratings. | Cold-start on new users/items. |
| **Content-based** (TF-IDF + genre embeddings) | Works on day one for any item with metadata. | Ignores community preferences. |
| **Neural CF** (two-tower MLP) | Captures non-linear interactions. | Needs careful regularisation; hard to explain. |
| **LLM re-rank** (optional) | Diversity, explanations, long-tail boosting. | Latency + cost; needs guardrails. |

The final score is a normalised weighted fusion across these signals, followed by **MMR** (Maximal Marginal Relevance) for diversity.

---

## Architecture

```
          ┌───────────┐   ┌──────────────┐   ┌─────────────┐
          │   Users   │   │   Movies     │   │   Ratings   │
          └─────┬─────┘   └──────┬───────┘   └──────┬──────┘
                └──────────┬─────┴───────────┬──────┘
                           ▼                 ▼
                   ┌──────────────┐   ┌──────────────┐
                   │ Preprocessor │──▶│ id maps,     │
                   │ (k-core,     │   │ train/val/   │
                   │  temporal)   │   │ test splits  │
                   └──────────────┘   └──────┬───────┘
                                             │
      ┌──────────────────────────┬───────────┼───────────┬──────────────────────┐
      ▼                          ▼           ▼           ▼                      ▼
┌───────────────┐  ┌──────────────────┐  ┌─────────┐  ┌─────────────┐  ┌─────────────────┐
│ Collaborative │  │ Content-Based    │  │ Neural  │  │ Hybrid      │  │ LLM Re-Ranker   │
│  (ALS/SVD)    │  │ (TF-IDF, genres) │  │   CF    │  │ fusion + MMR│  │ (optional)      │
└───────┬───────┘  └────────┬─────────┘  └────┬────┘  └──────┬──────┘  └────────┬────────┘
        │                   │                 │              │                  │
        └───────────────────┴─────────────────┴──────────────┴──────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │  FastAPI /      │
                                    │  recommend      │
                                    └─────────────────┘
```

---

## Tech stack

- **Python 3.12** with strict type checking ([`mypy --strict`](https://mypy.readthedocs.io/)).
- **Data:** MovieLens (ml-25m for training, ml-latest-small for dev iteration), NumPy, pandas.
- **Modelling:** scikit-learn (baseline CF), implicit / SciPy (ALS), PyTorch (neural CF).
- **Serving:** FastAPI + Pydantic v2 + asyncio. Models exported to ONNX for portable inference.
- **Quality:** [ruff](https://docs.astral.sh/ruff/) · [mypy](https://mypy-lang.org/) · [pytest](https://docs.pytest.org/) · [pre-commit](https://pre-commit.com/) · coverage gate ≥ 85 %.
- **Tooling:** [uv](https://docs.astral.sh/uv/) for environment and dependency management.

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone git@github.com:mohamed-elkholy95/recommendation-engine.git
cd recommendation-engine

# 2. Create the venv (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dev extras (ruff, mypy, pytest, pre-commit, ...)
pip install -e ".[dev]"
pre-commit install

# 4. Run the full quality loop
ruff check .
ruff format --check .
mypy src tests
pytest --cov=src --cov-fail-under=85
```

Using [`uv`](https://docs.astral.sh/uv/) is faster and does not require a system Python 3.12:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## Project structure

```
.
├── CONTRIBUTING.md            # setup, code style, commit conventions
├── README.md                  # this file
├── pyproject.toml             # single source of truth for deps and tool configs
├── .pre-commit-config.yaml    # hooks mirror CI
├── src/
│   ├── data/                  # loaders, preprocessors
│   ├── models/                # collaborative, content_based, neural_cf, hybrid
│   ├── api/                   # FastAPI app, schemas
│   ├── evaluation.py          # metrics (NDCG, MAP, diversity)
│   └── utils/
│       └── seed.py            # deterministic seeding helper
├── tests/                     # mirrors src/ 1:1
└── scripts/                   # training / evaluation entry points
```

---

## Roadmap

The project is delivered as a sequence of small, well-tested phases. Each phase lands with its own tests, metrics, and documentation.

- [x] **Phase 0 — Tooling & style.** ruff, mypy, pytest, pre-commit, CI-ready scaffolding, determinism helper.
- [x] **Phase 1 — Data pipeline.** MovieLens loaders, k-core filtering, temporal train/val/test splits, `user_map` / `item_map`.
- [x] **Phase 2 — Collaborative filtering baseline.** ALS + SVD baselines with offline NDCG@k / MAP@k.
- [x] **Phase 3 — Content-based retrieval.** TF-IDF over titles/overviews, genre embeddings.
- [x] **Phase 4 — Neural CF.** Two-tower MLP trained on implicit signals, ONNX export.
- [x] **Phase 5 — Hybrid fusion.** Normalised score fusion + MMR re-ranking for diversity.
- [x] **Phase 6 — API layer.** FastAPI service with `/recommend`, `/rate`, Prometheus metrics.
- [x] **Phase 7 — LLM re-ranker (optional).** Bounded-latency re-rank with explanations.
- [x] **Phase 8 — Evaluation harness.** Side-by-side comparison of every model on the same test split.
- [x] **Phase 9 — Deployment.** Docker image, container health checks, basic load test.

---

## Engineering principles

- **Hand-computed expected values** in every metric test — range-only asserts hide math bugs.
- **Typed ID spaces** (`UserIdx`, `RawMovieId`, ...) keep dense model indices and raw dataset IDs from silently aliasing.
- **Determinism by default** — every entry point calls `set_all_seeds()` so runs are reproducible bit-for-bit.
- **Strict mypy, zero bare `except:`.** The type checker is treated as a real reviewer, not decoration.
- **One concept per commit.** Each commit should be reviewable on its own.

Full style guide, testing standards, and commit conventions live in [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, code style, testing standards, and commit conventions.

---

## License

MIT. See `LICENSE` (to be added).
