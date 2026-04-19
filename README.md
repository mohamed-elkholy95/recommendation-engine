<div align="center">

# Hybrid Recommendation Engine

**A production-shaped movie recommender that fuses four models and re-ranks with a local LLM.**

Built from the ground up as a systems-engineering exercise — every layer is tested, typed, deterministic, and served behind a FastAPI HTTP interface.

[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-cu132-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/Gemma%204-E2B-4285F4?logo=google&logoColor=white)](https://huggingface.co/google/gemma-4-E2B-it)
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)](Dockerfile)

[![Tests](https://img.shields.io/badge/tests-148%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)](pyproject.toml)
[![mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](pyproject.toml)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-D7FF64)](https://docs.astral.sh/ruff/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)

**[Architecture & User Manual](ARCHITECTURE.md)** · **[Offline benchmark](#offline-benchmark)** · **[Live LLM output](#what-the-llm-actually-says)** · **[Quick start](#quick-start)**

</div>

---

## At a glance

| | |
|---|---|
| **Models shipped** | SVD · Koren-baseline ALS · TF-IDF content-based · Two-tower Neural CF · Weighted-fusion Hybrid with MMR · LLM re-ranker |
| **Phases delivered** | 9 of 9 (Phase 0 → Phase 9) + CI, CUDA auto-detect, persistence, HF adapter, comparison harness |
| **Tests** | **148 passing** (unit + integration), hand-computed expected values for every metric |
| **Coverage** | **96.70 %** (`pytest --cov-fail-under=85` hard-gated in CI) |
| **Type checking** | `mypy --strict`, 4 `NewType`s for ID spaces, zero bare `except:` |
| **Commits** | 48 atomic `feat(area): …` commits — every one ≤ ~200 lines, reviewable on its own |
| **Deploy target** | Multi-stage `Dockerfile`, non-root runtime, health-check, ~2 s warm boot |

---

## Why a hybrid?

No single recommender works on day one. Each of the four signals below has a known failure mode; blending them is what produces a system that holds up under real traffic:

| Signal | Strength | Weakness |
|---|---|---|
| **Collaborative filtering** (SVD / Koren ALS) | Recovers taste from co-ratings | Cold-start on new users / items |
| **Content-based** (TF-IDF + genre) | Works on day one for any item with metadata | Ignores community preferences |
| **Neural CF** (two-tower MLP, GPU) | Captures non-linear interactions | Hard to regularise, hard to explain |
| **LLM re-rank** (Gemma 4 E2B / Qwen3) | Diversity + explanations + long-tail boost | Latency, cost, needs guardrails |

Final score = **min-max-normalised weighted sum across models → MMR re-rank for diversity → optional LLM re-rank for explanation**. Every layer is swappable through a thin Protocol.

---

## Architecture

```
            ┌───────────┐    ┌──────────────┐    ┌─────────────┐
            │   Users   │    │    Movies    │    │   Ratings   │
            └─────┬─────┘    └──────┬───────┘    └──────┬──────┘
                  └──────────┬──────┴─────────────┬─────┘
                             ▼                    ▼
                   ┌─────────────────┐    ┌──────────────────┐
                   │  load_movielens │───▶│     RawData      │
                   │  (+ SHA-256)    │    │ ratings, movies  │
                   └─────────────────┘    └────────┬─────────┘
                                                   │ preprocess()
                                                   ▼
                                  ┌──────────────────────────────┐
                                  │  k-core + id maps + split    │
                                  │  → PreprocessedData          │
                                  └────┬───────────┬───────────┬─┘
                                       │           │           │
  ┌────────────────────────────────────┘           │           └─────────────────────┐
  ▼                           ▼                    ▼                                 ▼
┌───────────────┐    ┌──────────────────┐    ┌──────────────────┐           ┌──────────────────┐
│   SvdModel    │    │    AlsModel      │    │  ContentModel    │           │    NcfModel      │
│  (scipy svds) │    │ (Koren baseline) │    │ (TF-IDF + profile│           │  (two-tower MLP, │
│               │    │                  │    │    + cosine)     │           │   CUDA auto)     │
└───────┬───────┘    └────────┬─────────┘    └────────┬─────────┘           └────────┬─────────┘
        │                     │                       │                              │
        └─────────────────┬───┴───────────────┬───────┴──────────────────────────────┘
                          ▼                   ▼
                  ┌──────────────────────────────────┐
                  │            HybridModel           │
                  │     min-max + weighted fusion    │
                  │      + MMR diversity re-rank     │
                  └────────────┬─────────────────────┘
                               │ top-n
                               ▼
                  ┌──────────────────────────────────┐
                  │         LlmReranker (opt)        │
                  │   HuggingFaceClient (Gemma 4)    │
                  │   JSON parse · safe fallback     │
                  └────────────┬─────────────────────┘
                               ▼
                  ┌──────────────────────────────────┐
                  │  FastAPI · /recommend · /rate    │
                  │  · /health · Pydantic v2         │
                  └──────────────────────────────────┘
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module-by-module reference and runtime flow.

---

## What the LLM actually says

Real end-to-end output — hybrid top-10 for MovieLens user 1, re-ranked by **`google/gemma-4-E2B-it`** running locally on an RTX 5080:

```
LLM-reranked (with explanations):
   1. Dark Knight, The (2008)                     score=0.5552
      → Strong action/epic appeal, similar to Gladiator and Star Wars.
   2. Lord of the Rings: The Fellowship of the Ring (2001)  score=0.5246
      → High-stakes epic adventure, aligning with the scope of Star Wars and Gladiator.
   3. Aliens (1986)                               score=0.6467
      → Classic action/sci-fi, fitting the general action/adventure taste.
   4. Die Hard (1988)                             score=0.5982
      → Iconic action film, similar vibe to high-energy movies.
   5. Batman Forever (1995)                       score=0.5734
      → Action/adventure from the 90s, fitting the general appeal.
   ...
```

The LLM reshuffled "Dark Knight" and "Fellowship of the Ring" to the top after reading the user's rated-highly history — a qualitative win the plain hybrid doesn't get. Full pipeline: `hybrid.recommend()` → `_build_prompt()` → `HuggingFaceClient.complete()` → `_parse_ranking()` (handles markdown-fenced JSON) → merge with original scores. Any failure at any step falls back silently to the hybrid's order.

---

## Offline benchmark

`python scripts/compare_models.py --ncf-epochs 5` — fits every model on the train split and evaluates on the held-out test split. Real numbers (`ml-latest-small`, n=610 users × 3,650 items, Torch cu132 on RTX 5080):

| Model    | NDCG@10 | HR@10      | MAP@10 | fit+eval |
|----------|---------|------------|--------|----------|
| SVD      | 0.0461  | 0.2180     | 0.0208 | 0.1 s |
| ALS      | 0.0031  | 0.0213     | 0.0011 | 0.1 s |
| Content  | 0.0171  | 0.0951     | 0.0069 | 0.1 s |
| NCF      | 0.0247  | 0.1262     | 0.0108 | 0.2 s |
| **Hybrid** | **0.0443** | **0.2246** | **0.0196** | 2.1 s |

Hybrid wins `HR@10` — finds more relevant items in the top-10 than any single model. The Koren-baseline ALS (`r̂ = μ + b_u + b_i + x_u · y_i`) is the textbook-correct algorithm but *underperforms on leave-last-n-out* splits because popularity is the wrong signal for a user's most-recent ratings. That's an honest, documented limitation — the comparison is the point.

---

## Tech stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.12 · strict `mypy` · `ruff` lint + format |
| **Data** | NumPy · pandas · SciPy sparse · PyArrow |
| **Classical ML** | `scipy.sparse.linalg.svds` · hand-rolled Koren ALS · scikit-learn `TfidfVectorizer` |
| **Deep learning** | PyTorch (CUDA cu132 + CPU fallback) · `torch.nn.Embedding` + MLP · BCE loss · Adam |
| **LLM** | `transformers` · `accelerate` · `huggingface_hub` · default: Gemma 4 E2B ・ swappable via env var |
| **Serving** | FastAPI · Pydantic v2 · uvicorn · `httpx` test client |
| **Quality** | `pytest` · `pytest-asyncio` · `pytest-cov` · `pre-commit` · GitHub Actions CI |
| **Deploy** | Multi-stage `Dockerfile` (non-root, healthcheck) · `uv` for reproducible builds |
| **Reproducibility** | `set_all_seeds()` at every entry point · `torch.use_deterministic_algorithms(True)` · SHA-256-verified data fetches |

---

## Engineering principles

These are enforced **mechanically on every commit**, not aspirationally:

- **Hand-computed expected values** in every metric test. Range-only asserts (`0 ≤ x ≤ 1`) are forbidden — they hide math bugs.
- **Typed ID spaces.** Four distinct `NewType`s (`RawUserId`, `RawMovieId`, `UserIdx`, `ItemIdx`) make `mypy --strict` refuse `% len(...)`-style aliasing — the dominant class of bug in hand-rolled CF code.
- **Determinism by default.** Every training / evaluation entry point calls `set_all_seeds()`. CUDA determinism enabled. Runs are bit-for-bit reproducible given the same seed.
- **One concept per commit.** Each commit is a single `feat(area): imperative subject`, reviewable on its own.
- **Coverage gate ≥ 85 %**, hard-enforced in CI. Project sits at 96.70 %.
- **Two-tier size limits.** Modules ≤ 300 lines, functions ≤ 40 lines. Over the cap ⇒ two concepts, split.
- **One-file, one-concept modules.** Open `collaborative.py` and find only collaborative filtering. No dumping-ground utils.
- **Graceful degradation everywhere.** If the LLM times out, the hybrid still answers. If GPU is missing, NCF runs on CPU. If the model cache is absent, `serve.py` fits + saves.

Full style guide in [`CONTRIBUTING.md`](CONTRIBUTING.md). Full technical reference in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Quick start

```bash
git clone git@github.com:mohamed-elkholy95/recommendation-engine.git
cd recommendation-engine

# One-liner install (Python 3.12, strict type-checking + LLM deps + dev tools)
pip install -e ".[dev,llm]" --extra-index-url https://download.pytorch.org/whl/nightly/cu128
pre-commit install

# Serve the API (fits models on first boot, warm-starts from cache after)
python scripts/serve.py
# → http://localhost:8000/health, /recommend, /rate
```

### Request a recommendation

```bash
curl -X POST http://localhost:8000/recommend \
    -H 'Content-Type: application/json' \
    -d '{"user_id": 1, "n": 10}'
```

### Run the end-to-end demo (hybrid + real Gemma 4 LLM)

```bash
python scripts/demo_llm_rerank.py
# Override: RECO_LLM_MODEL=Qwen/Qwen3-1.7B USER_ID=42 N=20 python scripts/demo_llm_rerank.py
```

### Compare every model side-by-side

```bash
python scripts/compare_models.py --ncf-epochs 5
# Outputs the NDCG@10 / HR@10 / MAP@10 markdown table (see above).
```

### Full quality loop

```bash
ruff check .
ruff format --check .
mypy src tests
pytest --cov=src --cov-fail-under=85
```

---

## Project structure

```
.
├── README.md                     ← this file
├── ARCHITECTURE.md               ← module reference + runtime flow + user manual
├── CONTRIBUTING.md               ← setup, style, commit conventions
├── Dockerfile                    ← multi-stage, non-root, healthcheck
├── .github/workflows/ci.yml      ← mirrors pre-commit
├── pyproject.toml                ← single source of truth for deps + tool configs
├── src/
│   ├── data/                     ← loader (SHA-256), preprocessor (k-core, temporal split)
│   ├── models/                   ← svd + als + content + ncf + hybrid + llm_rerank
│   ├── api/                      ← FastAPI app factory, schemas, rating store
│   ├── evaluation.py             ← NDCG, HR, AP @ k (hand-computed tests)
│   ├── benchmark.py              ← evaluate_ranking harness
│   ├── persistence.py            ← save / load (torch-aware)
│   └── utils/seed.py             ← set_all_seeds()
├── tests/                        ← 1:1 mirror of src/, 148 tests
└── scripts/
    ├── serve.py                  ← prod entry point, warm-start cache
    ├── compare_models.py         ← offline NDCG / HR / MAP table
    ├── demo_llm_rerank.py        ← end-to-end Gemma 4 demo
    └── e2e_test.sh               ← real uvicorn + curl smoke (CI)
```

---

## Roadmap — shipped

- [x] **Phase 0** · Tooling & style — ruff, mypy, pytest, pre-commit, CI scaffolding, determinism helper.
- [x] **Phase 1** · Data pipeline — MovieLens loader (SHA-256 verified), k-core, per-user temporal split.
- [x] **Phase 2** · Collaborative baselines — truncated SVD + Koren-baseline ALS on real `ml-latest-small`.
- [x] **Phase 3** · Content-based retrieval — TF-IDF + rating-weighted user profile + cosine similarity.
- [x] **Phase 4** · Neural CF — two-tower MLP, CUDA auto-detect, BCE on implicit positives + sampled negatives.
- [x] **Phase 5** · Hybrid fusion — per-model min-max normalise + weighted sum + MMR diversity re-rank.
- [x] **Phase 6** · API layer — FastAPI `/recommend`, `/rate`, `/health`; Pydantic v2; in-memory rating store.
- [x] **Phase 7** · LLM re-ranker — Protocol-based + real `HuggingFaceClient` + markdown-fence JSON parser + safe fallback.
- [x] **Phase 8** · Evaluation harness — `evaluate_ranking` + side-by-side comparison script.
- [x] **Phase 9** · Deployment — multi-stage `Dockerfile`, warm-start `scripts/serve.py`, `scripts/e2e_test.sh` real-HTTP smoke.

### Plus the polish round

- [x] GitHub Actions CI mirroring every pre-commit hook.
- [x] CUDA auto-detection in `NcfModel` (falls back to CPU cleanly).
- [x] Pickle persistence with warm-start — first boot ~15 s, warm boot ~2 s.
- [x] Real HF adapter (`HuggingFaceClient`) — default Gemma 4 E2B, swappable via `RECO_LLM_MODEL`.
- [x] Hardened JSON extractor — tolerates markdown code fences and surrounding prose.
- [x] Full `ARCHITECTURE.md` — module reference + runtime flow + user manual + troubleshooting.

---

## License

MIT. See [`LICENSE`](LICENSE) (to be added).

---

<div align="center">

**Built by [Mohamed Elkholy](mailto:melkholy@techmatrix.com)** · [github.com/mohamed-elkholy95](https://github.com/mohamed-elkholy95)

*Questions, feedback, or interested in hiring? I'd love to hear from you.*

</div>
