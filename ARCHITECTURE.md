# Hybrid Recommendation Engine — Architecture & User Manual

> A single-file technical reference covering scope, design decisions, module
> layout, runtime flow, and every command you can run against the project.

---

## Table of contents

1. [Pitch](#1-pitch)
2. [Scope](#2-scope)
3. [Architecture at a glance](#3-architecture-at-a-glance)
4. [Phase-by-phase build log](#4-phase-by-phase-build-log)
5. [Directory layout](#5-directory-layout)
6. [Module reference](#6-module-reference)
7. [Runtime data flow](#7-runtime-data-flow)
8. [Quality Gates](#8-quality-gates)
9. [User manual — commands](#9-user-manual--commands)
10. [Testing guide](#10-testing-guide)
11. [Deployment](#11-deployment)
12. [Offline benchmark](#12-offline-benchmark)
13. [Extending the project](#13-extending-the-project)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Pitch

A production-shaped recommendation engine that fuses four complementary
signals — two collaborative-filtering baselines (SVD + Koren-ALS), a
content-based retriever (TF-IDF), and a neural collaborative filter
(two-tower MLP) — behind a single weighted-fusion + MMR re-ranker, then
optionally sends the top-N through a local LLM (Gemma 4 / Qwen) for
explanation-aware re-ranking.

Designed as a portfolio project. Every layer is:

- **Tested** — 148 pytest tests with hand-computed expected values.
- **Typed** — `mypy --strict`, four distinct `NewType`s for the ID spaces.
- **Deterministic** — single `set_all_seeds()` at every entry point, CUDA
  determinism enabled.
- **Served** — FastAPI + Pydantic v2, Dockerised, warm-start cache.

---

## 2. Scope

### In scope

| Area | Delivered |
|---|---|
| Data | MovieLens loader (download + SHA-256 verify + extract), k-core filter, per-user leave-last-n-out temporal split, typed id maps. |
| Models | Truncated-SVD baseline, Koren-baseline ALS, TF-IDF content-based, two-tower Neural CF (PyTorch, CUDA auto-detect). |
| Fusion | Min-max-normalised weighted score fusion + MMR diversity re-rank. |
| LLM | Protocol-based re-ranker with graceful fallback; real adapter via `transformers.pipeline` (default Gemma 4 E2B). |
| Serving | FastAPI app with `/health`, `/recommend`, `/rate`; in-memory rating store. |
| Persistence | Pickle-based save / load + warm-start cache on container boot. |
| Evaluation | Hand-computed-test ranking metrics (NDCG@k, HR@k, MAP@k) + side-by-side comparison harness. |
| Deployment | Multi-stage `Dockerfile` (non-root user, health-check), GitHub Actions CI, shell + pytest e2e tests. |

### Out of scope (explicit non-goals)

- Online learning / retraining from `/rate` submissions.
- Persistent rating store (swap `InMemoryRatingStore` for Kafka / Postgres
  in production).
- Multi-user API auth / rate limiting.
- Prometheus metrics / distributed tracing (flagged as a follow-up).
- Kaggle ingestion — MovieLens is fetched directly from grouplens.org and
  already public.
- ONNX export of the NCF model — Phase 4 uses the in-process PyTorch model;
  ONNX export is a serving concern to revisit alongside multi-worker
  deployment.

---

## 3. Architecture at a glance

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
                                  │  (train / val / test,        │
                                  │   user_map, item_map)        │
                                  └────┬───────────┬───────────┬─┘
                                       │           │           │
  ┌────────────────────────────────────┘           │           └─────────────────────┐
  ▼                           ▼                    ▼                                 ▼
┌───────────────┐    ┌──────────────────┐    ┌──────────────────┐           ┌──────────────────┐
│  SvdModel     │    │  AlsModel        │    │  ContentModel    │           │  NcfModel        │
│  (scipy svds) │    │  (Koren          │    │  (TF-IDF + rating│           │  (two-tower MLP, │
│               │    │   baseline)      │    │   -weighted user │           │   CUDA auto)     │
│               │    │                  │    │   profile)       │           │                  │
└───────┬───────┘    └────────┬─────────┘    └────────┬─────────┘           └────────┬─────────┘
        │                     │                       │                              │
        └─────────────────┬───┴───────────────┬───────┴──────────────────────────────┘
                          ▼                   ▼
                  ┌──────────────────────────────────┐
                  │  HybridModel                     │
                  │    min-max normalise per model   │
                  │    weighted sum (wc, ww, wn)     │
                  │    exclude_seen mask             │
                  │    top `pool_size` candidates    │
                  │    MMR re-rank using content     │
                  │      embeddings as sim kernel    │
                  └────────────┬─────────────────────┘
                               │ top-n candidates
                               ▼
                  ┌──────────────────────────────────┐
                  │  LlmReranker (optional)          │
                  │    build JSON prompt             │
                  │    HuggingFaceClient             │
                  │      (Gemma 4 / Qwen)            │
                  │    parse JSON out of fences      │
                  │    safe fallback on any error    │
                  └────────────┬─────────────────────┘
                               │
                               ▼
                  ┌──────────────────────────────────┐
                  │  FastAPI /recommend              │
                  │  Pydantic response: movie_id +   │
                  │  score (+ LLM explanation)       │
                  └──────────────────────────────────┘
```

### Key design decisions (and the tradeoffs)

- **Four models with divergent signal characteristics, fused late.** SVD
  captures warm-user taste; Content handles cold-start items; NCF models
  non-linear interactions; ALS (Koren) anchors popularity. Late fusion via
  min-max + weighted sum is simpler and more testable than an end-to-end
  stacked model; the tradeoff is that the fusion weights are a hyperparameter
  the project does not auto-tune.
- **MMR over a content-similarity kernel.** Diversity comes from
  content-based TF-IDF cosine distance — not from learned factors — so the
  diversity dimension is semantically interpretable ("don't show three
  Marvel movies in a row").
- **Hand-rolled Koren ALS instead of the `implicit` library.** Keeps the
  math inspectable and the file under 300 lines; `implicit` would hide the
  algorithm behind C++ bindings. The cost is lower ceiling performance on
  large datasets — the numbers on ml-latest-small are intentionally weak
  (see §12) and documented as such.
- **Typed ID spaces.** Four `NewType`s (`RawUserId`, `RawMovieId`,
  `UserIdx`, `ItemIdx`) make `mypy --strict` refuse `% len(...)`-style
  aliasing — the dominant class of bug in hand-rolled CF code.
- **Determinism by default.** `set_all_seeds()` at every script entry
  point; `torch.use_deterministic_algorithms(True, warn_only=True)`; ALS
  init seeded via `np.random.default_rng(seed)`. Runs are bit-for-bit
  reproducible given the same seed.
- **Pickle persistence, not ONNX.** `serve.py` fits → saves → reloads on
  next boot. ~30 s first boot, ~2 s warm boot. ONNX export exists as a
  follow-up once the NCF architecture stabilises.

---

## 4. Phase-by-phase build log

Every phase lands as its own set of commits with a `chore(phase-N): tick`
marker pointing at the `README.md` roadmap.

| # | Phase | Shipped |
|---|---|---|
| 0 | Tooling & style | `pyproject.toml`, `.pre-commit-config.yaml`, `set_all_seeds()`, strict ruff + mypy + pytest + 85 % coverage gate. |
| 1 | Data pipeline | `ids.py`, `loader.py`, `preprocessor.py`. Downloads ml-latest-small / ml-25m, verifies SHA-256, k-core-filters, builds id maps, splits train / val / test by timestamp. |
| 2 | CF baselines | `SvdModel` + `AlsModel` (Koren μ + b_u + b_i + x·y baseline). `src/evaluation.py` with NDCG@k / HR@k / AP@k. |
| 3 | Content-based | `ContentModel` (TF-IDF over title + genres, rating-weighted user profile). Extracted `src/models/_ranking.py` for shared top-k logic. |
| 4 | Neural CF | `NcfModel` — two-tower MLP on implicit positives (rating ≥ 4) + uniform-random negatives, Adam + BCE, CUDA auto-detect. |
| 5 | Hybrid fusion | `HybridModel` — min-max normalise each model's full score vector, weighted sum, MMR re-rank against content-embedding similarity. |
| 6 | API layer | `src/api/` — FastAPI app factory + Pydantic v2 schemas + `InMemoryRatingStore`. `/health`, `/recommend`, `/rate`. |
| 7 | LLM re-ranker | `LlmReranker` + `LlmClient` Protocol. Prompt construction + hardened JSON extraction from markdown fences / prose. |
| 8 | Evaluation harness | `evaluate_ranking(recommend_fn, test, k)` — provider-agnostic, used by the compare script for side-by-side numbers. |
| 9 | Deployment | Multi-stage `Dockerfile` (non-root user, healthcheck), `scripts/serve.py` warm-start entry point, `scripts/smoke_test.sh` + `scripts/e2e_test.sh` for real-HTTP checks. |

### Polish rounds on top of the 9 phases

- **CI** — `.github/workflows/ci.yml` mirrors the pre-commit hooks.
- **CUDA auto-detect in NCF** — `device="auto"` resolves to CUDA when
  available. Training on RTX 5080 via Torch cu132 drops the convergence
  test from 8 s → 3 s.
- **Koren baseline in ALS** — `μ + b_u + b_i + x_u · y_i`; numbers are
  honest (popularity doesn't help leave-last-n splits).
- **Persistence** — `src/persistence.py` + warm-start in `serve.py`.
- **Real HF LLM adapter** — `HuggingFaceClient` with lazy, cached pipeline
  construction. Default model: `google/gemma-4-E2B-it`.
- **Offline comparison script** — `scripts/compare_models.py` produces the
  README benchmark table in ~2 seconds of inference time.

---

## 5. Directory layout

```
.
├── ARCHITECTURE.md            # this file
├── CONTRIBUTING.md            # setup + style + commits
├── README.md                  # pitch + roadmap + offline benchmark
├── Dockerfile                 # multi-stage: builder → runtime (slim, non-root)
├── .dockerignore
├── .github/workflows/ci.yml   # ruff + mypy + pytest in CI
├── .pre-commit-config.yaml    # local mirror of CI checks
├── pyproject.toml             # deps + tool configs (one source of truth)
├── src/
│   ├── __init__.py
│   ├── benchmark.py           # evaluate_ranking + tune_hybrid_weights stub
│   ├── evaluation.py          # NDCG@k / HR@k / AP@k — binary relevance
│   ├── persistence.py         # save_model / load_model (pickle, torch-aware)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py             # create_app factory — all deps injected
│   │   ├── deps.py            # InMemoryRatingStore
│   │   └── schemas.py         # Pydantic v2 request / response
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ids.py             # UserIdx / ItemIdx / RawUserId / RawMovieId + dataclasses
│   │   ├── loader.py          # load_movielens(...) + SHA-256 checksums
│   │   └── preprocessor.py    # k-core, id maps, per-user temporal split
│   ├── models/
│   │   ├── __init__.py
│   │   ├── _ranking.py        # top_k_from_scores + fit-state helpers
│   │   ├── collaborative.py   # SvdModel + AlsModel
│   │   ├── content_based.py   # ContentModel (TF-IDF + user profile)
│   │   ├── hf_llm_client.py   # HuggingFaceClient (transformers adapter)
│   │   ├── hybrid.py          # HybridModel + min_max_normalise + mmr_rerank
│   │   ├── llm_rerank.py      # LlmReranker + LlmClient Protocol + JSON extractor
│   │   └── neural_cf.py       # NcfModel two-tower MLP
│   └── utils/
│       ├── __init__.py
│       └── seed.py            # set_all_seeds() — random + numpy + torch (+ CUDA)
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_benchmark.py
│   ├── test_e2e.py            # @pytest.mark.integration — full-stack TestClient
│   ├── test_evaluation.py
│   ├── test_persistence.py
│   ├── test_seed.py
│   ├── api/test_routes.py
│   ├── data/{conftest,test_ids,test_loader,test_preprocessor}.py
│   └── models/{test_als,test_collaborative,test_content_based,test_hf_llm_client,test_hybrid,test_llm_rerank,test_neural_cf}.py
└── scripts/
    ├── compare_models.py      # offline NDCG/HR/MAP for every model
    ├── demo_llm_rerank.py     # end-to-end hybrid + Gemma 4 demo
    ├── e2e_test.sh            # real uvicorn + curl smoke
    ├── serve.py               # production entry point (warm-start cache)
    └── smoke_test.sh          # GET /health + POST /recommend against a running server
```

---

## 6. Module reference

### `src/utils/seed.py`

- `set_all_seeds(seed: int = 42) -> None` — seeds Python `random`, NumPy,
  `PYTHONHASHSEED`, and (if importable) PyTorch CPU + CUDA. Also enables
  `torch.use_deterministic_algorithms(True, warn_only=True)`. Raises
  `ValueError` on negative seeds.

### `src/data/ids.py`

- **`UserIdx` / `ItemIdx` / `RawUserId` / `RawMovieId`** — `NewType(..., int)`.
  Kept distinct so `mypy --strict` refuses accidental aliasing.
- **`RawData`** — frozen dataclass holding `ratings` + `movies` DataFrames.
- **`PreprocessedData`** — frozen dataclass holding `train` / `val` / `test`
  splits, `user_map`, `item_map`, `n_users`, `n_items`. Every downstream
  model consumes this one object.

### `src/data/loader.py`

- `load_movielens(name, data_dir=Path("data/raw")) -> RawData` — downloads
  the published zip, verifies SHA-256 against the hard-coded `CHECKSUMS`
  dict (ml-latest-small + ml-25m), extracts, reads the CSVs with explicit
  dtypes (`int32` IDs, `float32` rating, `int64` timestamp). Skips the
  network step if the extracted files already exist.
- Raises: `ValueError` (unknown name), `NotADirectoryError` (path
  collision), `RuntimeError` (checksum mismatch).

### `src/data/preprocessor.py`

- `preprocess(raw, *, k=5, n_test=5, n_val=5) -> PreprocessedData` — the
  single public entry point. Runs four pure helpers in sequence:
  1. `_k_core_filter` — drop users with `< k_user` ratings / items with
     `< k_item` until stable. `k_user = max(k, 2*max(n_test, n_val) + 1)`
     so every survivor can be split cleanly.
  2. `_build_id_maps` — sorted-enumeration; deterministic given the input.
  3. `_apply_id_maps` — replace raw IDs with dense indices; enforce dtypes.
  4. `_per_user_split` — sort each user's rows by timestamp; last `n_test`
     → test, preceding `n_val` → val, rest → train.

### `src/evaluation.py`

- `hit_rate_at_k(recommended, relevant, k)` — 1.0 if any of the top-k hits.
- `ndcg_at_k(recommended, relevant, k)` — DCG / IDCG with binary relevance.
- `average_precision_at_k(recommended, relevant, k)` — per-user AP@k;
  MAP@k is the caller's mean across users.
- All three return `0.0` when `relevant` is empty and raise `ValueError`
  on `k < 1`. Tests hand-compute every fraction.

### `src/models/_ranking.py`

Private helpers shared by every model:

- `top_k_from_scores(scores, *, seen, n, exclude_seen)` — mask seen items
  to `-inf`, stable-sort by descending score, take top `n`.
- `require_fitted(fitted, model_name)` / `require_non_empty_train(train)`.
- `build_seen_dict(train)` — `UserIdx → set[ItemIdx]`.

### `src/models/collaborative.py`

- **`SvdModel(n_factors=50, seed=42)`** — truncated SVD via
  `scipy.sparse.linalg.svds`. Folds singular values into the left factor.
  `predict` = dot product; `recommend` = top-k.
- **`AlsModel(n_factors=50, n_iter=15, reg=0.1, seed=42)`** — Koren
  baseline + ALS. `_fit_baselines` computes `μ`, `b_u`, `b_i` in closed
  form on the raw ratings; ALS fits on the residuals
  `r_ui - (μ + b_u + b_i)`. `predict` returns `μ + b_u + b_i + x_u · y_i`.

### `src/models/content_based.py`

- **`ContentModel()`** — takes `(movies, preprocessed)`. Builds per-item
  documents from `title + genres`, fits sklearn `TfidfVectorizer`, builds
  per-user profiles as the rating-weighted mean of item embeddings
  (L2-normalised). `recommend` = cosine similarity = sparse dot product.
- Users with zero training ratings get a zero profile (all-zero cosines).

### `src/models/neural_cf.py`

- **`NcfModel`** — dataclass config:
  - `n_factors=32`, `hidden=(64, 32, 16)`, `n_epochs=10`, `batch_size=256`,
    `lr=1e-3`, `negatives_per_positive=4`, `positive_threshold=4.0`,
    `seed=42`, `device="auto"`.
- **`_TwoTower`** — separate user + item embeddings (n_factors dim each)
  → concat → MLP(hidden...) → Linear(1) → logit. Trained with
  `BCEWithLogitsLoss` + Adam.
- `device="auto"` picks `cuda` when available via
  `torch.cuda.is_available()` and moves the whole network + batch tensors
  onto it. Explicit `"cpu"` / `"cuda:0"` override the auto path.

### `src/models/hybrid.py`

- **`HybridModel(collaborative, content, neural, n_items, weights=(0.4, 0.3, 0.3), mmr_lambda=0.7, pool_size=50)`**
  — composes already-fitted components. `__post_init__` validates that
  `weights` sums to 1 and `mmr_lambda ∈ [0, 1]`.
- `recommend(user_idx, *, n, exclude_seen=True)`:
  1. Fetch full per-model score vectors via `recommend(..., n=n_items, exclude_seen=False)`.
  2. `min_max_normalise` each into `[0, 1]`.
  3. Weighted sum.
  4. Mask seen items.
  5. Take top `pool_size`.
  6. `mmr_rerank` against content-embedding cosine similarity, return
     the top `n`.
- **`min_max_normalise(scores)`** — public pure function, flat vector →
  all-zero.
- **`mmr_rerank(candidates, *, similarity, n, mmr_lambda)`** — public pure
  function; `λ=1` disables the diversity penalty.

### `src/models/llm_rerank.py`

- **`LlmClient` Protocol** — one method: `complete(prompt, *, timeout) -> str`.
  Any adapter (Anthropic, OpenAI, HuggingFace, LM Studio) just has to
  implement it.
- **`LlmReranker(client, timeout=2.0)`** — `rerank(candidates, catalogue, user_context) -> list[RerankedItem]`.
  - Builds a JSON-in / JSON-out prompt.
  - Calls the client.
  - Parses the response (handles `{"ranking": [...]}` wrapped in markdown
    fences and prose).
  - Merges the ranking back with the original candidate scores.
  - On any failure (exception, malformed JSON, missing schema) falls back
    to the original hybrid order with empty explanations.
- **`RerankedItem(item_idx, score, explanation)`** — frozen dataclass.

### `src/models/hf_llm_client.py`

- **`HuggingFaceClient(model="google/gemma-4-E2B-it", max_new_tokens=512, device="auto")`**
  — thin wrapper around `transformers.pipeline("text-generation", ...)`.
- Model loading is **lazy** (deferred to first `complete` call) and **cached**
  across subsequent calls.
- `device="auto"` resolves via `accelerate`.
- Gated models (e.g. `google/gemma-*`) require the HF account running the
  code has accepted the license on the HF UI first.

### `src/api/app.py`

- `create_app(*, model, user_map, item_map, rating_store) -> FastAPI` —
  everything injected at construction time, no global state. Tests swap
  tiny fitted models in; production wires `scripts/serve.py`.
- Stashed on `app.state`: `model`, `user_map`, `item_map`,
  `reverse_item_map`, `rating_store`.
- Routes:
  - `GET /health` → `HealthResponse`.
  - `POST /recommend` → `RecommendRequest` → `RecommendResponse`. 404 on
    unknown `user_id`; 422 on `n <= 0` via Pydantic.
  - `POST /rate` → `RateRequest` → `RateResponse`. 404 on unknown
    `user_id` or `movie_id`; 422 on `rating ∉ [0.5, 5.0]`.

### `src/api/schemas.py`

Pydantic v2 models — no numpy / pandas leaks across the HTTP boundary.

### `src/api/deps.py`

- **`InMemoryRatingStore`** — append-only list of `(user_id, movie_id, rating)`.
  Swap for a durable store (Kafka / Postgres) in production.

### `src/benchmark.py`

- `evaluate_ranking(recommend_fn, test, *, k) -> dict[str, float]` — mean
  NDCG@k / HR@k / MAP@k across every user in `test`. `recommend_fn` is a
  duck-typed callable — any model's `.recommend` method plugs in.

### `src/persistence.py`

- `save_model(model, path)` — pickles the model; for torch-backed models
  moves the net to CPU first so the artifact is device-portable.
- `load_model(path) -> Any` — unpickles; raises `FileNotFoundError` on
  missing file.

---

## 7. Runtime data flow

### Startup (first boot — `scripts/serve.py`)

```
1. load_movielens("ml-latest-small", data/raw)
     → RawData   (downloads + SHA-256 + extract if absent)
2. preprocess(raw)
     → PreprocessedData
3. Fit SvdModel     (~0.1 s on CPU)
4. Fit ContentModel (~0.2 s on CPU — sklearn TfidfVectorizer)
5. Fit NcfModel     (~3 s on GPU cu132 / ~30 s on CPU)
6. save_model(*, models/)
7. HybridModel(collaborative=svd, content=content, neural=ncf, n_items=…)
8. create_app(…)
9. uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Startup (warm boot)

```
1. load_movielens(...)       — reuses extracted CSVs, no network.
2. preprocess(raw)
3. load_model(models/svd.pkl) / content.pkl / ncf.pkl
4. HybridModel(...)           — no re-fit.
5. create_app(...); uvicorn.run(...)
```

### Per-request flow (`POST /recommend`)

```
client   →  POST /recommend { "user_id": 1, "n": 10 }
uvicorn  →  FastAPI route: validate via Pydantic RecommendRequest
             → look up user_map[RawUserId(1)]  (404 if missing)
             → hybrid.recommend(user_idx, n=10, exclude_seen=True)
                 ↳ svd.recommend(u, n=n_items, exclude_seen=False)
                 ↳ content.recommend(u, n=n_items, exclude_seen=False)
                 ↳ neural.recommend(u, n=n_items, exclude_seen=False)
                 ↳ min_max_normalise each vector
                 ↳ weighted sum
                 ↳ mask seen
                 ↳ top pool_size=50
                 ↳ mmr_rerank against content embeddings
                 ↳ top n=10
             → map ItemIdx back to RawMovieId via reverse_item_map
             → RecommendResponse(user_id, [RecommendedItem(...)])
client   ←  JSON { "user_id": 1, "items": [...] }
```

### Per-request flow (`POST /rate`)

```
client   →  POST /rate { "user_id": 1, "movie_id": 1, "rating": 4.5 }
FastAPI  →  validate via Pydantic RateRequest (rating ∈ [0.5, 5.0])
             → check user_id ∈ user_map  (404)
             → check movie_id ∈ item_map (404)
             → rating_store.add(...)
             → RateResponse(accepted=True, stored_count=N)
```

### LLM re-rank pipeline (`scripts/demo_llm_rerank.py`)

```
candidates = hybrid.recommend(user_idx, n=10)
prompt     = _build_prompt(candidates, catalogue, user_context)
raw        = HuggingFaceClient.complete(prompt, timeout=60)
              ↳ transformers.pipeline(...) — loaded lazily, cached
              ↳ runs on CUDA via device_map="auto"
ranking    = _parse_ranking(raw)
              ↳ strip markdown fence OR outermost { … }
              ↳ json.loads
              ↳ extract [(movie_id, reason)]
result     = _merge_ranking_with_candidates(ranking, candidates)
              ↳ LLM's order wins for items it mentioned
              ↳ any omitted candidate appended in hybrid order
              ↳ any hallucinated movie_id dropped
```

On any exception in any of the above: `_fallback(candidates)` — original
hybrid order, empty explanations. The hybrid recommender stays available
even when the LLM is not.

---

## 8. Quality Gates

Every commit satisfies:

1. **Hand-computed test expectations** — metrics and fusion tests must
   assert exact values, never "in range" (`0 ≤ x ≤ 1`) shortcuts.
2. **No `% len(...)` indexing** — raw IDs never alias into dense index
   space.
3. **Strict `mypy`** — four `NewType`s separate the ID spaces at compile
   time.
4. **Deterministic seeding** — `set_all_seeds()` at every entry point;
   CUDA determinism on.
5. **One concept per commit** — `feat(area): imperative subject`.
6. **Coverage ≥ 85 %** — `pytest --cov-fail-under=85`. Current 96.70 %.
7. **Two-tier module size** — modules ≤ 300 lines, functions ≤ 40 lines.
   Over the cap ⇒ two concepts; split.

Enforcement lives in `.pre-commit-config.yaml` and
`.github/workflows/ci.yml`.

---

## 9. User manual — commands

### Prerequisites

- **Conda env `ai`** — `/home/ai/miniforge3/envs/ai` on this machine.
  Already contains numpy, pandas, scipy, scikit-learn, torch (cu132),
  fastapi, pydantic, uvicorn, httpx, transformers, pytest, mypy, ruff,
  pandas-stubs.
- **HuggingFace auth** — `huggingface-cli login` with a token that has
  accepted the Gemma license on https://huggingface.co/google/gemma-4-E2B-it.

### Install (for a fresh checkout)

```bash
conda activate ai
pip install -e ".[dev,llm]" --extra-index-url https://download.pytorch.org/whl/nightly/cu128
pre-commit install
```

### Fit + serve the API

```bash
python scripts/serve.py
# Env overrides:
#   RECO_DATASET=ml-latest-small       (or ml-25m)
#   RECO_DATA_DIR=data/raw
#   RECO_MODELS_DIR=models             # warm-start cache
#   RECO_N_FACTORS=32
#   RECO_NCF_EPOCHS=3
#   HOST=0.0.0.0  PORT=8000
```

First boot ~15 s (fit + pickle); warm boots ~2 s (cache hit). Serves on
`http://0.0.0.0:8000` by default.

### Hit the API

```bash
# Health
curl http://localhost:8000/health
# → {"status":"ok"}

# Recommend
curl -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 1, "n": 10}'

# Rate
curl -X POST http://localhost:8000/rate \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 1, "movie_id": 1, "rating": 4.5}'
```

### Offline model comparison

```bash
python scripts/compare_models.py --ncf-epochs 5
# → markdown table of NDCG@10 / HR@10 / MAP@10 / fit time per model
```

### End-to-end LLM re-rank demo

```bash
# Uses whatever is in RECO_MODELS_DIR as the warm start.
python scripts/demo_llm_rerank.py

# Override the LLM
RECO_LLM_MODEL=Qwen/Qwen3-1.7B python scripts/demo_llm_rerank.py
# (or Qwen/Qwen3-0.6B for fastest iteration)

# Override the user
USER_ID=42 N=20 python scripts/demo_llm_rerank.py
```

### Shell-based end-to-end smoke

```bash
PYTHON=/home/ai/miniforge3/envs/ai/bin/python ./scripts/e2e_test.sh
# Spawns uvicorn, hits 7 endpoints over real HTTP, tears down.
```

### Manual smoke against a running server

```bash
BASE_URL=http://localhost:8000 USER_ID=1 ./scripts/smoke_test.sh
```

### Quality loop (run before every commit)

```bash
ruff check . && ruff format --check . && mypy src tests && pytest --cov=src --cov-fail-under=85
```

---

## 10. Testing guide

```
tests/
├── test_seed.py                — set_all_seeds determinism
├── test_evaluation.py          — NDCG / HR / AP hand-computed fractions
├── test_benchmark.py           — evaluate_ranking across synthetic users
├── test_persistence.py         — save → load round-trip for every model
├── test_e2e.py                 — @integration: full-stack TestClient
├── data/
│   ├── test_ids.py             — NewType round-trip + frozen dataclasses
│   ├── test_loader.py          — unit + @integration real download
│   └── test_preprocessor.py    — k-core / id maps / split boundaries
├── api/test_routes.py          — /health, /recommend, /rate via TestClient
└── models/
    ├── test_als.py             — rank-1 convergence, determinism, biases
    ├── test_collaborative.py   — SvdModel (rank-1 reconstruct, sort, mask)
    ├── test_content_based.py   — TF-IDF ordering, zero-profile fallback
    ├── test_hybrid.py          — min_max + mmr + end-to-end recommend
    ├── test_hf_llm_client.py   — Protocol compliance, mock pipeline
    ├── test_llm_rerank.py      — JSON extraction (fence / prose), fallback
    └── test_neural_cf.py       — convergence, determinism, device="auto"
```

**Run all:**

```bash
pytest                                     # 148 tests
pytest -m integration                      # +9 slow integration tests (~5 s on GPU)
pytest --cov=src --cov-report=term-missing # coverage report
```

**Markers:** `integration` (real data / real network / fits on real ml-latest-small), `slow` (excluded from default CI).

---

## 11. Deployment

### Dockerfile

Multi-stage build:

1. **builder** — `python:3.12-slim` + `uv` installs deps into `/opt/venv`.
   Pulls torch from the CPU index by default; override with the
   `--extra-index-url` during `docker build` to use cu128 nightly.
2. **runtime** — `python:3.12-slim` copies only `/opt/venv` + `src/` +
   `scripts/`. Runs as non-root user `app` (uid 10001). Healthcheck hits
   `/health` every 30 s with a 60 s start period.

```bash
docker build -t reco-engine .
docker run --rm -p 8000:8000 reco-engine
# → fits models, serves on :8000
```

### GitHub Actions CI

`.github/workflows/ci.yml` mirrors the pre-commit hooks exactly:

- `ruff check .`
- `ruff format --check .`
- `mypy src tests`
- `pytest --cov=src --cov-fail-under=85 -m "not integration"`

Integration tests (`tests/test_e2e.py`, `@pytest.mark.integration`) are
**excluded from CI** — they need the real ml-latest-small download.

---

## 12. Offline benchmark

Representative run on `ml-latest-small` (n_users=610, n_items=3650,
train=84,174, test=3,050, RTX 5080 + Torch cu132):

| Model    | NDCG@10 | HR@10  | MAP@10 | fit+eval (s) |
|----------|---------|--------|--------|--------------|
| SVD      | 0.0461  | 0.2180 | 0.0208 | 0.1 |
| ALS      | 0.0031  | 0.0213 | 0.0011 | 0.1 |
| Content  | 0.0171  | 0.0951 | 0.0069 | 0.1 |
| NCF      | 0.0247  | 0.1262 | 0.0108 | 0.2 |
| **Hybrid** | 0.0443 | **0.2246** | 0.0196 | 2.1 |

The hybrid wins `HR@10` (recall of relevant items in the top-10). SVD
edges ahead on NDCG / MAP — it ranks its hits slightly higher. ALS
underperforms because its popularity bias `b_i` boosts globally popular
items, but the leave-last-n-out split rewards niche items (users rate
popular stuff early in their history, then explore niche). This is an
honest limitation of the baseline, not a bug.

LLM re-rank output (Gemma 4 E2B, real) promoted "Dark Knight" and "Lord
of the Rings: Fellowship" to the top of user 1's list with reasoning:

> 1. **Dark Knight, The (2008)** — Strong action/epic appeal, similar to Gladiator and Star Wars.
> 2. **Lord of the Rings: Fellowship** — High-stakes epic adventure, aligning with the scope of Star Wars and Gladiator.
> 3. **Aliens (1986)** — Classic action/sci-fi, fitting the general action/adventure taste.

---

## 13. Extending the project

### Add a new recommender model

1. Create `src/models/<name>.py`.
2. Implement `fit(...)`, `predict(user_idx, item_idx) -> float`, and
   `recommend(user_idx, *, n, exclude_seen=True) -> list[tuple[ItemIdx, float]]`.
3. Reuse the shared helpers in `src/models/_ranking.py`:
   `top_k_from_scores`, `require_fitted`, `require_non_empty_train`,
   `build_seen_dict`.
4. Mirror the tests in `tests/models/test_<name>.py` — rank-1 or similar
   convergence test, determinism test, before-fit error test,
   seen-exclusion test.
5. If the new model produces a full-user score vector cheaply, add it to
   `HybridModel`'s fusion (update `weights` to a 4-tuple, adjust the
   `abs(sum(weights) - 1.0)` check).

### Add a new LLM backend

1. Create a class with `complete(self, prompt: str, *, timeout: float) -> str`.
2. Feed it into `LlmReranker(client=MyClient(...))`.
3. The existing `_parse_ranking` handles JSON-in-markdown-fence /
   prose-wrapped outputs.

### Swap the default Gemma model

- Env var override: `RECO_LLM_MODEL=<HF-model-id>`.
- Or change the `HuggingFaceClient.model` default in
  `src/models/hf_llm_client.py`.

### Switch to ml-25m

```bash
RECO_DATASET=ml-25m python scripts/serve.py
```

First boot downloads ~260 MB from grouplens.org, verifies SHA-256
`8b21cfb7...`, and fits the models. Roughly 5-10 min first-boot on CPU;
NCF training dominates. On a GPU with cu132, ~30-60 s.

---

## 14. Troubleshooting

### "All checks passed!" but CI fails

Pre-commit runs mypy in an isolated env with a slightly different set of
additional_dependencies. If you edited `.pre-commit-config.yaml` but not
the main `pyproject.toml`, add the dep there too.

### `ModuleNotFoundError: No module named 'transformers'`

Install the `llm` extra: `pip install -e ".[llm]"`.

### "No module named 'torch'" in the main project venv

Project uses conda `ai`, not `.venv`. Run commands via
`/home/ai/miniforge3/envs/ai/bin/python`.

### Gemma model download stalls

HF xet transfer can slow on some connections. Options:
- Just wait — downloads resume from `.incomplete` blobs.
- Override to a smaller model:
  `RECO_LLM_MODEL=Qwen/Qwen3-0.6B python scripts/demo_llm_rerank.py`.

### `RuntimeError: unable to start app: address already in use`

Another server is on port 8000. `PORT=8100 python scripts/serve.py`.

### `404 user not found` on /recommend

`user_id` must be a raw MovieLens `userId` that survived k-core filtering.
In ml-latest-small, user IDs run 1–610 (most of them).

### Recommendation scores look weird after MMR

By design. MMR re-ranks the top `pool_size` for diversity — the emitted
order is MMR-optimised, not score-sorted. Use `mmr_lambda=1.0` to disable
the diversity term.

### Test coverage drops below 85 %

`pytest --cov=src --cov-report=term-missing` highlights lines. Cover them
or annotate with `# pragma: no cover` where truly unreachable (e.g.
`torch.cuda` availability branches).

---

*Last updated: 2026-04-18.*
