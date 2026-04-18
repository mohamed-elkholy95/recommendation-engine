# Contributing

Thanks for reading. This guide covers **setup**, **code style**, and **how to land a change** — everything a new contributor needs before opening a PR.

---

## 1. Getting set up

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

If Python 3.12 isn't on your system, `uv python install 3.12` (via [uv](https://docs.astral.sh/uv/)) is the fastest way to get a managed interpreter without touching system packages.

---

## 2. Values

We care about more than passing lint. We want this codebase to stay small, calm, and readable — a reader should be able to open any file and understand the concept it implements without jumping around the repo.

Code should feel:

- **Simple** — prefer the smallest change that solves the real problem.
- **Clear** — optimise for the next reader, not for cleverness.
- **Decoupled** — keep boundaries clean, avoid unnecessary new abstractions.
- **Honest** — do not hide complexity, but do not create extra complexity either.
- **Durable** — choose solutions that are easy to maintain, test, and extend.

**In practice:**

- **Target:** Python 3.12.
- **Line length:** 100 characters (ruff formatter owns this).
- **Linting:** ruff with rules `E, F, I, N, W` (`E501` ignored).
- **Formatting:** `ruff format`. Do not hand-format.
- **Typing:** mypy in strict mode.
- **Async:** asyncio throughout the API layer; pytest with `asyncio_mode = "auto"`.
- **No magic:** no metaclasses, no decorator-returning-decorators, no `exec` / `eval`.
- **Focused patches:** one concept per commit.

---

## 3. Principles

Six rules. Each has a one-line rationale. Everything else in this doc is a consequence of these.

1. **One module, one concept.** A reader opens `collaborative.py` and finds only collaborative filtering.
2. **Names encode concepts.** `user_idx`, `rating_matrix`, `mmr_lambda` — never `i`, `m`, `x`.
3. **Explicit over clever.** A reader should be able to predict what a function does without running it.
4. **Fail loud, fail early.** Silent wrong-math is the dominant failure mode in recommenders. Raise on missing IDs, unknown genres, wrong shapes.
5. **Types are contracts.** Strict mypy catches ID-space aliasing and tensor-shape mismatches before they become silent bugs.
6. **Small units.** Modules ≤ 300 lines, functions ≤ 40 lines. If it grows, the concept was two concepts.

---

## 4. Structure

### Repo layout

```
src/
  data/          # loaders, preprocessors
  models/        # collaborative, content_based, neural_cf, hybrid
  api/           # FastAPI app, schemas, deps
  evaluation.py  # metrics
  utils/         # shared helpers (seed, logging, ids)
tests/           # mirrors src/
scripts/         # training, dataset, eval entry points
notebooks/       # EDA only; not imported from src/
```

### Size limits

- **Module:** ≤ 300 lines hard cap, ≤ 200 lines target.
- **Function:** ≤ 40 lines. Each function does one thing you can name in imperative form.

### Imports

Order: stdlib → third-party → local, sorted by ruff-isort. No star imports. No circular dependencies. Local imports use absolute paths (`from src.utils.seed import set_all_seeds`), never relative.

### IDs are typed

MovieLens raw IDs and the dense indices used by matrix factorisation are *different spaces*. We keep them separate at the type level:

```python
from typing import NewType

UserIdx = NewType("UserIdx", int)        # 0-based dense index used by models
ItemIdx = NewType("ItemIdx", int)        # 0-based dense index used by models
RawUserId = NewType("RawUserId", int)    # MovieLens userId
RawMovieId = NewType("RawMovieId", int)  # MovieLens movieId
```

The preprocessor builds `user_map` and `item_map` **once** and every consumer receives them explicitly. Zero tolerance for `% len(...)` indexing.

---

## 5. Writing Python

### Typing

- Strict mypy — enforced by `pyproject.toml`.
- Prefer concrete types (`list[int]`) over abstract (`Iterable[int]`) in public signatures unless the abstraction is load-bearing.
- Use `NewType` for the ID spaces above.

### Docstrings

Three tiers:

1. **Public API** (exported classes/functions, FastAPI endpoints, CLI entry points): **Google-style**. Summary, `Args`, `Returns`, `Raises`, short `Example` when it helps.
2. **Core ML internals** (`fit`, `predict`, `recommend`, every metric): one short paragraph explaining *what it does and why the math works that way*.
3. **Private helpers** (prefix `_`): no docstring — the name is the contract.

### Concept anchors

Two comment tags only:

- `# concept: <term>` — ties a line to a domain concept.
- `# gate: <Quality-Gate name>` — ties a line to a quality gate.

Anchors go **above** the line they explain, never trailing:

```python
# concept: min-max normalise so CF (0-5) and NCF (0-1) are comparable
scores = (scores - scores.min()) / (scores.max() - scores.min())
```

No other comment style is encouraged. Names and types should carry the rest of the meaning.

### Errors

- `ValueError` — invalid input (e.g. `n < 1`, unknown genre).
- `LookupError` — missing ID in `user_map` / `item_map`.
- `RuntimeError` — model used before `fit()`; export shape mismatch.
- Never bare `except:`. Only the API layer translates exceptions to HTTP.

### Logging

`structlog`, JSON output. Every line carries `trace_id` and hashed `user_id`. Never log PII, raw ratings, or API keys.

### Determinism

Every training / evaluation script calls `set_all_seeds()` at entry. The helper lives at `src/utils/seed.py` and seeds Python, NumPy, and (if installed) PyTorch + CUDA.

### I/O boundaries

pandas and NumPy types live inside `src/data/` and `src/models/`. The API layer converts to and from Pydantic v2 models. No pandas DataFrames in request or response bodies.

---

## 6. Testing

- **Framework:** pytest. **Coverage gate:** ≥ 85 % on `src/` (fails CI below).
- **Layout:** one test file per source file. `src/models/hybrid.py` → `tests/test_hybrid.py`.
- **Hand-computed expected values** for every metric and fusion step. Range-only assertions (`0 <= x <= 1`) are forbidden in model and metric tests — they hide math bugs.
- **Fixtures:** tiny synthetic datasets (≤ 10 users, ≤ 10 items) in `tests/conftest.py`. Name them by what they exercise: `users_with_temporal_overlap`, `movies_with_empty_genres`.
- **Property tests** for invariants (normalised scores in `[0, 1]`, dedupe, no future leakage in temporal splits).
- **Test names** describe behaviour: `test_mmr_penalizes_near_duplicates_across_full_selected_set`, not `test_mmr_1`.
- **Integration tests** live in `tests/test_integration.py` and run under `pytest -m integration`.

---

## 7. Before you commit

Pre-commit hooks run these automatically, but you can also run them yourself:

```bash
ruff check .
ruff format --check .
mypy src tests
pytest --cov=src --cov-fail-under=85
```

If any fail, fix the root cause. Do not disable a rule or lower the gate without a linked issue explaining why.

---

## 8. Commit messages

One concept per commit. Subject line in imperative mood, prefixed with the area:

- `feat(models): add MMR re-rank for diversity`
- `fix(api): handle missing user_id in /rate with 404`
- `docs: clarify temporal-split rule`
- `test(evaluation): add hand-computed NDCG case for rank-3 hits`
- `build: bump ruff to 0.6.9`
- `refactor(data): extract k-core filter into pure function`

Keep the body focused on *why*, not *what* — the diff is the what.
