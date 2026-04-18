# Contributing

Thanks for reading. Before you write code, please read [`docs/CODE_STYLE.md`](docs/CODE_STYLE.md) — it is short and explains the conventions every module in this repo follows.

## Getting set up

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before you commit

The pre-commit hooks run these automatically, but you can run them yourself any time:

```bash
ruff check .
ruff format --check .
mypy src tests
pytest --cov=src --cov-fail-under=85
```

If any of them fail, fix the root cause. Do not disable a rule or lower the gate without a linked issue explaining why.

## Commit messages

One concept per commit. Subject line in imperative mood, prefixed with the area:

- `feat(models): add MMR re-rank for diversity`
- `fix(api): handle missing user_id in /rate with 404`
- `docs: clarify temporal-split rule in CODE_STYLE`
- `test(evaluation): add hand-computed NDCG case for rank-3 hits`
- `build: bump ruff to 0.6.9`
- `refactor(data): extract k-core filter into pure function`

Keep the body focused on *why*, not *what* — the diff is the what.
