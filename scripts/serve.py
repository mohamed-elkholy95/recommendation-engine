"""Production entry point — fit every model on ml-latest-small, start the API.

Usage:
    python scripts/serve.py              # runs on 0.0.0.0:8000
    HOST=127.0.0.1 PORT=9000 python scripts/serve.py

Heavier models (n_factors, n_epochs, pool_size) are intentionally modest so
the container boots in under ~30 seconds on a laptop-class CPU. Tune via the
``RECO_*`` environment variables for a beefier deployment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

from src.api.app import create_app
from src.api.deps import InMemoryRatingStore
from src.data.loader import load_movielens
from src.data.preprocessor import preprocess
from src.models.collaborative import SvdModel
from src.models.content_based import ContentModel
from src.models.hybrid import HybridModel
from src.models.neural_cf import NcfModel


def main() -> int:
    dataset = os.environ.get("RECO_DATASET", "ml-latest-small")
    data_dir = Path(os.environ.get("RECO_DATA_DIR", "data/raw"))
    n_factors = int(os.environ.get("RECO_N_FACTORS", "32"))
    n_epochs = int(os.environ.get("RECO_NCF_EPOCHS", "3"))

    print(f"[serve] loading dataset={dataset} from {data_dir}", flush=True)
    raw = load_movielens(dataset, data_dir=data_dir)  # type: ignore[arg-type]
    pre = preprocess(raw)

    print(f"[serve] fitting SvdModel(n_factors={n_factors})", flush=True)
    svd = SvdModel(n_factors=n_factors, seed=42)
    svd.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)

    print("[serve] fitting ContentModel", flush=True)
    content = ContentModel()
    content.fit(raw.movies, pre)

    print(f"[serve] fitting NcfModel(n_epochs={n_epochs})", flush=True)
    neural = NcfModel(n_factors=16, n_epochs=n_epochs, seed=42)
    neural.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)

    hybrid = HybridModel(
        collaborative=svd,
        content=content,
        neural=neural,
        n_items=pre.n_items,
    )

    app = create_app(
        model=hybrid,
        user_map=pre.user_map,
        item_map=pre.item_map,
        rating_store=InMemoryRatingStore(),
    )

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print(f"[serve] listening on http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
