"""Production entry point — fit every model on ml-latest-small (or reload a
cached fit), then start the API.

Usage:
    python scripts/serve.py                      # runs on 0.0.0.0:8000
    HOST=127.0.0.1 PORT=9000 python scripts/serve.py
    RECO_MODELS_DIR=/tmp/cache python scripts/serve.py

Fitted models are pickled to ``$RECO_MODELS_DIR`` (default ``models/``) after
the first successful fit; subsequent boots reload from that cache and skip
training entirely (~30s saved per startup on a laptop).
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
from src.persistence import load_model, save_model


def main() -> int:
    dataset = os.environ.get("RECO_DATASET", "ml-latest-small")
    data_dir = Path(os.environ.get("RECO_DATA_DIR", "data/raw"))
    models_dir = Path(os.environ.get("RECO_MODELS_DIR", "models"))
    n_factors = int(os.environ.get("RECO_N_FACTORS", "32"))
    n_epochs = int(os.environ.get("RECO_NCF_EPOCHS", "3"))

    def stage(msg: str) -> None:
        print(f"[serve] {msg}", flush=True)

    stage(f"loading dataset={dataset} from {data_dir}")
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
        stage(f"fitting SvdModel(n_factors={n_factors})")
        svd = SvdModel(n_factors=n_factors, seed=42)
        svd.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)
        stage("fitting ContentModel")
        content = ContentModel()
        content.fit(raw.movies, pre)
        stage(f"fitting NcfModel(n_epochs={n_epochs})")
        neural = NcfModel(n_factors=16, n_epochs=n_epochs, seed=42)
        neural.fit(pre.train, n_users=pre.n_users, n_items=pre.n_items)

        stage(f"saving fitted models to {models_dir}")
        save_model(svd, svd_path)
        save_model(content, content_path)
        save_model(neural, ncf_path)

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
    stage(f"listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
