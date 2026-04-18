"""Pickle-based save / load for every fitted recommender.

``save_model`` round-trips any model built in this project: SvdModel,
AlsModel, ContentModel, NcfModel, and HybridModel. For NcfModel the network
is temporarily moved to CPU before pickling so the artifact loads on any
machine; the original device is restored immediately after.

The format is plain Python ``pickle`` — simple, readable by any downstream
scripts without extra deps, and dependent only on the installed torch /
numpy / scipy major versions matching between save and load. For production
deployments swap this for ``torch.save`` + ``state_dict`` on the NCF tower
and JSON / Parquet for everything else; for a portfolio project the one-file
pickle is enough.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch


def save_model(model: Any, path: Path) -> None:
    """Pickle ``model`` to ``path``, creating the parent directory if needed.

    Torch-backed models are temporarily moved to CPU before serialising so the
    resulting artifact is device-independent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    original_device = None
    net = getattr(model, "_net", None)
    if isinstance(net, torch.nn.Module):
        original_device = getattr(model, "_device", torch.device("cpu"))
        net.to("cpu")
        model._device = torch.device("cpu")

    try:
        with path.open("wb") as f:
            pickle.dump(model, f)
    finally:
        if original_device is not None and isinstance(net, torch.nn.Module):
            net.to(original_device)
            model._device = original_device


def load_model(path: Path) -> Any:
    """Unpickle a previously ``save_model``-d artifact from ``path``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)
