"""Tree model disk yolları (joblib) — model_cache/trees/{model_id}.joblib.

model_cache/ zaten .gitignore'lu; trees/ alt dizini de öyle. .h5 şemasından ayrı tutulur.
"""
from __future__ import annotations

import os

TREES_DIR = os.getenv("TREE_MODELS_DIR", "model_cache/trees")


def tree_path(model_id: str) -> str:
    return os.path.join(TREES_DIR, f"{model_id}.joblib")


def ensure_dir() -> str:
    os.makedirs(TREES_DIR, exist_ok=True)
    return TREES_DIR
