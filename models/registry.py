"""ModelRegistry — mevcut ``model_registry`` shared tablosunun OO facade'ı (yeni migration YOK).

model_id şeması: ``{SYMBOL}_{model_type}_{feature_set_version}_{data_hash8}`` (sembol sanitize).
config JSONB'ye framework/class_name/feature_set_version/feature_names/hyperparams yazılır;
metrics JSONB son eval snapshot'ı. Disk artefaktı file_path (ör. model_cache/trees/{model_id}.joblib).

LSTM/DQN/Hybrid'i DEĞİŞTİRMEZ; yeni modeller (Faz 3) bu registry'ye kaydolur. İstenirse
mevcut .h5 modelleri de `register(...)` ile geriye dönük kaydedilebilir (backfill).
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Optional

from . import repository as repo


def sanitize_symbol(symbol: str) -> str:
    """'BTC/USDT' → 'BTC_USDT' (model_id / dosya adı güvenli)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", (symbol or "").upper()).strip("_")


def data_hash_of(payload) -> str:
    """Config/veri imzasından deterministik md5 (freshness/versiyonlama için)."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def build_model_id(symbol: str, model_type: str, feature_set_version: str,
                   data_hash: Optional[str] = None) -> str:
    base = f"{sanitize_symbol(symbol)}_{model_type}_{feature_set_version}"
    return f"{base}_{data_hash[:8]}" if data_hash else base


class ModelRegistry:
    """model_registry tablosu üzerinde kayıt/sorgulama facade'ı."""

    def register(
        self,
        *,
        symbol: str,
        model_type: str,
        feature_set_version: str,
        file_path: Optional[str] = None,
        config: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        feature_count: Optional[int] = None,
        data_hash: Optional[str] = None,
        version: str = "1",
    ) -> Optional[str]:
        """Bir model versiyonunu registry'ye yazar (upsert). Dönüş: model_id."""
        config = dict(config or {})
        config.setdefault("feature_set_version", feature_set_version)
        model_id = build_model_id(symbol, model_type, feature_set_version, data_hash)
        meta = {
            "coin_symbol": (symbol or "").upper(),
            "model_type": model_type,
            "model_id": model_id,
            "file_path": file_path,
            "config": config,
            "metrics": metrics,
            "feature_count": feature_count,
            "data_hash": data_hash,
            "version": version,
        }
        rid = repo.upsert_model(meta)
        return model_id if rid is not None else None

    def get(self, symbol: str, model_type: str,
            feature_set_version: Optional[str] = None) -> Optional[Dict]:
        return repo.get_model(symbol, model_type, feature_set_version)

    def get_by_id(self, model_id: str) -> Optional[Dict]:
        return repo.get_model_by_id(model_id)

    def list(self, symbol: Optional[str] = None) -> List[Dict]:
        return repo.list_models(symbol)

    def record_metrics(self, model_id: str, metrics: Dict) -> bool:
        return repo.record_metrics(model_id, metrics)


# Modül seviyesinde kullanışlı tek örnek
registry = ModelRegistry()
