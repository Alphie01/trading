"""Process-ömürlü (bellek-içi) model cache — LSTM/DQN modellerini request'ler arası
yeniden kullanmak için. En büyük performans kazancı: her analiz isteğinde modelin
diskten yüklenmesini / sıfırdan eğitilmesini engeller.

Güvenlik/tasarım notları:
- Scaler fit/transform mantığına DOKUNULMAZ (mevcut preprocessing davranışı korunur).
  Bu modül yalnız EĞİTİLMİŞ model nesnesini tutar; scaler her istekte eskisi gibi işlenir.
- `model_id` yerine coin bazlı anahtar + coin bazlı LOCK → aynı coin için eşzamanlı
  isteklerde çift yükleme/eğitim engellenir.
- TTL ile bayat modeller düşürülür (yeniden eğitim/güncelleme yansısın).
- tf_config import sırası korunur: bu modül TF'i doğrudan import ETMEZ; model
  sınıfları (lstm_model/dqn) import edildiğinde tf_config zaten devreye girer.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional, Tuple

try:
    from perf_timing import cache_event
except Exception:  # perf_timing yoksa no-op
    def cache_event(kind, key, hit):  # type: ignore
        pass

logger = logging.getLogger("model.memcache")

_TTL = int(os.getenv("MODEL_MEMCACHE_TTL_SECONDS", str(6 * 3600)))  # 6 saat
_MAX = int(os.getenv("MODEL_MEMCACHE_MAX", "50"))

_lstm: "dict[str, dict]" = {}
_dqn: "dict[str, dict]" = {}
_locks: "dict[str, threading.RLock]" = {}
_locks_guard = threading.Lock()


def lock_for(coin: str) -> threading.RLock:
    """Coin bazlı reentrant lock (eşzamanlı çift yükleme/eğitim engeli)."""
    key = coin.lower()
    with _locks_guard:
        lk = _locks.get(key)
        if lk is None:
            lk = threading.RLock()
            _locks[key] = lk
        return lk


def _fresh(entry: Optional[dict]) -> bool:
    return bool(entry) and (time.time() - entry["ts"]) < _TTL


def _evict_if_needed(store: dict):
    if len(store) > _MAX:
        # en eski girişi düşür (basit LRU-benzeri)
        oldest = min(store.items(), key=lambda kv: kv[1]["ts"])[0]
        store.pop(oldest, None)


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
def _lstm_h5(coin: str) -> str:
    return f"model_cache/lstm_{coin.lower()}_model.h5"


def get_or_load_lstm(coin: str):
    """Bellek → disk. Eğitilmiş CryptoLSTMModel döner ya da None (caller eğitir).

    Disk yalnız `.h5` VARSA yüklenir (eğitilmiş model). Bellek TTL içinde ise
    doğrudan döner. Yükleme başarısızsa None (caller ilk eğitim yapar).
    """
    key = coin.lower()
    entry = _lstm.get(key)
    if _fresh(entry):
        cache_event("lstm_model", coin, True)
        return entry["model"]

    h5 = _lstm_h5(coin)
    if os.path.exists(h5):
        try:
            from lstm_model import CryptoLSTMModel

            m = CryptoLSTMModel()
            m.load_model(h5)  # eğitilmiş modeli DİSKTEN yükle (yeniden eğitim YOK)
            _lstm[key] = {"model": m, "ts": time.time()}
            _evict_if_needed(_lstm)
            cache_event("lstm_disk", coin, True)
            logger.info("LSTM diskten yüklendi (retrain yok): %s", coin)
            return m
        except Exception as e:
            logger.warning("LSTM disk load başarısız %s: %s", coin, e)

    cache_event("lstm_model", coin, False)
    return None


def store_lstm(coin: str, model) -> None:
    """İlk eğitim/güncelleme sonrası modeli bellekte sakla (disk .h5 caller tarafından yazılır)."""
    if model is None:
        return
    _lstm[coin.lower()] = {"model": model, "ts": time.time()}
    _evict_if_needed(_lstm)


# --------------------------------------------------------------------------- #
# DQN (bellek-içi; disk yükleme caller'ın mevcut mantığında kalır, burada saklanır)
# --------------------------------------------------------------------------- #
def get_dqn(coin: str):
    key = coin.lower()
    entry = _dqn.get(key)
    if _fresh(entry):
        cache_event("dqn_model", coin, True)
        return entry["model"]
    cache_event("dqn_model", coin, False)
    return None


def store_dqn(coin: str, model) -> None:
    if model is None:
        return
    _dqn[coin.lower()] = {"model": model, "ts": time.time()}
    _evict_if_needed(_dqn)


def invalidate(coin: Optional[str] = None) -> None:
    """Yeniden eğitim/güncelleme sonrası cache temizliği (ör. training_scheduler)."""
    if coin is None:
        _lstm.clear()
        _dqn.clear()
    else:
        k = coin.lower()
        _lstm.pop(k, None)
        _dqn.pop(k, None)


def stats() -> dict:
    return {"lstm_cached": len(_lstm), "dqn_cached": len(_dqn), "ttl_seconds": _TTL}
