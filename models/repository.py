"""Model Registry & Model Weights veri erişimi — SHARED (get_session; best-effort).

- model_registry: MEVCUT tablo (yeni migration YOK). config JSONB'ye framework/class_name/
  feature_set_version/feature_names/hyperparams stashlenir; metrics JSONB son eval snapshot'ı.
- model_weights: shared_0005 (ensemble dinamik ağırlıkları).

Desen: intelligence/repository.py — lazy import, Decimal→float, hatalar yutulur.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, List, Optional

logger = logging.getLogger("models.repository")


def _f(v):
    return float(v) if isinstance(v, Decimal) else v


# --------------------------------------------------------------------------- #
# model_registry (mevcut tablo)
# --------------------------------------------------------------------------- #
def _model_to_dict(r) -> Dict:
    return {
        "id": r.id, "coin_symbol": r.coin_symbol, "model_type": r.model_type,
        "model_id": r.model_id, "file_path": r.file_path, "config": r.config,
        "metrics": r.metrics, "feature_count": r.feature_count, "data_hash": r.data_hash,
        "version": r.version,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "last_trained": r.last_trained.isoformat() if r.last_trained else None,
    }


def upsert_model(meta: Dict) -> Optional[int]:
    """model_registry'ye model_id ile upsert (insert veya güncelle). Dönüş: satır id."""
    try:
        from datetime import datetime, timezone
        from sqlalchemy import select
        from trading_db.models_shared import ModelRegistry
        from trading_db.session import get_session
        with get_session() as s:
            row = s.execute(
                select(ModelRegistry).where(ModelRegistry.model_id == meta.get("model_id"))
            ).scalar_one_or_none()
            if row is None:
                row = ModelRegistry(model_id=meta.get("model_id"))
                s.add(row)
            row.coin_symbol = (meta.get("coin_symbol") or "").upper()
            row.model_type = meta.get("model_type")
            row.file_path = meta.get("file_path")
            row.config = meta.get("config")
            if meta.get("metrics") is not None:
                row.metrics = meta.get("metrics")
            row.feature_count = meta.get("feature_count")
            row.data_hash = meta.get("data_hash")
            row.version = meta.get("version")
            row.last_trained = meta.get("last_trained") or datetime.now(timezone.utc)
            s.flush()
            return row.id
    except Exception as e:
        logger.info("upsert_model atlandı: %s", e)
        return None


def get_model(coin_symbol: str, model_type: str,
              feature_set_version: Optional[str] = None) -> Optional[Dict]:
    """Sembol+tip için en güncel (last_trained) model; feature_set_version verilirse filtreler."""
    try:
        from sqlalchemy import select
        from trading_db.models_shared import ModelRegistry as M
        from trading_db.session import get_session
        with get_session() as s:
            rows = s.execute(
                select(M).where(M.coin_symbol == (coin_symbol or "").upper(),
                                M.model_type == model_type)
                .order_by(M.last_trained.desc().nullslast())
            ).scalars().all()
            for r in rows:
                if feature_set_version is None or (r.config or {}).get("feature_set_version") == feature_set_version:
                    return _model_to_dict(r)
            return None
    except Exception as e:
        logger.info("get_model atlandı: %s", e)
        return None


def get_model_by_id(model_id: str) -> Optional[Dict]:
    try:
        from sqlalchemy import select
        from trading_db.models_shared import ModelRegistry as M
        from trading_db.session import get_session
        with get_session() as s:
            r = s.execute(select(M).where(M.model_id == model_id)).scalar_one_or_none()
            return _model_to_dict(r) if r else None
    except Exception as e:
        logger.info("get_model_by_id atlandı: %s", e)
        return None


def list_models(coin_symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
    try:
        from sqlalchemy import select
        from trading_db.models_shared import ModelRegistry as M
        from trading_db.session import get_session
        with get_session() as s:
            q = select(M).order_by(M.last_trained.desc().nullslast()).limit(limit)
            if coin_symbol:
                q = q.where(M.coin_symbol == coin_symbol.upper())
            return [_model_to_dict(r) for r in s.execute(q).scalars().all()]
    except Exception as e:
        logger.info("list_models atlandı: %s", e)
        return []


def record_metrics(model_id: str, metrics: Dict) -> bool:
    """Bir modelin metrics JSONB'sini günceller (son eval snapshot'ı)."""
    try:
        from sqlalchemy import select
        from trading_db.models_shared import ModelRegistry as M
        from trading_db.session import get_session
        with get_session() as s:
            r = s.execute(select(M).where(M.model_id == model_id)).scalar_one_or_none()
            if r is None:
                return False
            r.metrics = metrics
            return True
    except Exception as e:
        logger.info("record_metrics atlandı: %s", e)
        return False


# --------------------------------------------------------------------------- #
# model_weights (shared_0005) — Faz 4'te EWMA ile güncellenir
# --------------------------------------------------------------------------- #
def _weight_to_dict(r) -> Dict:
    return {
        "id": r.id, "symbol": r.symbol, "model_type": r.model_type,
        "feature_set_version": r.feature_set_version, "regime": r.regime,
        "timeframe": r.timeframe, "weight": _f(r.weight), "sample_count": r.sample_count,
        "win_rate": _f(r.win_rate), "profit_factor": _f(r.profit_factor),
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }


def upsert_weight(row: Dict) -> Optional[int]:
    """(symbol, model_type, feature_set_version, regime, timeframe) anahtarıyla upsert."""
    try:
        from sqlalchemy import select
        from trading_db.models_shared import ModelWeight
        from trading_db.session import get_session
        regime = row.get("regime") or "all"
        timeframe = row.get("timeframe") or "all"
        with get_session() as s:
            w = s.execute(
                select(ModelWeight).where(
                    ModelWeight.symbol == (row.get("symbol") or "").upper(),
                    ModelWeight.model_type == row.get("model_type"),
                    ModelWeight.feature_set_version == row.get("feature_set_version"),
                    ModelWeight.regime == regime,
                    ModelWeight.timeframe == timeframe,
                )
            ).scalar_one_or_none()
            if w is None:
                w = ModelWeight(
                    symbol=(row.get("symbol") or "").upper(),
                    model_type=row.get("model_type"),
                    feature_set_version=row.get("feature_set_version"),
                    regime=regime, timeframe=timeframe,
                )
                s.add(w)
            if row.get("weight") is not None:
                w.weight = row.get("weight")
            if row.get("sample_count") is not None:
                w.sample_count = row.get("sample_count")
            if row.get("win_rate") is not None:
                w.win_rate = row.get("win_rate")
            if row.get("profit_factor") is not None:
                w.profit_factor = row.get("profit_factor")
            s.flush()
            return w.id
    except Exception as e:
        logger.info("upsert_weight atlandı: %s", e)
        return None


def get_weight(symbol: str, model_type: str, feature_set_version: Optional[str] = None,
               regime: str = "all", timeframe: str = "all") -> Optional[Dict]:
    try:
        from sqlalchemy import select
        from trading_db.models_shared import ModelWeight as W
        from trading_db.session import get_session
        with get_session() as s:
            q = select(W).where(W.symbol == (symbol or "").upper(),
                                W.model_type == model_type,
                                W.regime == regime, W.timeframe == timeframe)
            if feature_set_version is not None:
                q = q.where(W.feature_set_version == feature_set_version)
            r = s.execute(q).scalar_one_or_none()
            return _weight_to_dict(r) if r else None
    except Exception as e:
        logger.info("get_weight atlandı: %s", e)
        return None


def get_weights(symbol: Optional[str] = None, regime: Optional[str] = None,
                timeframe: Optional[str] = None, limit: int = 200) -> List[Dict]:
    try:
        from sqlalchemy import select
        from trading_db.models_shared import ModelWeight as W
        from trading_db.session import get_session
        with get_session() as s:
            q = select(W).limit(limit)
            if symbol:
                q = q.where(W.symbol == symbol.upper())
            if regime:
                q = q.where(W.regime == regime)
            if timeframe:
                q = q.where(W.timeframe == timeframe)
            return [_weight_to_dict(r) for r in s.execute(q).scalars().all()]
    except Exception as e:
        logger.info("get_weights atlandı: %s", e)
        return []


# --------------------------------------------------------------------------- #
# signal_feedback (shared_0007) — Faz 8: simülasyon → sinyal kalitesi agregatı
# --------------------------------------------------------------------------- #
def _feedback_to_dict(r) -> Dict:
    return {
        "id": r.id, "symbol": r.symbol, "feature_set_version": r.feature_set_version,
        "regime": r.regime, "timeframe": r.timeframe, "signal_bucket": r.signal_bucket,
        "sample_count": r.sample_count, "win_count": r.win_count, "win_rate": _f(r.win_rate),
        "avg_pnl": _f(r.avg_pnl), "profit_factor": _f(r.profit_factor),
        "quality_score": _f(r.quality_score), "false_signal_reasons": r.false_signal_reasons,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }


def upsert_signal_feedback(row: Dict) -> Optional[int]:
    """(symbol, regime, timeframe, signal_bucket) anahtarıyla agregat upsert."""
    try:
        from sqlalchemy import select
        from trading_db.models_shared import SignalFeedback
        from trading_db.session import get_session
        with get_session() as s:
            fb = s.execute(
                select(SignalFeedback).where(
                    SignalFeedback.symbol == (row.get("symbol") or "").upper(),
                    SignalFeedback.regime == (row.get("regime") or "all"),
                    SignalFeedback.timeframe == (row.get("timeframe") or "all"),
                    SignalFeedback.signal_bucket == (row.get("signal_bucket") or "unknown"),
                )
            ).scalar_one_or_none()
            if fb is None:
                fb = SignalFeedback(
                    symbol=(row.get("symbol") or "").upper(),
                    regime=row.get("regime") or "all",
                    timeframe=row.get("timeframe") or "all",
                    signal_bucket=row.get("signal_bucket") or "unknown",
                )
                s.add(fb)
            fb.feature_set_version = row.get("feature_set_version")
            fb.sample_count = row.get("sample_count")
            fb.win_count = row.get("win_count")
            fb.win_rate = row.get("win_rate")
            fb.avg_pnl = row.get("avg_pnl")
            fb.profit_factor = row.get("profit_factor")
            fb.quality_score = row.get("quality_score")
            fb.false_signal_reasons = row.get("false_signal_reasons")
            s.flush()
            return fb.id
    except Exception as e:
        logger.info("upsert_signal_feedback atlandı: %s", e)
        return None


def get_signal_feedback(symbol: Optional[str] = None, regime: Optional[str] = None,
                        limit: int = 500) -> List[Dict]:
    try:
        from sqlalchemy import select
        from trading_db.models_shared import SignalFeedback as F
        from trading_db.session import get_session
        with get_session() as s:
            q = select(F).order_by(F.updated_at.desc()).limit(limit)
            if symbol:
                q = q.where(F.symbol == symbol.upper())
            if regime:
                q = q.where(F.regime == regime)
            return [_feedback_to_dict(r) for r in s.execute(q).scalars().all()]
    except Exception as e:
        logger.info("get_signal_feedback atlandı: %s", e)
        return []
