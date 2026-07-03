"""Evaluation veri erişimi — SHARED şema (get_session; best-effort, hatalar yutulur).

Desen: intelligence/repository.py ile birebir (Decimal→float, lazy import, try/except).
Yeni tablolar: model_evaluations, feature_snapshots (shared_0004).
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, List, Optional

logger = logging.getLogger("evaluation.repository")


def _f(v):
    return float(v) if isinstance(v, Decimal) else v


def save_evaluation(row: Dict) -> Optional[int]:
    """Bir değerlendirme kaydını model_evaluations'a yazar; eklenen id'yi döndürür."""
    try:
        from trading_db.models_shared import ModelEvaluation
        from trading_db.session import get_session
        with get_session() as s:
            ev = ModelEvaluation(
                model_id=row.get("model_id"),
                symbol=(row.get("symbol") or "").upper(),
                model_type=row.get("model_type"),
                feature_set_version=row.get("feature_set_version"),
                eval_type=row.get("eval_type") or "walk_forward",
                timeframe=row.get("timeframe"),
                horizon=row.get("horizon"),
                sample_count=row.get("sample_count"),
                metrics=row.get("metrics"),
                folds=row.get("folds"),
                window_start=row.get("window_start"),
                window_end=row.get("window_end"),
            )
            s.add(ev)
            s.flush()
            return ev.id
    except Exception as e:
        logger.info("save_evaluation atlandı: %s", e)
        return None


def get_evaluations(symbol: Optional[str] = None, model_type: Optional[str] = None,
                    limit: int = 50) -> List[Dict]:
    """En güncel değerlendirme kayıtları (dashboard/Lab için)."""
    try:
        from sqlalchemy import select
        from trading_db.models_shared import ModelEvaluation as M
        from trading_db.session import get_session
        with get_session() as s:
            q = select(M).order_by(M.created_at.desc()).limit(limit)
            if symbol:
                q = q.where(M.symbol == symbol.upper())
            if model_type:
                q = q.where(M.model_type == model_type)
            rows = s.execute(q).scalars().all()
            return [{
                "id": r.id, "model_id": r.model_id, "symbol": r.symbol,
                "model_type": r.model_type, "feature_set_version": r.feature_set_version,
                "eval_type": r.eval_type, "timeframe": r.timeframe, "horizon": r.horizon,
                "sample_count": r.sample_count, "metrics": r.metrics, "folds": r.folds,
                "window_start": r.window_start.isoformat() if r.window_start else None,
                "window_end": r.window_end.isoformat() if r.window_end else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            } for r in rows]
    except Exception as e:
        logger.info("get_evaluations atlandı: %s", e)
        return []


def save_feature_snapshots(rows: List[Dict]) -> int:
    """feature_snapshots toplu insert (Faz 3 eğitim verisi biriktirme). Dönüş: yazılan sayı."""
    if not rows:
        return 0
    try:
        from trading_db.models_shared import FeatureSnapshot
        from trading_db.session import get_session
        written = 0
        with get_session() as s:
            for r in rows:
                s.add(FeatureSnapshot(
                    symbol=(r.get("symbol") or "").upper(),
                    feature_set_version=r.get("feature_set_version"),
                    timeframe=r.get("timeframe"),
                    features=r.get("features"),
                    feature_hash=r.get("feature_hash"),
                    horizon=r.get("horizon"),
                    label=r.get("label"),
                    label_type=r.get("label_type"),
                    resolved=bool(r.get("resolved", False)),
                    bar_time=r.get("bar_time"),
                ))
                written += 1
        return written
    except Exception as e:
        logger.info("save_feature_snapshots atlandı: %s", e)
        return 0
