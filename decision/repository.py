"""Decision veri erişimi — SHARED (get_session; best-effort). market_regime_snapshots (shared_0006)."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, Optional

logger = logging.getLogger("decision.repository")


def _f(v):
    return float(v) if isinstance(v, Decimal) else v


def save_regime_snapshot(symbol: str, regime: Optional[Dict], anomaly: Optional[Dict] = None,
                         timeframe: str = "4h") -> Optional[int]:
    """Rejim + anomali skorlarını market_regime_snapshots'a yazar (her hesaplama yeni satır)."""
    if not regime:
        return None
    try:
        from trading_db.models_shared import MarketRegimeSnapshot
        from trading_db.session import get_session
        an = anomaly or {}
        with get_session() as s:
            row = MarketRegimeSnapshot(
                symbol=(symbol or "").upper(),
                timeframe=timeframe,
                regime=regime.get("regime"),
                regime_confidence=regime.get("regime_confidence"),
                method=regime.get("method", "rule"),
                adx=regime.get("adx"),
                volatility=regime.get("volatility"),
                anomaly_score=an.get("anomaly_score"),
                pump_risk_score=an.get("pump_risk_score"),
                dump_risk_score=an.get("dump_risk_score"),
                features={"regime": regime.get("features"), "anomaly": {k: an.get(k) for k in
                          ("volume_spike_score", "price_spike_score", "risk_contribution")}},
            )
            s.add(row)
            s.flush()
            return row.id
    except Exception as e:
        logger.info("save_regime_snapshot atlandı: %s", e)
        return None


def get_regime_snapshot(symbol: str, timeframe: str = "4h") -> Optional[Dict]:
    """Bir sembol+timeframe için en güncel rejim snapshot'ı."""
    try:
        from sqlalchemy import select
        from trading_db.models_shared import MarketRegimeSnapshot as M
        from trading_db.session import get_session
        with get_session() as s:
            r = s.execute(
                select(M).where(M.symbol == (symbol or "").upper(), M.timeframe == timeframe)
                .order_by(M.computed_at.desc()).limit(1)
            ).scalar_one_or_none()
            if not r:
                return None
            return {
                "symbol": r.symbol, "timeframe": r.timeframe, "regime": r.regime,
                "regime_confidence": _f(r.regime_confidence), "method": r.method,
                "adx": _f(r.adx), "volatility": _f(r.volatility),
                "anomaly_score": _f(r.anomaly_score), "pump_risk_score": _f(r.pump_risk_score),
                "dump_risk_score": _f(r.dump_risk_score), "features": r.features,
                "computed_at": r.computed_at.isoformat() if r.computed_at else None,
            }
    except Exception as e:
        logger.info("get_regime_snapshot atlandı: %s", e)
        return None
