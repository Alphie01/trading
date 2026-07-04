"""Anomaly / pump-dump tespiti — z-score spike'ları + sklearn IsolationForest.

Üretilen skorlar (0-100): anomaly_score, pump_risk_score, dump_risk_score,
volume_spike_score, price_spike_score. Bu skorlar DOĞRUDAN BUY üretmez; yalnız RİSK'i
artırır (risk_contribution) → decision/risk manager sıkılaşır. TF gerektirmez.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _clamp(v, lo=0.0, hi=100.0) -> float:
    return float(max(lo, min(hi, v)))


def _isoforest_score(d: pd.DataFrame) -> float:
    """Çok değişkenli anomali skoru (son bar) 0-100. Az veri/hata → 0."""
    try:
        from sklearn.ensemble import IsolationForest
        feats = pd.DataFrame({
            "ret": d["close"].pct_change(),
            "vol_ratio": d["volume"] / d["volume"].rolling(20).mean(),
            "range": (d["high"] - d["low"]) / d["close"],
        }).replace([np.inf, -np.inf], np.nan).dropna()
        if len(feats) < 50:
            return 0.0
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        iso.fit(feats.values)
        s = iso.score_samples(feats.values)  # düşük = daha anormal
        med = float(np.median(s))
        mn = float(s.min())
        if med <= mn:
            return 0.0
        frac = (med - float(s[-1])) / (med - mn)
        return _clamp(max(0.0, min(1.0, frac)) * 100.0)
    except Exception:
        return 0.0


def detect_anomaly(df: pd.DataFrame, symbol: Optional[str] = None, window: int = 200) -> Dict:
    """Son bar için anomali/pump-dump risk skorları."""
    empty = {"anomaly_score": 0.0, "pump_risk_score": 0.0, "dump_risk_score": 0.0,
             "volume_spike_score": 0.0, "price_spike_score": 0.0,
             "risk_contribution": 0.0, "warnings": []}
    if df is None or len(df) < 40:
        return empty

    d = df.tail(window).copy()
    close = d["close"]
    volume = d["volume"]
    ret = close.pct_change()

    hist_ret = ret.iloc[:-1].dropna()
    hist_vol = volume.iloc[:-1].dropna()
    if len(hist_ret) < 20 or len(hist_vol) < 20:
        return empty

    r_mean, r_std = float(hist_ret.mean()), float(hist_ret.std())
    v_mean, v_std = float(hist_vol.mean()), float(hist_vol.std())
    last_ret = float(ret.iloc[-1]) if not np.isnan(ret.iloc[-1]) else 0.0
    last_vol = float(volume.iloc[-1])

    ret_z = (last_ret - r_mean) / r_std if r_std > 0 else 0.0
    vol_z = (last_vol - v_mean) / v_std if v_std > 0 else 0.0

    price_spike_score = _clamp(abs(ret_z) * 25.0)      # |z|=4 → 100
    volume_spike_score = _clamp(vol_z * 20.0)          # z=5 → 100 (hacim yalnız yukarı spike)
    pump = _clamp((ret_z if ret_z > 0 else 0.0) * 20.0 + (vol_z if vol_z > 0 else 0.0) * 10.0)
    dump = _clamp((-ret_z if ret_z < 0 else 0.0) * 20.0 + (vol_z if vol_z > 0 else 0.0) * 10.0)
    iso = _isoforest_score(d)
    anomaly_score = _clamp(max(price_spike_score, 0.8 * volume_spike_score, iso))

    warnings: List[str] = []
    if pump >= 60:
        warnings.append("pump_risk")
    if dump >= 60:
        warnings.append("dump_risk")
    if volume_spike_score >= 70:
        warnings.append("volume_spike")
    if price_spike_score >= 70:
        warnings.append("price_spike")

    # Risk katkısı (YALNIZ artırır, ≤30): pump/dump + genel anomali
    risk_contribution = round(min(30.0, 0.15 * max(pump, dump) + 0.10 * anomaly_score), 2)

    return {
        "anomaly_score": round(anomaly_score, 2),
        "pump_risk_score": round(pump, 2),
        "dump_risk_score": round(dump, 2),
        "volume_spike_score": round(volume_spike_score, 2),
        "price_spike_score": round(price_spike_score, 2),
        "risk_contribution": risk_contribution,
        "warnings": warnings,
    }
