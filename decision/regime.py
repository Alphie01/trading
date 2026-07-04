"""Market Regime tespiti — kural-tabanlı (ADX/EMA/volatilite/BB), yorumlanabilir + deterministik.

Rejimler: BULL_TREND / BEAR_TREND / SIDEWAYS / HIGH_VOLATILITY / LOW_VOLATILITY / BREAKOUT / BREAKDOWN.
(NEWS_DRIVEN / MANIPULATIVE fiyat-dışı sinyal gerektirir → intelligence entegrasyonuyla eklenir.)

Rejim, strateji ağırlıklarını segmentlemek için kullanılır (model_weights.regime). TF gerektirmez.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

REGIMES = ["BULL_TREND", "BEAR_TREND", "SIDEWAYS",
           "HIGH_VOLATILITY", "LOW_VOLATILITY", "BREAKOUT", "BREAKDOWN"]


def _fnum(v, default=0.0) -> float:
    try:
        f = float(v)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _safe_div(a, b):
    b = b.replace(0, np.nan)
    return a / b


def _adx(df: pd.DataFrame, period: int = 14):
    """Wilder ADX + DI (trend gücü). Dönüş: (adx, plus_di, minus_di) Series."""
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = pd.concat([(high - low),
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * _safe_div(plus_dm.ewm(alpha=1.0 / period, adjust=False).mean(), atr)
    minus_di = 100 * _safe_div(minus_dm.ewm(alpha=1.0 / period, adjust=False).mean(), atr)
    dx = 100 * _safe_div((plus_di - minus_di).abs(), (plus_di + minus_di))
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx, plus_di, minus_di


def classify_regime(df: pd.DataFrame, symbol: Optional[str] = None,
                    timeframe: str = "4h") -> Dict:
    """OHLCV DataFrame'den market rejimi. Returns {regime, regime_confidence, method, adx, ...}."""
    if df is None or len(df) < 60:
        return {"regime": "SIDEWAYS", "regime_confidence": 0.2, "method": "rule",
                "adx": 0.0, "volatility": 0.0, "features": {}, "reason": "insufficient_data"}

    d = df.tail(300).copy()
    close = d["close"]
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()
    adx_s, _plus, _minus = _adx(d, 14)
    ret = close.pct_change()
    vol = ret.rolling(20).std()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_width = _safe_div(4 * bb_std, bb_mid)
    vol_sma = d["volume"].rolling(20).mean()

    c = _fnum(close.iloc[-1])
    ef = _fnum(ema_fast.iloc[-1])
    es = _fnum(ema_slow.iloc[-1])
    adx_now = _fnum(adx_s.iloc[-1])
    vol_now = _fnum(vol.iloc[-1])
    vol_med = _fnum(vol.median())
    bbw_now = _fnum(bb_width.iloc[-1])
    bbw_med = _fnum(bb_width.median())
    ret_5 = _fnum(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) > 6 else 0.0
    vol_ratio = _fnum(d["volume"].iloc[-1] / vol_sma.iloc[-1], 1.0) if _fnum(vol_sma.iloc[-1]) > 0 else 1.0

    trend_up = c > es and ef > es
    trend_dn = c < es and ef < es
    strong = adx_now >= 25.0
    vol_high = vol_med > 0 and vol_now > 1.8 * vol_med
    vol_low = vol_med > 0 and vol_now < 0.6 * vol_med
    move_sig = vol_now > 0 and abs(ret_5) > 2.2 * (vol_now * (5 ** 0.5))
    was_compressed = bbw_med > 0 and _fnum(bb_width.iloc[-10:-1].mean()) < 0.85 * bbw_med
    breakout = move_sig and vol_ratio > 1.5 and was_compressed

    if breakout and ret_5 > 0:
        regime, conf = "BREAKOUT", 0.6
    elif breakout and ret_5 < 0:
        regime, conf = "BREAKDOWN", 0.6
    elif strong and trend_up:
        regime, conf = "BULL_TREND", min(1.0, adx_now / 50.0)
    elif strong and trend_dn:
        regime, conf = "BEAR_TREND", min(1.0, adx_now / 50.0)
    elif vol_high:
        regime, conf = "HIGH_VOLATILITY", min(1.0, vol_now / (2.0 * vol_med)) if vol_med > 0 else 0.5
    elif vol_low:
        regime, conf = "LOW_VOLATILITY", 0.5
    else:
        regime, conf = "SIDEWAYS", 0.4

    return {
        "regime": regime,
        "regime_confidence": round(float(conf), 4),
        "method": "rule",
        "adx": round(adx_now, 2),
        "volatility": round(vol_now, 6),
        "features": {
            "close": round(c, 8), "ema_fast": round(ef, 8), "ema_slow": round(es, 8),
            "adx": round(adx_now, 2), "vol_now": round(vol_now, 6), "vol_med": round(vol_med, 6),
            "bb_width": round(bbw_now, 5), "ret_5": round(ret_5, 5), "vol_ratio": round(vol_ratio, 3),
        },
    }


# Rejime göre strateji ağırlığı ipuçları (Faz 7 decision layer kullanır)
REGIME_STRATEGY_HINT = {
    "BULL_TREND": {"trend_following": 1.2, "mean_reversion": 0.8, "short_bias": 0.6},
    "BEAR_TREND": {"trend_following": 1.2, "mean_reversion": 0.8, "long_bias": 0.6},
    "SIDEWAYS": {"mean_reversion": 1.2, "breakout_unconfirmed": 0.7},
    "HIGH_VOLATILITY": {"position_size": 0.6, "stop_distance": 1.5},
    "LOW_VOLATILITY": {"breakout_watch": 1.1},
    "BREAKOUT": {"trend_following": 1.1, "confirmation_required": 1.0},
    "BREAKDOWN": {"risk": 1.3},
}
