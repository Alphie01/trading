"""Multi-timeframe confirmation (Faz 6) — 5m/15m/1h/4h/1d yön uyumu.

Her timeframe için hafif teknik yön sinyali (EMA20/50 + MACD) → uyum (alignment) skoru.
Tek timeframe ile karar verilmemesi ilkesi. TF gerektirmez (saf pandas). Ağ yalnız
`fetch_and_confirm` içinde (data_fetcher; sunucuda).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]


def _tf_signal(df: pd.DataFrame) -> Dict:
    """Tek timeframe hafif yön sinyali. {signal: -1/0/1, confidence: 0..1}."""
    if df is None or len(df) < 30:
        return {"signal": 0, "confidence": 0.0}
    close = df["close"]
    ema_f = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema_s = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_now = float(macd_line.iloc[-1])
    macd_sig = float(macd_line.ewm(span=9, adjust=False).mean().iloc[-1])
    c = float(close.iloc[-1])

    up = int(c > ema_s) + int(ema_f > ema_s) + int(macd_now > macd_sig)
    dn = int(c < ema_s) + int(ema_f < ema_s) + int(macd_now < macd_sig)
    if up >= 2 and up > dn:
        return {"signal": 1, "confidence": round(up / 3.0, 3)}
    if dn >= 2 and dn > up:
        return {"signal": -1, "confidence": round(dn / 3.0, 3)}
    return {"signal": 0, "confidence": 0.34}


def multi_timeframe_confirmation(symbol: str, dfs_by_tf: Dict[str, pd.DataFrame],
                                 min_alignment: float = 0.6) -> Dict:
    """Timeframe'lerin yön uyumu. dfs_by_tf: {tf: OHLCV DataFrame}."""
    tfs: Dict[str, Dict] = {}
    dirs: List[int] = []
    for tf, df in dfs_by_tf.items():
        sig = _tf_signal(df)
        tfs[tf] = sig
        if sig["signal"] != 0:
            dirs.append(sig["signal"])

    if dirs:
        n_up = dirs.count(1)
        n_dn = dirs.count(-1)
        agree = max(n_up, n_dn)
        alignment = agree / len(dirs)
        lean = 1 if n_up >= n_dn else -1
        if alignment < min_alignment:
            final_signal = "HOLD"
        else:
            final_signal = "BUY" if lean == 1 else "SELL"
    else:
        alignment = 0.0
        final_signal = "HOLD"

    return {
        "symbol": symbol,
        "timeframes": tfs,
        "multi_timeframe_alignment": round(alignment, 4),
        "final_signal": final_signal,
        "n_timeframes": len(tfs),
    }


def fetch_and_confirm(symbol: str, timeframes: Optional[List[str]] = None,
                      days_by_tf: Optional[Dict[str, int]] = None) -> Optional[Dict]:
    """Her timeframe için OHLCV çek (ağ; sunucuda) → confirmation. Hata → mevcut olanlarla."""
    try:
        from data_fetcher import CryptoDataFetcher
        f = CryptoDataFetcher()
    except Exception:
        return None
    tfs = timeframes or ["15m", "1h", "4h", "1d"]
    default_days = {"5m": 5, "15m": 15, "1h": 30, "4h": 120, "1d": 400}
    dfs: Dict[str, pd.DataFrame] = {}
    for tf in tfs:
        try:
            d = (days_by_tf or {}).get(tf, default_days.get(tf, 60))
            df = f.fetch_ohlcv_data(symbol, timeframe=tf, days=d)
            if df is not None and len(df) >= 30:
                dfs[tf] = df
        except Exception:
            continue
    if not dfs:
        return None
    return multi_timeframe_confirmation(symbol, dfs)
