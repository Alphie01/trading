"""Kural-tabanlı baseline sinyaller (ML/TF YOK) — ölçüm referans noktası.

Bu sinyaller yalnız `data_preprocessor.add_technical_indicators` çıktısındaki nedensel
göstergeleri (rsi/macd/sma/ema) kullanır → look-ahead yoktur; walk-forward'da güvenle
her bara uygulanır. Amaç: yeni modeller eklenmeden ÖNCE ölçülebilir bir baseline.
"""
from __future__ import annotations

import math
from typing import Mapping


def _num(v):
    """Sayıya çevir; NaN/None/geçersiz → None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def technical_signal_row(row: Mapping) -> int:
    """Tek bar için kural-tabanlı yön sinyali. +1 BUY / -1 SELL / 0 HOLD.

    Kural: MACD>signal + close>SMA25 + RSI<70 → BUY; tersi → SELL. Eksik gösterge → HOLD.
    `row` bir dict veya pandas Series olabilir (`.get` destekli).
    """
    rsi = _num(row.get("rsi"))
    macd = _num(row.get("macd"))
    macd_sig = _num(row.get("macd_signal"))
    close = _num(row.get("close"))
    sma25 = _num(row.get("sma_25"))
    if None in (rsi, macd, macd_sig, close, sma25):
        return 0

    bull = (macd > macd_sig) and (close > sma25) and (rsi < 70.0)
    bear = (macd < macd_sig) and (close < sma25) and (rsi > 30.0)
    if bull and not bear:
        return 1
    if bear and not bull:
        return -1
    return 0
