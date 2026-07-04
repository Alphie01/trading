"""v2_ensemble_advanced feature builder — ham/ÖLÇEKSİZ (tree modelleri için).

Tree'ler ölçekten bağımsızdır → MinMaxScaler'a HİÇ dokunmaz (scaler drift riski yok).
Feature'lar `add_technical_indicators` + fiyat-ölçeğinden bağımsız türetilmiş oranlar
(nedensel → look-ahead yok). Order-book/futures/xasset aileleri Faz 6'da dolacak; şimdilik
None-default (0) → sabit kolonlar tree'ler tarafından yok sayılır (Faz 6'da bilgilendirici olur).

25-feature LSTM setine (v1_lstm_25) DOKUNMAZ; tamamen ayrı bir feature uzayıdır.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# OHLCV'den türetilen bilgilendirici, ölçeksiz, nedensel feature'lar (sabit sıra)
V2_BASE_FEATURES: List[str] = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "ema12_ratio", "sma7_ratio", "sma25_ratio",
    "bb_position", "bb_width",
    "price_change", "volume_change",
    "yigit_position", "yigit_trend_strength", "yigit_bar_buy", "yigit_bar_sell", "yigit_atr_ratio",
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "vol_20", "range_pct", "volume_ratio",
]
# Faz 5/6'da dolacak aileler — şimdilik 0-default (sabit → tree yok sayar)
V2_PLACEHOLDER_FEATURES: List[str] = [
    "ob_imbalance", "ob_spread_pct",   # order book (Faz 6)
    "funding_rate", "oi_change",       # futures (Faz 6)
    "btc_corr", "eth_corr",            # cross-asset (Faz 6)
]
V2_FEATURE_NAMES: List[str] = V2_BASE_FEATURES + V2_PLACEHOLDER_FEATURES


def _safe_div(a, b):
    """Eleman-bazlı bölme; payda 0 → NaN (sonra dropna/mask ile elenir)."""
    b = b.replace(0, np.nan)
    return a / b


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → add_technical_indicators + türetilmiş ölçeksiz feature kolonları."""
    from data_preprocessor import CryptoDataPreprocessor  # lazy (matplotlib import'unu geciktir)

    t = CryptoDataPreprocessor().add_technical_indicators(df.copy())
    close = t["close"]

    t["macd_hist"] = t["macd"] - t["macd_signal"]
    t["ema12_ratio"] = _safe_div(close, t["ema_12"]) - 1.0
    t["sma7_ratio"] = _safe_div(close, t["sma_7"]) - 1.0
    t["sma25_ratio"] = _safe_div(close, t["sma_25"]) - 1.0
    bb_range = t["bb_upper"] - t["bb_lower"]
    t["bb_position"] = _safe_div(close - t["bb_lower"], bb_range)
    t["bb_width"] = _safe_div(bb_range, t["bb_middle"])
    t["yigit_atr_ratio"] = _safe_div(t["yigit_atr"], close)

    for n in (1, 3, 6, 12, 24):
        t[f"ret_{n}"] = close.pct_change(n)
    t["vol_20"] = t["ret_1"].rolling(20).std()
    t["range_pct"] = _safe_div(t["high"] - t["low"], close)
    t["volume_ratio"] = _safe_div(t["volume"], t["volume"].rolling(20).mean())
    return t


def _apply_placeholders(t: pd.DataFrame, context: Optional[Dict]) -> pd.DataFrame:
    ctx = context or {}
    for c in V2_PLACEHOLDER_FEATURES:
        t[c] = float(ctx.get(c, 0.0) or 0.0)
    return t


def build_matrix(df: pd.DataFrame, *, horizon: int = 1, deadband: float = 0.0,
                 context: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, List[str], pd.Index]:
    """Eğitim/backtest matrisi. Returns (X, y, feature_names, times).

    y: 1=up (fwd_ret>deadband) / 0=down (fwd_ret<-deadband); |fwd_ret|<=deadband → belirsiz, atılır.
    """
    t = _apply_placeholders(enrich(df), context)
    close = t["close"].to_numpy(dtype=float)
    n = len(close)

    fr = np.full(n, np.nan)
    if horizon >= 1 and n > horizon:
        fr[:-horizon] = close[horizon:] / close[:-horizon] - 1.0

    y = np.full(n, np.nan)
    y[fr > deadband] = 1.0
    y[fr < -deadband] = 0.0

    feat = t[V2_FEATURE_NAMES].to_numpy(dtype=float)
    mask = ~np.isnan(feat).any(axis=1) & ~np.isnan(y)
    return feat[mask], y[mask].astype(int), list(V2_FEATURE_NAMES), t.index[mask]


def build_row(df: pd.DataFrame, *, context: Optional[Dict] = None
              ) -> Tuple[Optional[np.ndarray], List[str]]:
    """Son (en güncel) bar için inference feature vektörü. Returns (x[1,F] veya None, names)."""
    t = _apply_placeholders(enrich(df), context)
    valid = t[V2_FEATURE_NAMES].dropna()
    if len(valid) == 0:
        return None, list(V2_FEATURE_NAMES)
    x = valid.iloc[-1].to_numpy(dtype=float)
    return x.reshape(1, -1), list(V2_FEATURE_NAMES)
