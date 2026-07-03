"""Değerlendirme metrikleri — saf numpy (TF/sklearn gerektirmez, JSON-güvenli).

Yön (directional), regresyon (mae/rmse/mape) ve sinyal (win_rate/profit_factor)
metrikleri. Tüm çıktılar JSONB'ye güvenli: inf/nan → None.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _finite(x):
    """inf/nan → None (JSONB güvenli); aksi halde float."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if np.isfinite(xf) else None


def evaluate_predictions(
    pred_dir: Sequence[float],
    realized_return: Sequence[float],
    deadband: float = 0.0,
) -> Dict:
    """Yön tahminlerini gerçekleşen getirilere göre değerlendirir.

    Args:
        pred_dir: her bar için tahmin yönü ∈ {-1, 0, +1} (0 = HOLD/çekimser).
        realized_return: her bar için ileriye dönük gerçekleşen getiri (oran, ör. 0.012).
        deadband: |getiri| < deadband ise gerçek yön 0 sayılır (gürültü bandı).

    Returns:
        directional_accuracy (yalnız non-flat tahminler üzerinde), coverage,
        BUY sinyali işlem proxy'si (win_rate/avg_return/total_return/profit_factor).
    """
    pred = np.asarray(list(pred_dir), dtype=float)
    ret = np.asarray(list(realized_return), dtype=float)
    n = int(pred.size)
    if n == 0:
        return {
            "n_samples": 0, "n_nonflat": 0, "coverage": 0.0,
            "directional_accuracy": 0.0, "n_buy_signals": 0,
            "buy_win_rate": 0.0, "buy_avg_return": 0.0,
            "buy_total_return": 0.0, "profit_factor": None,
        }

    real_dir = np.sign(ret)
    if deadband and deadband > 0:
        real_dir = np.where(np.abs(ret) < deadband, 0.0, real_dir)

    nonflat = pred != 0
    n_nonflat = int(nonflat.sum())
    dir_acc = float((pred[nonflat] == real_dir[nonflat]).mean()) if n_nonflat else 0.0

    buys = pred > 0
    n_buy = int(buys.sum())
    buy_ret = ret[buys]
    win_rate = float((buy_ret > 0).mean()) if n_buy else 0.0
    avg_ret = float(buy_ret.mean()) if n_buy else 0.0
    total_ret = float(buy_ret.sum()) if n_buy else 0.0
    gains = float(buy_ret[buy_ret > 0].sum()) if n_buy else 0.0
    losses = float(-buy_ret[buy_ret < 0].sum()) if n_buy else 0.0
    pf = (gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)

    return {
        "n_samples": n,
        "n_nonflat": n_nonflat,
        "coverage": round(n_nonflat / n, 4),
        "directional_accuracy": round(dir_acc, 4),
        "n_buy_signals": n_buy,
        "buy_win_rate": round(win_rate, 4),
        "buy_avg_return": round(avg_ret, 6),
        "buy_total_return": round(total_ret, 6),
        "profit_factor": _finite(round(pf, 4)),
    }


def regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> Dict:
    """Fiyat/regresyon metrikleri (LSTM tahmin fiyatı vb. için — Faz 3+)."""
    a = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    if a.size == 0 or a.size != p.size:
        return {"mae": None, "rmse": None, "mape": None, "n": int(a.size)}
    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = a != 0
    mape = float(np.mean(np.abs(err[nz] / a[nz])) * 100.0) if nz.any() else None
    return {
        "mae": _finite(round(mae, 8)),
        "rmse": _finite(round(rmse, 8)),
        "mape": _finite(round(mape, 4)) if mape is not None else None,
        "n": int(a.size),
    }
