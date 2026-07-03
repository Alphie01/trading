"""Rolling-origin walk-forward backtester (ilk gerçek historical backtester).

`performance_tester.py` yalnız canlı forward-test yapar; bu modül geçmiş veri üzerinde
out-of-sample değerlendirme sağlar. Nedensellik garantisi: her test barı `i` için tahmin
yalnız `<= i` veriden üretilir; gerçekleşen getiri `close[i+horizon]/close[i]-1`.

İki mod:
- Stateless (baseline): `signal_fn(row_dict) -> int`. `fit_fn=None`.
- Fit'li (Faz 3 tree modelleri): `fit_fn(train_df) -> state`, `signal_fn(state, hist_df) -> int`.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import metrics


def walk_forward(
    df: pd.DataFrame,
    signal_fn: Callable,
    *,
    horizon: int = 1,
    n_folds: int = 5,
    min_train: int = 200,
    deadband: float = 0.0,
    fit_fn: Optional[Callable] = None,
) -> Dict:
    """df: datetime-index'li, en az 'close' + göstergeler içerir.

    Returns: {metrics, folds[], sample_count, horizon, window_start, window_end}.
    """
    n = len(df)
    if n == 0 or "close" not in df.columns:
        return {"metrics": metrics.evaluate_predictions([], []), "folds": [],
                "sample_count": 0, "horizon": horizon,
                "window_start": None, "window_end": None}

    close = df["close"].to_numpy(dtype=float)
    index = df.index
    usable_end = n - horizon  # i+horizon erişilebilir olmalı

    # Küçük veri için güvenli daralt
    min_train = max(20, min(min_train, n // 2))
    if usable_end <= min_train:
        return {"metrics": metrics.evaluate_predictions([], []), "folds": [],
                "sample_count": 0, "horizon": horizon,
                "window_start": None, "window_end": None}
    n_folds = max(1, min(n_folds, usable_end - min_train))

    # Stateless baseline hızlı yol: satırları bir kez dict listesine çevir
    stateless = fit_fn is None
    rows = df.to_dict("records") if stateless else None

    bounds = np.linspace(min_train, usable_end, n_folds + 1).astype(int)
    folds_out: List[Dict] = []
    all_pred: List[float] = []
    all_ret: List[float] = []

    for k in range(n_folds):
        test_start = int(bounds[k])
        test_end = int(bounds[k + 1])
        if test_end <= test_start:
            continue

        state = fit_fn(df.iloc[:test_start]) if (fit_fn is not None) else None

        preds: List[float] = []
        rets: List[float] = []
        for i in range(test_start, test_end):
            if stateless:
                d = signal_fn(rows[i])
            else:
                d = signal_fn(state, df.iloc[: i + 1])
            try:
                d = int(d)
            except (TypeError, ValueError):
                d = 0
            base = close[i]
            r = (close[i + horizon] / base - 1.0) if base else 0.0
            preds.append(d)
            rets.append(r)

        fm = metrics.evaluate_predictions(preds, rets, deadband=deadband)
        fm["fold"] = k + 1
        fm["test_start_time"] = index[test_start].isoformat()
        fm["test_end_time"] = index[test_end - 1].isoformat()
        folds_out.append(fm)
        all_pred.extend(preds)
        all_ret.extend(rets)

    overall = metrics.evaluate_predictions(all_pred, all_ret, deadband=deadband)
    w_start = index[int(bounds[0])]
    w_end = index[min(usable_end, n - 1)]
    return {
        "metrics": overall,
        "folds": folds_out,
        "sample_count": len(all_pred),
        "horizon": horizon,
        "window_start": w_start.to_pydatetime() if hasattr(w_start, "to_pydatetime") else None,
        "window_end": w_end.to_pydatetime() if hasattr(w_end, "to_pydatetime") else None,
    }
