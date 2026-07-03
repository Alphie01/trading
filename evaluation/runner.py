"""Walk-forward evaluation runner (Faz 1) — uçtan uca ölçüm.

fetch OHLCV → add_technical_indicators (TAZE preprocessor) → walk_forward(baseline) →
model_evaluations'a persist. TensorFlow/ML gerektirmez.

CLI:
    python -m evaluation.runner BTC/USDT
    python -m evaluation.runner ETH/USDT --timeframe 4h --horizon 1 --folds 5 --no-persist
"""
from __future__ import annotations

from typing import Dict, Optional

from . import baselines
from . import repository as repo
from . import walk_forward as wf

# Baseline yalnız teknik göstergeleri kullanır → LSTM'in dondurulmuş 25-feature setinden ayrı etiket.
BASELINE_FEATURE_SET = "technical_baseline_v1"


def run_walk_forward_eval(
    symbol: str,
    *,
    timeframe: str = "4h",
    days: Optional[int] = None,
    horizon: int = 1,
    n_folds: int = 5,
    deadband: float = 0.0,
    persist: bool = True,
) -> Dict:
    """Bir sembol için baseline teknik sinyalin walk-forward değerlendirmesi.

    Returns: {success, evaluation_id?, symbol, model_type, metrics, folds, ...} veya {success:False, error}.
    """
    # Ağır importlar burada (paket import'unu TF/matplotlib'ten uzak tutar)
    from data_fetcher import CryptoDataFetcher
    from data_preprocessor import CryptoDataPreprocessor

    try:
        df = CryptoDataFetcher().fetch_ohlcv_data(symbol, timeframe=timeframe, days=days)
    except Exception as e:
        return {"success": False, "error": f"OHLCV çekilemedi: {e}"}
    if df is None or len(df) < 300:
        return {"success": False, "error": f"yetersiz veri ({0 if df is None else len(df)} bar; >=300 gerekli)"}

    # TAZE preprocessor: add_technical_indicators frozen feature_columns .extend'ini TETİKLEMEZ
    # (o yalnız prepare_data'da olur). Sadece df'e gösterge kolonları ekler.
    enriched = CryptoDataPreprocessor().add_technical_indicators(df).dropna().copy()
    if len(enriched) < 300:
        return {"success": False, "error": f"gösterge sonrası yetersiz veri ({len(enriched)} bar)"}

    result = wf.walk_forward(
        enriched, baselines.technical_signal_row,
        horizon=horizon, n_folds=n_folds, deadband=deadband,
    )

    payload = {
        "model_id": None,
        "symbol": symbol.upper(),
        "model_type": "baseline_technical",
        "feature_set_version": BASELINE_FEATURE_SET,
        "eval_type": "walk_forward",
        "timeframe": timeframe,
        "horizon": horizon,
        "sample_count": result["sample_count"],
        "metrics": result["metrics"],
        "folds": result["folds"],
        "window_start": result["window_start"],
        "window_end": result["window_end"],
    }
    eval_id = repo.save_evaluation(payload) if persist else None
    out = {"success": True, "evaluation_id": eval_id}
    out.update(payload)
    # window_* datetime → çıktı JSON'unda ISO
    out["window_start"] = result["window_start"].isoformat() if result["window_start"] else None
    out["window_end"] = result["window_end"].isoformat() if result["window_end"] else None
    return out


def _cli():
    import argparse

    p = argparse.ArgumentParser(description="Walk-forward evaluation (Faz 1, baseline teknik sinyal)")
    p.add_argument("symbol", help="ör. BTC/USDT")
    p.add_argument("--timeframe", default="4h")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--horizon", type=int, default=1, help="ileri bar sayısı (getiri penceresi)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--deadband", type=float, default=0.0, help="|getiri|<deadband → HOLD sayılır")
    p.add_argument("--no-persist", action="store_true", help="DB'ye yazma")
    a = p.parse_args()

    out = run_walk_forward_eval(
        a.symbol, timeframe=a.timeframe, days=a.days, horizon=a.horizon,
        n_folds=a.folds, deadband=a.deadband, persist=not a.no_persist,
    )
    if not out.get("success"):
        print(f"❌ {out.get('error')}")
        return
    m = out["metrics"]
    print(f"📊 {out['symbol']} baseline walk-forward (H={out['horizon']} bar, {out['timeframe']}):")
    print(f"   directional_accuracy = {m['directional_accuracy']}  coverage = {m['coverage']}")
    print(f"   BUY sinyalleri = {m['n_buy_signals']}  win_rate = {m['buy_win_rate']}  "
          f"avg_return = {m['buy_avg_return']}  profit_factor = {m['profit_factor']}")
    print(f"   sample_count = {out['sample_count']}  folds = {len(out['folds'])}  "
          f"eval_id = {out.get('evaluation_id')}")


if __name__ == "__main__":
    _cli()
