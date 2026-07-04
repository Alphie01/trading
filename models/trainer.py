"""Tree model eğitimi — fetch/build → train → walk-forward eval → register + save + cache.

Request thread'ini KİLİTLEMEZ (background/CLI). Walk-forward sonucu Faz 1'in
model_evaluations tablosuna, model metadata'sı model_registry'ye yazılır.

CLI:
    python -m models.trainer BTC/USDT --algo random_forest --days 400 --horizon 1
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from features.builders import build_matrix
from . import serialization
from .registry import ModelRegistry, build_model_id, data_hash_of
from .tree_models import SUPPORTED_ALGOS, TreeDirectionModel

_registry = ModelRegistry()


def _walk_forward_matrix(X, y, *, algo, fsv, hp, names, n_folds, symbol, timeframe,
                         horizon, persist) -> Optional[Dict]:
    """Rolling-origin out-of-sample: her fold'da <= split ile eğit, sonraki dilimi tahmin et."""
    n = len(X)
    min_train = max(50, n // (n_folds + 1))
    if n - min_train < n_folds:
        return None
    bounds = np.linspace(min_train, n, n_folds + 1).astype(int)
    folds: List[Dict] = []
    all_pred: List[int] = []
    all_true: List[int] = []
    for k in range(n_folds):
        s, e = int(bounds[k]), int(bounds[k + 1])
        if e <= s:
            continue
        ytr = y[:s]
        if len(set(ytr.tolist())) < 2:
            continue
        m = TreeDirectionModel(algo, fsv, hp, names)
        m.train(X[:s], ytr, names)
        pred = m.predict(X[s:e])
        acc = float((pred == y[s:e]).mean())
        folds.append({"fold": k + 1, "n": int(e - s), "accuracy": round(acc, 4)})
        all_pred.extend(pred.tolist())
        all_true.extend(y[s:e].tolist())
    if not all_pred:
        return None
    overall = float((np.array(all_pred) == np.array(all_true)).mean())
    result = {"directional_accuracy": round(overall, 4),
              "sample_count": len(all_pred), "n_folds": len(folds)}
    if persist:
        try:
            from evaluation import repository as evrepo
            evrepo.save_evaluation({
                "model_id": None, "symbol": symbol, "model_type": algo,
                "feature_set_version": fsv, "eval_type": "walk_forward",
                "timeframe": timeframe, "horizon": horizon,
                "sample_count": len(all_pred), "metrics": result, "folds": folds,
                "window_start": None, "window_end": None,
            })
        except Exception:
            pass
    return {**result, "folds": folds}


def train_tree_model(symbol: str, *, algo: str = "random_forest", df=None,
                     timeframe: str = "4h", days: Optional[int] = None,
                     horizon: int = 1, deadband: float = 0.0,
                     feature_set_version: str = "v2_ensemble_advanced",
                     hyperparams: Optional[Dict] = None, persist: bool = True,
                     do_evaluate: bool = True, n_folds: int = 5) -> Dict:
    """Bir sembol için tree yön modeli eğit + değerlendir + kaydet. Returns özet dict."""
    if algo not in SUPPORTED_ALGOS:
        return {"success": False, "error": f"desteklenmeyen algo: {algo} (izinli: {SUPPORTED_ALGOS})"}

    if df is None:
        try:
            from data_fetcher import CryptoDataFetcher
            df = CryptoDataFetcher().fetch_ohlcv_data(symbol, timeframe=timeframe, days=days)
        except Exception as e:
            return {"success": False, "error": f"OHLCV çekilemedi: {e}"}
    if df is None or len(df) < 200:
        return {"success": False, "error": f"yetersiz veri ({0 if df is None else len(df)} bar; >=200)"}

    X, y, names, _times = build_matrix(df, horizon=horizon, deadband=deadband)
    if len(X) < 100 or len(set(y.tolist())) < 2:
        return {"success": False, "error": f"yetersiz/dengesiz örnek (n={len(X)})"}

    model = TreeDirectionModel(algo, feature_set_version, hyperparams, names)
    train_metrics = model.train(X, y, names)

    eval_metrics = None
    if do_evaluate:
        eval_metrics = _walk_forward_matrix(
            X, y, algo=algo, fsv=feature_set_version, hp=hyperparams, names=names,
            n_folds=n_folds, symbol=symbol, timeframe=timeframe, horizon=horizon, persist=persist,
        )

    dh = data_hash_of({"algo": algo, "fsv": feature_set_version, "hp": hyperparams,
                       "horizon": horizon, "deadband": deadband, "n": int(len(X))})
    model_id = build_model_id(symbol, algo, feature_set_version, dh)
    path = serialization.tree_path(model_id)
    try:
        model.save(path)
    except Exception as e:
        return {"success": False, "error": f"model kaydedilemedi: {e}"}

    if persist:
        _registry.register(
            symbol=symbol, model_type=algo, feature_set_version=feature_set_version,
            file_path=path, feature_count=len(names), data_hash=dh,
            config={"class_name": type(model._clf).__name__, "hyperparams": model.hyperparams,
                    "horizon": horizon, "deadband": deadband, "feature_names": names},
            metrics={"train": train_metrics, "walk_forward": eval_metrics},
        )
    try:
        import model_memory_cache as mc
        mc.store_tree(symbol, algo, feature_set_version, model)
    except Exception:
        pass

    return {"success": True, "model_id": model_id, "algo": algo, "symbol": symbol.upper(),
            "n_samples": int(len(X)), "n_features": len(names),
            "train": train_metrics, "walk_forward": eval_metrics, "file_path": path}


def _cli():
    import argparse

    p = argparse.ArgumentParser(description="Tree model eğitimi (Faz 3)")
    p.add_argument("symbol", help="ör. BTC/USDT")
    p.add_argument("--algo", default="random_forest", choices=list(SUPPORTED_ALGOS))
    p.add_argument("--timeframe", default="4h")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--deadband", type=float, default=0.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--no-persist", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    a = p.parse_args()

    out = train_tree_model(a.symbol, algo=a.algo, timeframe=a.timeframe, days=a.days,
                           horizon=a.horizon, deadband=a.deadband, n_folds=a.folds,
                           persist=not a.no_persist, do_evaluate=not a.no_eval)
    if not out.get("success"):
        print(f"❌ {out['error']}")
        return
    print(f"✅ {out['symbol']} {out['algo']} eğitildi → model_id={out['model_id']}")
    print(f"   train_acc={out['train']['train_accuracy']}  n={out['n_samples']}  feat={out['n_features']}  "
          f"class_balance={out['train']['class_balance']}")
    if out.get("walk_forward"):
        wf = out["walk_forward"]
        print(f"   walk_forward dir_acc={wf['directional_accuracy']}  "
              f"samples={wf['sample_count']}  folds={wf['n_folds']}")


if __name__ == "__main__":
    _cli()
