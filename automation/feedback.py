"""Feedback — model performansından model_weights EWMA güncellemesi (background job).

Faz 4: ağırlık kaynağı = walk-forward directional_accuracy (model_evaluations).
        new_weight = alpha*derived + (1-alpha)*old ; derived = clamp(2*(dir_acc-0.5), 0, 1).
Faz 8: kapalı simülasyon sonuçları + false-signal analizi ile kapanış (signal_feedback).

Request thread'ini KİLİTLEMEZ (scheduler `_job` sonunda / CLI). Yalnız model_weights yazar
→ ensemble KAPALIYKEN (default) canlı akışa etkisi YOKTUR (inert).
"""
from __future__ import annotations

from typing import Dict, List, Optional


def _derived_weight(dir_acc: Optional[float]) -> float:
    """directional_accuracy → [0,1] ağırlık. 0.5→0 (şanstan iyi değil), 0.75→0.5, 1.0→1.0."""
    if dir_acc is None:
        return 0.0
    return max(0.0, min(1.0, 2.0 * (float(dir_acc) - 0.5)))


def update_weights_from_evaluations(symbol: Optional[str] = None, *, alpha: float = 0.3,
                                    regime: str = "all", timeframe: str = "all") -> Dict:
    """En güncel walk_forward değerlendirmelerinden (symbol, model_type) ağırlıklarını EWMA günceller."""
    try:
        from evaluation import repository as evrepo
        from models import repository as mrepo
    except Exception as e:
        return {"updated": 0, "error": str(e)}

    evs = evrepo.get_evaluations(symbol=symbol, limit=300)  # created_at DESC
    latest: Dict = {}
    for e in evs:
        if e.get("eval_type") != "walk_forward":
            continue
        if e.get("model_type") == "baseline_technical":
            continue  # baseline referanstır; ağırlıklandırılmaz
        key = (e["symbol"], e["model_type"], e.get("feature_set_version"))
        if key not in latest:  # ilk görülen = en güncel
            latest[key] = e

    updated = 0
    details: List[Dict] = []
    for (sym, mtype, fsv), e in latest.items():
        dir_acc = (e.get("metrics") or {}).get("directional_accuracy")
        derived = _derived_weight(dir_acc)
        old = mrepo.get_weight(sym, mtype, fsv, regime, timeframe)
        old_w = float(old["weight"]) if old else None
        new_w = round(derived if old_w is None else alpha * derived + (1.0 - alpha) * old_w, 4)
        mrepo.upsert_weight({
            "symbol": sym, "model_type": mtype, "feature_set_version": fsv,
            "regime": regime, "timeframe": timeframe, "weight": new_w,
            "sample_count": e.get("sample_count"),
            "win_rate": round(float(dir_acc) * 100, 2) if dir_acc is not None else None,
        })
        updated += 1
        details.append({"symbol": sym, "model_type": mtype, "dir_acc": dir_acc,
                        "derived": round(derived, 4), "old": old_w, "new": new_w})
    return {"updated": updated, "details": details}


def run_feedback(symbol: Optional[str] = None) -> Dict:
    """Scheduler/CLI giriş noktası (best-effort — hata canlı turu düşürmez)."""
    try:
        from .config import AutomationConfig as C
        alpha = float(getattr(C, "WEIGHT_EWMA_ALPHA", 0.3))
        return update_weights_from_evaluations(symbol, alpha=alpha)
    except Exception as e:
        return {"updated": 0, "error": str(e)}


def _cli():
    import argparse

    p = argparse.ArgumentParser(description="Model ağırlık feedback (Faz 4)")
    p.add_argument("--symbol", default=None)
    p.add_argument("--alpha", type=float, default=0.3)
    a = p.parse_args()
    out = update_weights_from_evaluations(a.symbol, alpha=a.alpha)
    if out.get("error"):
        print(f"❌ {out['error']}")
        return
    print(f"✅ Ağırlık güncellendi: {out['updated']} model")
    for d in out.get("details", [])[:20]:
        print(f"   {d['symbol']:<12} {d['model_type']:<18} dir_acc={d['dir_acc']} "
              f"→ w={d['new']} (eski={d['old']})")


if __name__ == "__main__":
    _cli()
