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


# --------------------------------------------------------------------------- #
# Faz 8: simülasyon sonuçları → signal_feedback + regime-spesifik model_weights
# --------------------------------------------------------------------------- #
def _bucket(opp) -> str:
    if opp is None:
        return "unknown"
    try:
        o = float(opp)
    except (TypeError, ValueError):
        return "unknown"
    return "high" if o >= 75 else ("mid" if o >= 60 else "low")


def _false_signal_reasons(pos: Dict) -> List[str]:
    """Kaybeden bir pozisyon için yanlış-sinyal sebepleri (mevcut bağlamdan çıkarım)."""
    md = pos.get("metadata") or {}
    out: List[str] = []
    risk = md.get("risk_score")
    opp = md.get("opportunity_score")
    regime = md.get("regime")
    side = (pos.get("side") or "").upper()
    try:
        if risk is not None and float(risk) >= 60:
            out.append("HIGH_RISK_ENTRY")
        if opp is not None and float(opp) < 60:
            out.append("LOW_OPPORTUNITY")
    except (TypeError, ValueError):
        pass
    if regime in ("BEAR_TREND", "BREAKDOWN") and side in ("LONG", "SPOT"):
        out.append("REGIME_MISMATCH")
    er = (pos.get("entry_reason") or "").lower()
    if "thin_volume" in er or "low_liquidity" in er:
        out.append("LOW_LIQUIDITY")
    if "pump" in er:
        out.append("PUMP_DUMP_RISK")
    if pos.get("exit_reason") == "liquidation":
        out.append("LIQUIDATED")
    return out or ["UNCLASSIFIED"]


def _all_closed_positions(schema: str) -> List[Dict]:
    out: List[Dict] = []
    try:
        from automation.simulation import repository as srepo
        for run in srepo.list_runs(schema):
            out.extend(srepo.get_closed_positions(schema, run["id"]))
    except Exception as e:
        logger.info("_all_closed_positions atlandı: %s", e)
    return out


def _quality_score(win_rate: float, profit_factor) -> float:
    """win_rate (0-1) + profit_factor → kalite [0,1]. pf 0.8→0, pf 2.0→1."""
    pf = 0.0 if profit_factor is None else max(0.0, min(1.0, (float(profit_factor) - 0.8) / 1.2))
    return round(max(0.0, min(1.0, 0.5 * float(win_rate) + 0.5 * pf)), 4)


def record_simulation_feedback(schema: str, feature_set_version: str = "v2_ensemble_advanced") -> Dict:
    """Kapalı simülasyon pozisyonlarını (symbol, regime, bucket) gruplar → signal_feedback upsert."""
    from models import repository as mrepo
    positions = _all_closed_positions(schema)
    groups: Dict = {}
    for p in positions:
        md = p.get("metadata") or {}
        key = ((p.get("symbol") or "").upper(), md.get("regime") or "all", _bucket(md.get("opportunity_score")))
        groups.setdefault(key, []).append(p)

    written = 0
    for (sym, regime, bucket), ps in groups.items():
        n = len(ps)
        pnls = [float(p.get("net_pnl") or 0) for p in ps]
        wins = sum(1 for x in pnls if x > 0)
        gains = sum(x for x in pnls if x > 0)
        losses = -sum(x for x in pnls if x < 0)
        pf = (gains / losses) if losses > 0 else None  # kayıp yok → inf → None (JSONB güvenli)
        win_rate = wins / n if n else 0.0
        avg_pnl = sum(pnls) / n if n else 0.0
        reasons: Dict[str, int] = {}
        for p in ps:
            if float(p.get("net_pnl") or 0) <= 0:
                for r in _false_signal_reasons(p):
                    reasons[r] = reasons.get(r, 0) + 1
        quality = _quality_score(win_rate, pf if pf is not None else (2.0 if gains > 0 else 0.0))
        mrepo.upsert_signal_feedback({
            "symbol": sym, "feature_set_version": feature_set_version, "regime": regime,
            "timeframe": "all", "signal_bucket": bucket, "sample_count": n, "win_count": wins,
            "win_rate": round(win_rate * 100, 2), "avg_pnl": round(avg_pnl, 8),
            "profit_factor": round(pf, 4) if pf is not None else None,
            "quality_score": quality, "false_signal_reasons": reasons,
        })
        written += 1
    return {"groups": written, "positions": len(positions)}


def update_weights_from_simulation(schema: Optional[str] = None, *, alpha: float = 0.3,
                                   min_samples: int = 8,
                                   feature_set_version: str = "v2_ensemble_advanced") -> Dict:
    """Simülasyon kalitesini regime-spesifik model_weights'e EWMA ile yansıtır.

    regime='all' walk-forward ağırlığı (Faz 4) DEĞİŞMEZ; regime-spesifik satır yazılır:
    weight = clamp(base_all_weight × (0.5+quality)). Böylece 'gerçek PnL' rejim bazında ağırlığı ayarlar.
    """
    from . import tenant_repo
    from models import repository as mrepo
    schema = schema or tenant_repo.default_tenant_schema()
    if not schema:
        return {"weights_updated": 0, "reason": "no_tenant"}

    rec = record_simulation_feedback(schema, feature_set_version)
    fb = mrepo.get_signal_feedback()
    agg: Dict = {}
    for f in fb:
        a = agg.setdefault((f["symbol"], f["regime"]), {"q": 0.0, "n": 0, "pf_sum": 0.0, "pf_n": 0})
        a["q"] += (f.get("quality_score") or 0) * (f.get("sample_count") or 0)
        a["n"] += (f.get("sample_count") or 0)
        if f.get("profit_factor") is not None:
            a["pf_sum"] += float(f["profit_factor"])
            a["pf_n"] += 1

    updated = 0
    for (sym, regime), a in agg.items():
        if regime == "all" or a["n"] < min_samples:
            continue  # regime='all' (Faz 4 saf accuracy) korunur; regime-spesifik yaz
        quality = a["q"] / a["n"]
        factor = 0.5 + quality  # 0.5..1.5
        avg_pf = (a["pf_sum"] / a["pf_n"]) if a["pf_n"] else None
        for bw in mrepo.get_weights(symbol=sym, regime="all"):
            base = float(bw.get("weight") or 0)
            new = max(0.0, min(1.0, base * factor))
            old = mrepo.get_weight(sym, bw["model_type"], bw.get("feature_set_version"), regime, "all")
            old_w = float(old["weight"]) if old else None
            new_w = round(new if old_w is None else alpha * new + (1.0 - alpha) * old_w, 4)
            mrepo.upsert_weight({
                "symbol": sym, "model_type": bw["model_type"],
                "feature_set_version": bw.get("feature_set_version"),
                "regime": regime, "timeframe": "all", "weight": new_w,
                "profit_factor": round(avg_pf, 4) if avg_pf is not None else None,
            })
            updated += 1
    return {"weights_updated": updated, "feedback": rec}


def run_feedback(symbol: Optional[str] = None) -> Dict:
    """Scheduler/CLI giriş noktası — hem walk-forward (Faz 4) hem simülasyon (Faz 8) feedback'i.

    best-effort: hata canlı turu düşürmez.
    """
    from .config import AutomationConfig as C
    alpha = float(getattr(C, "WEIGHT_EWMA_ALPHA", 0.3))
    ev = {}
    sim = {}
    try:
        ev = update_weights_from_evaluations(symbol, alpha=alpha)
    except Exception as e:
        ev = {"updated": 0, "error": str(e)}
    try:
        sim = update_weights_from_simulation(alpha=alpha)
    except Exception as e:
        sim = {"weights_updated": 0, "error": str(e)}
    updated = int(ev.get("updated", 0) or 0) + int(sim.get("weights_updated", 0) or 0)
    return {"updated": updated, "evaluations": ev, "simulation": sim}


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
