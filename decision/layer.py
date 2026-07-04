"""DecisionEngine (Faz 7) — Faz 3-6 parçalarını nihai karara birleştirir.

Pipeline: data_quality → regime → anomaly → ensemble(research.ai) → intelligence(research) →
(multi-tf) → risk → final. `score_full` çıktısının SÜPERSETİNİ döner (tüm eski anahtarlar
korunur → generate_signal/save_score/RiskManager/simülasyon kırılmaz). Yalnız DÜŞÜRÜR/veto eder
(BUY→HOLD olabilir, asla HOLD→BUY). DECISION_LAYER_ENABLED default KAPALI. TF gerektirmez.
"""
from __future__ import annotations

from typing import Dict, Optional


def _data_quality(research: Dict, df) -> float:
    """Veri kalitesi [0,1]: teknik var mı, yeterli bar, NaN oranı, AI mevcut mu."""
    q = 1.0
    tech = research.get("technical")
    if not isinstance(tech, dict) or tech.get("technical_score") is None:
        q -= 0.4
    if df is not None:
        try:
            n = len(df)
            if n < 100:
                q -= 0.3
            elif n < 200:
                q -= 0.1
            nan_ratio = float(df["close"].isna().mean())
            q -= min(0.3, nan_ratio)
        except Exception:
            q -= 0.1
    else:
        q -= 0.1
    if not research.get("ai"):
        q -= 0.1
    return max(0.0, min(1.0, q))


def decide(research: Dict, df=None) -> Dict:
    """score_full superseti — regime/data_quality/ensemble/multi-tf + veto."""
    from automation.config import AutomationConfig as C
    from automation.scoring import score_full

    if df is None:
        df = research.get("_df")
    symbol = research.get("symbol")
    base = score_full(research)

    dq = _data_quality(research, df)

    regime = research.get("regime")
    if not regime and df is not None:
        try:
            from decision.regime import classify_regime
            regime = classify_regime(df, symbol)
        except Exception:
            regime = None
    regime = regime or {}

    anomaly = research.get("anomaly")
    if not anomaly and df is not None:
        try:
            from decision.anomaly import detect_anomaly
            anomaly = detect_anomaly(df, symbol)
        except Exception:
            anomaly = None
    anomaly = anomaly or {}

    ai = research.get("ai") if isinstance(research.get("ai"), dict) else {}
    ensemble = ai.get("ensemble")

    mtf = None
    try:
        if getattr(C, "MULTI_TIMEFRAME_ENABLED", False):
            from decision.multi_timeframe import fetch_and_confirm
            mtf = fetch_and_confirm(symbol, getattr(C, "MULTI_TIMEFRAMES", None))
    except Exception:
        mtf = None

    # confidence: data_quality ile ölçekle (yalnız DÜŞÜRÜR: dq=1→değişmez, dq=0→yarı)
    conf = float(base.get("confidence", 0) or 0) * (0.5 + 0.5 * dq)

    blocked = []
    if dq < 0.4:
        blocked.append("low_data_quality")
    if float(anomaly.get("dump_risk_score", 0) or 0) >= 70:
        blocked.append("high_dump_risk")
    if float(anomaly.get("pump_risk_score", 0) or 0) >= 80:
        blocked.append("extreme_pump_risk")
    if float(base.get("risk_score", 0) or 0) > float(getattr(C, "MAX_RISK_SCORE", 70)):
        blocked.append("risk_score_exceeds_limit")
    if mtf and base.get("recommendation") in ("BUY", "STRONG_BUY") \
            and float(mtf.get("multi_timeframe_alignment", 1) or 1) < 0.5:
        blocked.append("timeframe_disagreement")

    base_rec = base.get("recommendation", "HOLD")
    final_action = "HOLD" if blocked else base_rec

    warnings = list(base.get("warnings") or [])
    for b in blocked:
        if b not in warnings:
            warnings.append(b)

    out = dict(base)  # TÜM eski anahtarlar korunur
    out.update({
        "recommendation": final_action,   # veto → HOLD (generate_signal buna göre sinyal üretir)
        "confidence": round(conf, 4),
        "warnings": sorted(set(warnings)),
        "regime": regime.get("regime"),
        "regime_confidence": regime.get("regime_confidence"),
        "data_quality": round(dq, 4),
        "anomaly_score": anomaly.get("anomaly_score"),
        "ensemble": ensemble,
        "multi_timeframe": ({"alignment": mtf.get("multi_timeframe_alignment"),
                             "final_signal": mtf.get("final_signal")} if mtf else None),
        "final_action": final_action,
        "blocked_reasons": blocked,
        "feature_set_version": "v2_ensemble_advanced",
        "decision_layer": True,
    })
    return out


class DecisionEngine:
    """OO facade (isteğe bağlı)."""

    def decide(self, research: Dict, df=None) -> Dict:
        return decide(research, df=df)
