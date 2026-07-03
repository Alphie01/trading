"""Sinyal üretimi + alert — tam skordan BUY/HOLD/SELL sinyali ve uyarılar (tenant'a yazılır).

Sinyal, tenant `automation_signals`'a; kritik durumlar `automation_alerts`'a yazılır.
Canlı trade YOK (Faz 7 risk manager). Bu katman yalnız sinyal/alarm üretir.
"""
from __future__ import annotations

from typing import Dict, Optional

from . import tenant_repo
from .config import AutomationConfig as C


def generate_signal(score: Dict, schema: Optional[str], persist: bool = True) -> Dict:
    """Tam skordan sinyal üretir ve (persist ise) tenant'a yazar.

    Returns: {symbol, signal, recommendation, opportunity_score, risk_score, confidence, alerted}
    """
    symbol = (score.get("symbol") or "").upper()
    recommendation = score.get("recommendation", "HOLD")
    opp = score.get("opportunity_score") or 0
    risk = score.get("risk_score") or 0
    conf = score.get("confidence") or 0

    signal = {"STRONG_BUY": "BUY", "BUY": "BUY", "SELL": "SELL", "STRONG_SELL": "SELL"}.get(
        recommendation, "HOLD"
    )

    alerted = False
    if persist and schema:
        # Yalnız anlamlı sinyalleri kaydet (HOLD spam'i önle)
        if signal != "HOLD" or opp >= C.WATCHLIST_MIN_OPPORTUNITY:
            tenant_repo.save_signal(schema, score)

        # Kritik/dikkat uyarıları
        warnings = score.get("warnings") or []
        if "pump_dump_risk" in warnings or risk >= 80:
            tenant_repo.save_alert(schema, symbol, "critical",
                                   f"{symbol} yüksek risk",
                                   f"Risk skoru {risk}. Uyarılar: {', '.join(warnings)}")
            alerted = True
        elif signal == "BUY" and opp >= C.HOT_MIN_OPPORTUNITY:
            tenant_repo.save_alert(schema, symbol, "info",
                                   f"{symbol} güçlü fırsat sinyali",
                                   f"Opportunity {opp}, risk {risk}, güven {conf}. Öneri: {recommendation}")
            alerted = True

    return {
        "symbol": symbol,
        "signal": signal,
        "recommendation": recommendation,
        "opportunity_score": opp,
        "risk_score": risk,
        "confidence": conf,
        "alerted": alerted,
    }
