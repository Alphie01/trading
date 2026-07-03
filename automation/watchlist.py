"""Watchlist state machine — aday coin'in statü geçişleri.

Geçişler (skor + risk'e göre):
  DISCOVERED → RESEARCH_QUEUE → WATCHLIST → HOT_CANDIDATE → TRADE_CANDIDATE
  DISCOVERED → REJECTED
  WATCHLIST → COOLDOWN
  HOT_CANDIDATE → WATCHLIST

Saf fonksiyon: (opportunity, risk, confidence) → hedef statü. Persist eden değil.
"""
from __future__ import annotations

from typing import Optional

from .config import AutomationConfig as C

STATUSES = [
    "discovered", "research_queue", "watchlist", "hot_candidate",
    "trade_candidate", "rejected", "cooldown",
]


def next_status(opportunity: float, risk: float, confidence: float,
                current: Optional[str] = None) -> str:
    """Skorlara göre hedef statü. Trade_candidate yalnız yüksek fırsat + düşük risk + güven."""
    if opportunity is None:
        return "research_queue"

    # Reddetme: çok düşük fırsat veya çok yüksek risk
    if opportunity < 40 or risk > 85:
        return "rejected"

    # Trade adayı: en yüksek eşik + güven (canlı trade YİNE de Faz 7 risk manager'a bağlı)
    if (opportunity >= C.HOT_MIN_OPPORTUNITY and risk <= C.MAX_RISK_SCORE
            and confidence is not None and confidence >= 0.6):
        return "trade_candidate"

    if opportunity >= C.HOT_MIN_OPPORTUNITY and risk <= C.MAX_RISK_SCORE:
        return "hot_candidate"

    if opportunity >= C.WATCHLIST_MIN_OPPORTUNITY and risk <= C.MAX_RISK_SCORE:
        return "watchlist"

    return "research_queue"


def is_promotion(current: Optional[str], target: str) -> bool:
    """target, current'tan daha ileri bir statü mü? (geri düşüşleri ayırt etmek için)"""
    order = {s: i for i, s in enumerate(
        ["rejected", "cooldown", "discovered", "research_queue", "watchlist",
         "hot_candidate", "trade_candidate"]
    )}
    return order.get(target, 0) > order.get(current or "discovered", 2)
