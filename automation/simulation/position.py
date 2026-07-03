"""Pozisyon mark-to-market + SL/TP/liquidation çıkış kararı (saf fonksiyonlar)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from . import pnl

LONG_LIKE = ("LONG", "SPOT")


def unrealized_pnl(pos: Dict, current_price: float) -> float:
    """Açık pozisyonun anlık (gerçekleşmemiş) PnL'i."""
    side = pos["side"].upper()
    if pos["mode"].upper() == "FUTURES":
        return pnl.futures_unrealized(side, float(pos["entry_price"]), current_price, float(pos["notional_value"] or 0))
    return pnl.spot_unrealized(float(pos["entry_price"]), current_price, float(pos["quantity"]))


def check_exit(pos: Dict, current_price: float, config: Dict) -> Tuple[bool, Optional[str]]:
    """SL / TP / liquidation kontrolü. (should_exit, reason)."""
    side = pos["side"].upper()
    entry = float(pos["entry_price"])
    sl = config.get("stop_loss")
    tp = config.get("take_profit")
    sl = float(sl) / 100.0 if sl not in (None, "") else None
    tp = float(tp) / 100.0 if tp not in (None, "") else None

    is_long = side in LONG_LIKE

    # Liquidation (futures)
    liq = pos.get("liquidation_price")
    if liq:
        liq = float(liq)
        if (is_long and current_price <= liq) or (not is_long and current_price >= liq):
            return True, "liquidation"

    if is_long:
        if sl and current_price <= entry * (1 - sl):
            return True, f"stop_loss:{sl*100:.1f}%"
        if tp and current_price >= entry * (1 + tp):
            return True, f"take_profit:{tp*100:.1f}%"
    else:  # SHORT
        if sl and current_price >= entry * (1 + sl):
            return True, f"stop_loss:{sl*100:.1f}%"
        if tp and current_price <= entry * (1 - tp):
            return True, f"take_profit:{tp*100:.1f}%"

    return False, None


def signal_exits_position(pos: Dict, signal: str) -> bool:
    """Gelen sinyal bu pozisyonu kapatır mı? (long'u SELL, short'u BUY kapatır)."""
    side = pos["side"].upper()
    sig = (signal or "").upper()
    if side in LONG_LIKE:
        return sig == "SELL"
    return sig == "BUY"  # SHORT'u BUY kapatır
