"""PnL / ROI / slippage / liquidation hesapları (saf fonksiyonlar; DB/ağ yok)."""
from __future__ import annotations

from typing import Dict, Optional

LONG_SIDES = ("BUY", "LONG", "SPOT")


def apply_slippage_entry(price: float, side: str, slip: float) -> float:
    """Giriş fiyatı: long/spot alımda yukarı, short girişte aşağı kayar."""
    return price * (1 + slip) if side.upper() in LONG_SIDES else price * (1 - slip)


def apply_slippage_exit(price: float, side: str, slip: float) -> float:
    """Çıkış fiyatı: long/spot satışta aşağı, short kapanışta yukarı kayar."""
    return price * (1 - slip) if side.upper() in LONG_SIDES else price * (1 + slip)


def spot_pnl(entry_price: float, exit_price: float, quantity: float,
             entry_fee: float, exit_fee: float) -> Dict:
    entry_value = entry_price * quantity
    exit_value = exit_price * quantity
    gross = exit_value - entry_value
    net = gross - entry_fee - exit_fee
    roi = (net / entry_value * 100) if entry_value else 0.0
    return {"gross_pnl": gross, "net_pnl": net, "roi_percent": roi,
            "entry_value": entry_value, "exit_value": exit_value}


def futures_pnl(side: str, entry_price: float, exit_price: float, notional: float,
                margin: float, entry_fee: float, exit_fee: float) -> Dict:
    if side.upper() == "LONG":
        pc = (exit_price - entry_price) / entry_price
    else:  # SHORT
        pc = (entry_price - exit_price) / entry_price
    gross = notional * pc
    net = gross - entry_fee - exit_fee
    roi = (net / margin * 100) if margin else 0.0
    return {"gross_pnl": gross, "net_pnl": net, "roi_percent": roi, "price_change_percent": pc * 100}


def futures_unrealized(side: str, entry_price: float, current_price: float, notional: float) -> float:
    if side.upper() == "LONG":
        pc = (current_price - entry_price) / entry_price
    else:
        pc = (entry_price - current_price) / entry_price
    return notional * pc


def spot_unrealized(entry_price: float, current_price: float, quantity: float) -> float:
    return (current_price - entry_price) * quantity


def liquidation_price(side: str, entry_price: float, leverage: Optional[float]) -> Optional[float]:
    """Yaklaşık (isolated) liquidation fiyatı — kesin Binance modeli değil, gösterim amaçlı."""
    if not leverage or leverage <= 0:
        return None
    if side.upper() == "LONG":
        return entry_price * (1 - 1.0 / leverage)
    return entry_price * (1 + 1.0 / leverage)
