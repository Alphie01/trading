"""Sanal portföy / pozisyon boyutlandırma yardımcıları (saf)."""
from __future__ import annotations

from typing import Dict, Tuple


def _risk_fraction(config: Dict) -> float:
    return max(0.0, float(config.get("risk_per_trade", 2)) / 100.0)


def size_spot(balance: float, market_price: float, config: Dict) -> Tuple[float, float]:
    """Spot: bakiyenin risk%'i kadar alım. Döner: (allocation_usdt, quantity)."""
    alloc = balance * _risk_fraction(config)
    if market_price <= 0:
        return 0.0, 0.0
    qty = alloc / market_price
    return alloc, qty


def size_futures(balance: float, market_price: float, config: Dict) -> Tuple[float, float, float]:
    """Futures: margin = bakiye*risk%, notional = margin*leverage. Döner: (margin, notional, quantity)."""
    margin = balance * _risk_fraction(config)
    leverage = max(1.0, float(config.get("leverage", 1)))
    notional = margin * leverage
    if market_price <= 0:
        return margin, 0.0, 0.0
    qty = notional / market_price
    return margin, notional, qty


def min_order_value(config: Dict) -> float:
    return float(config.get("min_order_value", 10.0))
