"""Sanal emir fill mantığı — GERÇEK EMİR GÖNDERMEZ.

🔴 GÜVENLİK: Bu modül (ve tüm automation/simulation paketi) binance_trader / trade_executor /
create_order çağırmaz. Yalnız fiyata slippage uygulayıp sanal fill + fee üretir.
"""
from __future__ import annotations

from typing import Dict, Optional

from . import fee_calculator as fees
from . import pnl

# Bu modül gerçek emir yeteneğine SAHİP DEĞİLDİR. Değiştirilmemeli.
REAL_ORDERS = False


def assert_simulation_only() -> None:
    """Fail-safe: gerçek emir kapasitesi kapalı olmalı."""
    assert REAL_ORDERS is False, "Simulation execution real-order göndermez (REAL_ORDERS=True olamaz)"


def fill_entry(mode: str, side: str, market_price: float, quantity: float,
               config: Optional[Dict] = None, fee_profile: Optional[Dict] = None) -> Dict:
    """Sanal giriş fill'i (slippage + fee). Gerçek emir YOK."""
    assert_simulation_only()
    slip = fees.slippage_rate(config)
    price = pnl.apply_slippage_entry(market_price, side, slip)
    notional = price * quantity
    fee = fees.calc_fee(notional, mode, fee_profile)
    return {"price": price, "quantity": quantity, "notional": notional, "fee": fee, "slippage": slip}


def fill_exit(mode: str, side: str, market_price: float, quantity: float,
              config: Optional[Dict] = None, fee_profile: Optional[Dict] = None) -> Dict:
    """Sanal çıkış fill'i (slippage + fee). Gerçek emir YOK."""
    assert_simulation_only()
    slip = fees.slippage_rate(config)
    price = pnl.apply_slippage_exit(market_price, side, slip)
    notional = price * quantity
    fee = fees.calc_fee(notional, mode, fee_profile)
    return {"price": price, "quantity": quantity, "notional": notional, "fee": fee, "slippage": slip}
