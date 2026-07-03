"""Binance fee + slippage simülasyonu — SIM_* env + per-sim fee_profile ile yönetilir (hardcode YOK)."""
from __future__ import annotations

import os
from typing import Dict, Optional


def _env_rate(mode: str, is_maker: bool) -> float:
    if mode.upper() == "FUTURES":
        return float(os.getenv("SIM_FUTURES_MAKER_FEE_RATE", "0.0002") if is_maker
                     else os.getenv("SIM_FUTURES_TAKER_FEE_RATE", "0.0005"))
    return float(os.getenv("SIM_SPOT_MAKER_FEE_RATE", "0.001") if is_maker
                 else os.getenv("SIM_SPOT_TAKER_FEE_RATE", "0.001"))


def fee_rate(mode: str, fee_profile: Optional[Dict] = None, is_maker: bool = False) -> float:
    """Etkin fee oranı: fee_profile (varsa) > SIM_* env. Opsiyonel BNB indirimi.

    fee_profile dict beklenir (rate override + is_maker/use_bnb_discount); dict değilse yok sayılır.
    """
    if not isinstance(fee_profile, dict):
        fee_profile = {}
    # fee_profile içinde açık is_maker varsa çağrı parametresini geçersiz kılar
    if fee_profile.get("is_maker") is not None:
        is_maker = bool(fee_profile["is_maker"])
    rate = None
    if fee_profile:
        key = ("futures_" if mode.upper() == "FUTURES" else "spot_") + ("maker" if is_maker else "taker")
        if fee_profile.get(key) is not None:
            rate = float(fee_profile[key])
    if rate is None:
        rate = _env_rate(mode, is_maker)
    use_bnb = (fee_profile or {}).get("use_bnb_discount")
    if use_bnb is None:
        use_bnb = os.getenv("SIM_USE_BNB_DISCOUNT", "false").lower() == "true"
    if use_bnb:
        rate *= 0.75  # ~%25 BNB indirimi
    return rate


def calc_fee(notional: float, mode: str, fee_profile: Optional[Dict] = None, is_maker: bool = False) -> float:
    return abs(float(notional)) * fee_rate(mode, fee_profile, is_maker)


def slippage_rate(config: Optional[Dict] = None) -> float:
    if config and config.get("slippage") is not None:
        return float(config["slippage"])
    return float(os.getenv("SIM_SLIPPAGE_RATE", "0.0005"))
