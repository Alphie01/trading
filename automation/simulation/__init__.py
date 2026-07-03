"""Simulation / Paper Trading — Automation Engine'in sanal (gerçek-para-yok) test modu.

🔴 GÜVENLİK: Bu paket binance_trader / trade_executor / create_order İMPORT ETMEZ, ÇAĞIRMAZ.
Yalnız sanal pozisyon/işlem kaydı üretir. Gerçek trading engine'den kesin ayrıdır.
"""
from .engine import (  # noqa: F401
    create_simulation,
    delete_simulation,
    get_coin_performance,
    get_metrics,
    get_period_reports,
    get_signal_accuracy,
    get_simulation,
    list_simulations,
    process_cycle,
    set_status,
)
