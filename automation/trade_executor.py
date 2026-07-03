"""Trade yürütücü — risk manager ONAYLI kararları BinanceTrader'a bağlar.

GÜVENLİK (fail-closed, çok katmanlı):
1. `AUTO_TRADE_ENABLED=false` → HİÇBİR işlem (varsayılan).
2. `AUTO_TRADE_ENABLED=true` + `BINANCE_TESTNET=true` → yalnız TESTNET (sahte para, güvenli).
3. `AUTO_TRADE_ENABLED=true` + `BINANCE_TESTNET=false` → GERÇEK PARA (kullanıcı açıkça istedi).
4. Binance API key yoksa → işlem yok.
5. Her işlemde ZORUNLU stop-loss + risk manager onayı; her karar `trade_decisions`'a audit'lenir.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from .config import AutomationConfig as C
from .risk_manager import audit_decision, should_execute, should_execute_live, testnet_enabled


def build_trade_signal(decision: Dict, current_price: float) -> Dict:
    """Risk kararından BinanceTrader.execute_signal formatında sinyal üretir (zorunlu SL/TP)."""
    action = decision["action"]  # long | short
    stop_pct = float(os.getenv("RISK_STOP_LOSS_PERCENT", "2")) / 100.0
    tp_pct = float(os.getenv("RISK_TAKE_PROFIT_PERCENT", "4")) / 100.0
    if action == "long":
        stop_loss = current_price * (1 - stop_pct)
        target = current_price * (1 + tp_pct)
    else:  # short
        stop_loss = current_price * (1 + stop_pct)
        target = current_price * (1 - tp_pct)
    return {
        "symbol": decision["symbol"],
        "action": action,
        "entry_price": current_price,
        "target_price": target,
        "stop_loss": stop_loss,
        "confidence": decision.get("confidence"),
        "exchange_type": os.getenv("AUTO_EXCHANGE_TYPE", "futures"),
        "risk_percent": decision.get("position_size_percent"),
    }


def execute_decision(decision: Dict, current_price: Optional[float], schema: Optional[str] = None,
                     binance_trader=None) -> Dict:
    """Onaylı kararı yürütür (testnet veya canlı). fail-closed; her durumda audit yazar."""
    symbol = decision.get("symbol")

    # 1) İşlem izni yok mu? (AUTO_TRADE_ENABLED=false veya onaysız) → yalnız audit
    if not should_execute(decision):
        audit_decision(schema, decision, decided_live=False)
        return {"executed": False, "reason": "auto_trade_disabled_or_not_approved", "symbol": symbol}

    # 2) Fiyat yok
    if not current_price or float(current_price) <= 0:
        audit_decision(schema, decision, decided_live=False)
        return {"executed": False, "reason": "no_price", "symbol": symbol}

    # 3) Binance anahtarları
    api = os.getenv("BINANCE_API_KEY")
    sec = os.getenv("BINANCE_SECRET_KEY")
    if binance_trader is None:
        if not api or not sec:
            audit_decision(schema, decision, decided_live=False)
            return {"executed": False, "reason": "binance_keys_missing", "symbol": symbol}
        from binance_trader import BinanceTrader
        binance_trader = BinanceTrader(api, sec, testnet=testnet_enabled())

    is_live = should_execute_live(decision)  # gerçek para mı?
    signal = build_trade_signal(decision, float(current_price))

    try:
        result = binance_trader.execute_signal(signal)
        success = bool(result and (result.get("success", True) if isinstance(result, dict) else result))
        # decided_live: yalnız CANLI (gerçek para) + başarılı ise True
        audit_decision(schema, decision, decided_live=(success and is_live))
        mode = "LIVE" if is_live else "TESTNET"
        print(f"{'🟢' if success else '🔴'} Trade [{mode}] {signal['action']} {symbol} → success={success}")
        return {"executed": success, "live": is_live, "testnet": testnet_enabled(),
                "signal": signal, "result": result, "symbol": symbol}
    except Exception as e:
        audit_decision(schema, decision, decided_live=False)
        print(f"❌ Trade execution hatası {symbol}: {e}")
        return {"executed": False, "reason": str(e), "symbol": symbol}
