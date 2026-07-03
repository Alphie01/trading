"""Risk yönetimi + trade güvenlik katmanı (Faz 7).

GÜVENLİK İLKESİ (fail-closed):
- Otomasyon motoru KENDİLİĞİNDEN canlı işlem AÇMAZ.
- Canlı işlem için TÜM şartlar sağlanmalı:
    AUTO_TRADE_ENABLED=true  VE  BINANCE_TESTNET=false (açıkça)  VE  risk manager onayı
    VE min_confidence  VE  max_risk  VE  günlük işlem/pozisyon limitleri  VE zorunlu stop-loss.
- Her karar `trade_decisions` tablosuna AUDIT olarak yazılır (decided=canlı açıldı mı).
- Varsayılan: `decided=False` (yalnız sinyal/karar kaydı; canlı emir YOK).

Bu modül trade KARARINI verir ve audit'ler; canlı emir gönderimi (BinanceTrader.execute_signal)
yalnız `should_execute_live()` True dönerse ve çağıran katman açıkça isterse yapılır.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .config import AutomationConfig as C


def _csv_set(name: str, default: str = "") -> set:
    return set(s.strip().upper() for s in os.getenv(name, default).split(",") if s.strip())


class RiskManager:
    def __init__(self):
        self.max_position_size_percent = float(os.getenv("RISK_MAX_POSITION_PERCENT", "5"))
        self.max_daily_loss_percent = float(os.getenv("RISK_MAX_DAILY_LOSS_PERCENT", "5"))
        self.max_daily_trades = int(os.getenv("RISK_MAX_DAILY_TRADES", "10"))
        self.max_open_positions = int(os.getenv("RISK_MAX_OPEN_POSITIONS", "5"))
        self.min_confidence = float(os.getenv("RISK_MIN_CONFIDENCE", "0.6"))
        self.max_risk_score = float(os.getenv("RISK_MAX_RISK_SCORE", "60"))
        self.cooldown_after_loss_minutes = int(os.getenv("RISK_COOLDOWN_AFTER_LOSS_MIN", "60"))
        self.require_stop_loss = os.getenv("RISK_REQUIRE_STOP_LOSS", "true").lower() != "false"
        self.allowed_symbols = _csv_set("RISK_ALLOWED_SYMBOLS")   # boş = hepsi (blocked hariç)
        self.blocked_symbols = _csv_set("RISK_BLOCKED_SYMBOLS")

    def evaluate(self, signal: Dict, *, open_positions: int = 0, daily_trades: int = 0,
                 daily_loss_percent: float = 0.0, has_stop_loss: bool = True) -> Dict:
        """Sinyali risk kurallarına göre değerlendirir. Canlı işlem AÇMAZ; karar + gerekçe döner."""
        symbol = (signal.get("symbol") or "").upper()
        rec = signal.get("recommendation", "HOLD")
        sig = signal.get("signal", "HOLD")
        conf = float(signal.get("confidence") or 0)
        risk = float(signal.get("risk_score") or 100)

        reasons: List[str] = []
        approved = True

        if sig == "HOLD":
            return {"approved": False, "action": "skip", "reasons": ["signal_hold"], "symbol": symbol}

        if self.blocked_symbols and symbol in self.blocked_symbols:
            approved = False; reasons.append("blocked_symbol")
        if self.allowed_symbols and symbol not in self.allowed_symbols:
            approved = False; reasons.append("not_in_allowed_symbols")
        if conf < self.min_confidence:
            approved = False; reasons.append(f"low_confidence:{conf:.2f}<{self.min_confidence}")
        if risk > self.max_risk_score:
            approved = False; reasons.append(f"risk_too_high:{risk:.0f}>{self.max_risk_score:.0f}")
        if open_positions >= self.max_open_positions:
            approved = False; reasons.append(f"max_open_positions:{open_positions}>={self.max_open_positions}")
        if daily_trades >= self.max_daily_trades:
            approved = False; reasons.append(f"max_daily_trades:{daily_trades}>={self.max_daily_trades}")
        if daily_loss_percent >= self.max_daily_loss_percent:
            approved = False; reasons.append(f"daily_loss_limit:{daily_loss_percent:.1f}%")
        if self.require_stop_loss and not has_stop_loss:
            approved = False; reasons.append("stop_loss_required")

        action = "skip"
        if approved:
            action = "long" if sig == "BUY" else ("short" if sig == "SELL" else "skip")
            reasons.append("risk_checks_passed")

        # Pozisyon büyüklüğü önerisi (sermaye yüzdesi, güven ile ölçekli)
        position_size_percent = round(
            min(self.max_position_size_percent, self.max_position_size_percent * conf), 4
        ) if approved else 0.0

        return {
            "symbol": symbol,
            "approved": approved,
            "action": action,
            "recommendation": rec,
            "confidence": conf,
            "risk_score": risk,
            "position_size_percent": position_size_percent,
            "reasons": reasons,
        }


def testnet_enabled() -> bool:
    return os.getenv("BINANCE_TESTNET", "true").lower() != "false"


def should_execute(decision: Dict) -> bool:
    """İşlem (testnet VEYA canlı) açılabilir mi? = onaylı VE AUTO_TRADE_ENABLED.

    fail-closed: AUTO_TRADE_ENABLED=false ise HİÇBİR işlem açılmaz (testnet dahil).
    Testnet mi canlı mı, BINANCE_TESTNET belirler (bkz. should_execute_live).
    """
    return bool(decision.get("approved")) and C.TRADE_ENABLED


def should_execute_live(decision: Dict) -> bool:
    """GERÇEK PARA işlemi SADECE tüm şartlar sağlanınca True.

    fail-closed: onaylı DEĞİLse, AUTO_TRADE_ENABLED=false ise VEYA testnet=true ise → False.
    """
    if not should_execute(decision):
        return False
    if testnet_enabled():  # testnet açıksa gerçek para değil
        return False
    return True


def audit_decision(schema: Optional[str], decision: Dict, decided_live: bool = False) -> None:
    """Kararı tenant `trade_decisions` tablosuna AUDIT olarak yazar (canlı açılsa da açılmasa da)."""
    if not schema:
        return
    try:
        from trading_db.models_tenant import TradeDecision
        from trading_db.session import get_tenant_session
        with get_tenant_session(schema) as s:
            s.add(TradeDecision(
                symbol=decision.get("symbol"),
                action=decision.get("action", "skip"),
                decided=decided_live,
                reason=", ".join(decision.get("reasons", [])),
                confidence=decision.get("confidence"),
                risk_score=decision.get("risk_score"),
                is_testnet=testnet_enabled(),
                is_simulated=not decided_live,
                details={"position_size_percent": decision.get("position_size_percent")},
            ))
    except Exception as e:
        print(f"❌ audit_decision hatası: {e}")


def process_signal(signal: Dict, schema: Optional[str] = None, *, open_positions: int = 0,
                   daily_trades: int = 0, has_stop_loss: bool = True) -> Dict:
    """Sinyali risk manager'dan geçir, AUDIT'le, canlı-uygunluğu döndür (canlı emri BURADA GÖNDERMEZ)."""
    rm = RiskManager()
    decision = rm.evaluate(signal, open_positions=open_positions, daily_trades=daily_trades,
                           has_stop_loss=has_stop_loss)
    live_ok = should_execute_live(decision)
    # Bu build canlı emir göndermez → decided_live=False (audit yalnız karar)
    audit_decision(schema, decision, decided_live=False)
    decision["live_execution_allowed"] = live_ok
    decision["testnet"] = testnet_enabled()
    decision["auto_trade_enabled"] = C.TRADE_ENABLED
    return decision
