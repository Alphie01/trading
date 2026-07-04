"""Pozisyon mark-to-market + SL/TP/trailing/liquidation çıkış kararı (saf fonksiyonlar).

Otonom çıkış (Faz-sonrası): kullanıcı SL/TP GİRMESE bile, `auto_exit` (default AÇIK) ile
sistem kendi kâr-al / zarar-kes / trailing-stop uygular → pozisyonu kendi yönetir.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from . import pnl

LONG_LIKE = ("LONG", "SPOT")

# Otonom çıkış varsayılanları (config bunları override edebilir; kullanıcı SL/TP girmezse devreye girer)
AUTO_STOP_LOSS_PCT = 10.0      # zarar kes: -%10
AUTO_TAKE_PROFIT_PCT = 40.0    # sert kâr-al tavanı: +%40 (trailing çoğu kazananı daha önce yakalar)
AUTO_TRAILING_PCT = 8.0        # tepe noktadan %8 geri çekilirse çık (kazancı kilitler)


def track_peak(pos: Dict, current_price: float) -> float:
    """Pozisyonun giriş sonrası en LEHTE fiyatını (long→max, short→min) döndürür (metadata peak_price ile)."""
    side = pos["side"].upper()
    md = pos.get("metadata") or {}
    peak = md.get("peak_price")
    peak = float(peak) if peak not in (None, "") else float(pos["entry_price"])
    if side in LONG_LIKE:
        return max(peak, current_price)
    return min(peak, current_price)


def unrealized_pnl(pos: Dict, current_price: float) -> float:
    """Açık pozisyonun anlık (gerçekleşmemiş) PnL'i."""
    side = pos["side"].upper()
    if pos["mode"].upper() == "FUTURES":
        return pnl.futures_unrealized(side, float(pos["entry_price"]), current_price, float(pos["notional_value"] or 0))
    return pnl.spot_unrealized(float(pos["entry_price"]), current_price, float(pos["quantity"]))


def check_exit(pos: Dict, current_price: float, config: Dict) -> Tuple[bool, Optional[str]]:
    """SL / TP / trailing-stop / liquidation kontrolü. (should_exit, reason).

    Kullanıcı SL/TP girmezse ve `auto_exit` açıksa (default), auto default'lar + trailing devreye girer
    → sistem pozisyonu kendi kapatır (kâr-al / zarar-kes / tepe-geri-çekilme).
    """
    side = pos["side"].upper()
    entry = float(pos["entry_price"])
    is_long = side in LONG_LIKE
    auto = config.get("auto_exit", True)  # DEFAULT AÇIK — SL/TP girilmese de kendi yönetsin

    # Explicit SL/TP; yoksa auto default (auto açıkken)
    def _pct(v):
        return float(v) / 100.0 if v not in (None, "") else None
    sl = _pct(config.get("stop_loss"))
    tp = _pct(config.get("take_profit"))
    if auto:
        if sl is None:
            sl = float(config.get("auto_stop_loss_percent") or AUTO_STOP_LOSS_PCT) / 100.0
        if tp is None:
            tp = float(config.get("auto_take_profit_percent") or AUTO_TAKE_PROFIT_PCT) / 100.0

    # Liquidation (futures) — her zaman önce
    liq = pos.get("liquidation_price")
    if liq:
        liq = float(liq)
        if (is_long and current_price <= liq) or (not is_long and current_price >= liq):
            return True, "liquidation"

    # Trailing stop — tepe (en lehte) noktadan geri çekilme (kazancı kilitler).
    # trailing_stop config'te açık VEYA auto açıkken devreye girer; yalnız kâra geçmiş pozisyonda.
    if bool(config.get("trailing_stop")) or auto:
        trail = float(config.get("trailing_stop_percent") or AUTO_TRAILING_PCT) / 100.0
        md = pos.get("metadata") or {}
        peak = md.get("peak_price")
        if peak not in (None, ""):
            peak = float(peak)
            if is_long and peak > entry and current_price <= peak * (1 - trail):
                return True, f"trailing_stop:{trail*100:.1f}%"
            if not is_long and peak < entry and current_price >= peak * (1 + trail):
                return True, f"trailing_stop:{trail*100:.1f}%"

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
