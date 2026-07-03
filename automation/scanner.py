"""Ön eleme (pre-filter) — bulunan her coin direkt analiz edilmez, önce filtrelerden geçer.

Saf fonksiyonlar (ağ/DB yok) → kolay test edilir.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .config import AutomationConfig as C


def prefilter(ticker: Dict) -> Tuple[bool, str, List[str]]:
    """Bir ticker dict'ini ön elemeden geçirir.

    Returns: (passed, reason, warnings)
    - reason: elenme nedeni (passed=False) veya kabul nedeni özeti (passed=True)
    - warnings: geçse bile dikkat edilecek noktalar
    """
    warnings: List[str] = []
    symbol = (ticker.get("symbol") or "").upper()
    base = (ticker.get("base") or "").upper()
    quote = (ticker.get("quote") or "").upper()

    if not symbol or "/" not in symbol:
        return False, "invalid_symbol", warnings

    # Quote filtresi
    if quote not in C.ALLOWED_QUOTES:
        return False, f"quote_not_allowed:{quote}", warnings

    # Dışlanan sembol
    if symbol in C.EXCLUDED_SYMBOLS:
        return False, "excluded_symbol", warnings

    # Stablecoin base
    if base in C.STABLE_BASES:
        return False, "stablecoin_base", warnings

    # Hacim filtresi
    quote_volume = float(ticker.get("quote_volume") or 0)
    if quote_volume < C.MIN_24H_VOLUME_USDT:
        return False, f"low_volume:{quote_volume:.0f}<{C.MIN_24H_VOLUME_USDT:.0f}", warnings

    # Fiyat / veri yeterliliği
    last = ticker.get("last")
    if not last or float(last) <= 0:
        return False, "no_price", warnings

    # Spread (likidite) kontrolü
    bid, ask = ticker.get("bid"), ticker.get("ask")
    if bid and ask and float(bid) > 0:
        spread_pct = (float(ask) - float(bid)) / float(bid) * 100
        if spread_pct > C.MAX_SPREAD_PERCENT:
            return False, f"high_spread:{spread_pct:.2f}%", warnings
        if spread_pct > C.MAX_SPREAD_PERCENT * 0.5:
            warnings.append(f"elevated_spread:{spread_pct:.2f}%")
    else:
        warnings.append("spread_unknown")

    # Aşırı volatilite uyarısı (elenmez ama işaretlenir)
    high, low = ticker.get("high"), ticker.get("low")
    if high and low and float(low) > 0:
        rng_pct = (float(high) - float(low)) / float(low) * 100
        if rng_pct > 30:
            warnings.append(f"high_volatility_24h:{rng_pct:.1f}%")

    return True, "passed", warnings


def prefilter_batch(tickers: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Toplu ön eleme. (passed_list, rejected_list) döner; her öğeye reason/warnings eklenir."""
    passed, rejected = [], []
    for t in tickers:
        ok, reason, warnings = prefilter(t)
        item = dict(t)
        item["_reason"] = reason
        item["_warnings"] = warnings
        (passed if ok else rejected).append(item)
    return passed, rejected
