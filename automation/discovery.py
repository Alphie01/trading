"""Coin keşfi — ccxt Binance ticker verisinden aday coinler.

Faz 4: yeni bağımlılık YOK (CoinGecko/CMP eklenmez); mevcut ccxt/Binance market
verisiyle çözülür. `fetch_market_tickers` test için ticker enjeksiyonu destekler.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .config import AutomationConfig as C


def fetch_market_tickers(fetcher=None) -> List[Dict]:
    """İzinli quote'lar için tüm ticker'ları çeker (data_fetcher.CryptoDataFetcher)."""
    if fetcher is None:
        from data_fetcher import CryptoDataFetcher
        fetcher = CryptoDataFetcher()

    all_tickers: List[Dict] = []
    for quote in C.ALLOWED_QUOTES:
        try:
            all_tickers.extend(fetcher.fetch_tickers(quote))
        except Exception as e:
            print(f"⚠️ {quote} ticker çekme hatası: {e}")
    return all_tickers


def annotate_discovery_reason(ticker: Dict) -> str:
    """Bir coin'in neden keşfedildiğini açıklar (hacim/momentum/volatilite)."""
    reasons = []
    qv = float(ticker.get("quote_volume") or 0)
    pct = ticker.get("percentage")
    pct = float(pct) if pct is not None else 0.0
    if qv >= C.MIN_24H_VOLUME_USDT * 5:
        reasons.append("very_high_24h_volume")
    elif qv >= C.MIN_24H_VOLUME_USDT:
        reasons.append("sufficient_24h_volume")
    if pct >= 5:
        reasons.append(f"price_breakout_up:{pct:+.1f}%")
    elif pct <= -5:
        reasons.append(f"price_drop:{pct:+.1f}%")
    high, low = ticker.get("high"), ticker.get("low")
    if high and low and float(low) > 0:
        rng = (float(high) - float(low)) / float(low) * 100
        if rng >= 15:
            reasons.append(f"elevated_volatility:{rng:.0f}%")
    return ", ".join(reasons) or "market_scan"
