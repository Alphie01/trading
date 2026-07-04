"""ccxt fetcher'ları — order book / funding rate / open interest / cross-asset korelasyon.

Graceful: her fonksiyon hata/erişimsizlikte None döner (sahte veri YOK). v2 placeholder
feature'larını (ob_imbalance/ob_spread_pct/funding_rate/oi_change/btc_corr/eth_corr) besler.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("market_data.fetchers")

try:
    from api_cache import TTLCache as _TTLCache
    _CACHE = _TTLCache(int(os.getenv("MARKET_DATA_CACHE_TTL", "30")))
except Exception:
    _CACHE = None


def _norm(symbol: str) -> str:
    return symbol if "/" in symbol else f"{symbol.upper()}/USDT"


def _spot():
    import ccxt
    return ccxt.binance({"enableRateLimit": True, "timeout": 15000})


def _futures():
    import ccxt
    return ccxt.binance({"enableRateLimit": True, "timeout": 15000,
                         "options": {"defaultType": "future"}})


# ── Saf hesaplama (ağ yok → test edilebilir) ─────────────────────────────────
def _ob_metrics(bids: Sequence, asks: Sequence) -> Optional[Dict]:
    """[[price, amount], ...] bid/ask → spread + derinlik + imbalance. Boş → None."""
    if not bids or not asks:
        return None
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    if best_bid <= 0 or best_ask <= 0:
        return None
    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2.0
    spread_pct = (spread / mid * 100.0) if mid > 0 else None
    depth_bid = float(sum(float(b[1]) for b in bids))
    depth_ask = float(sum(float(a[1]) for a in asks))
    tot = depth_bid + depth_ask
    imbalance = ((depth_bid - depth_ask) / tot) if tot > 0 else 0.0  # -1..+1 (bid ağırlıklı → +)
    return {
        "bid_ask_spread": round(spread, 8),
        "spread_percent": round(spread_pct, 5) if spread_pct is not None else None,
        "depth_bid": round(depth_bid, 4), "depth_ask": round(depth_ask, 4),
        "imbalance": round(imbalance, 4),
    }


def _return_corr(a_close: Sequence, b_close: Sequence) -> Optional[float]:
    """İki fiyat serisinin getiri korelasyonu (-1..1). <20 ortak bar → None."""
    import numpy as np
    a = np.asarray(a_close, dtype=float)
    b = np.asarray(b_close, dtype=float)
    n = min(len(a), len(b))
    if n < 21:
        return None
    a, b = a[-n:], b[-n:]
    ra = np.diff(a) / a[:-1]
    rb = np.diff(b) / b[:-1]
    if ra.std() == 0 or rb.std() == 0:
        return None
    c = float(np.corrcoef(ra, rb)[0, 1])
    return None if (c != c) else round(c, 4)  # NaN guard


# ── ccxt fetcher'ları (ağ; graceful None) ────────────────────────────────────
def order_book_features(symbol: str, limit: int = 20) -> Optional[Dict]:
    try:
        ob = _spot().fetch_order_book(_norm(symbol), limit=limit)
        return _ob_metrics(ob.get("bids") or [], ob.get("asks") or [])
    except Exception as e:
        logger.info("order_book_features atlandı %s: %s", symbol, e)
        return None


def funding_rate(symbol: str) -> Optional[Dict]:
    try:
        fr = _futures().fetch_funding_rate(_norm(symbol))
        rate = fr.get("fundingRate")
        return {"funding_rate": float(rate)} if rate is not None else None
    except Exception as e:
        logger.info("funding_rate atlandı %s: %s", symbol, e)
        return None


def open_interest_change(symbol: str, timeframe: str = "1h", limit: int = 8) -> Optional[float]:
    """Son `limit` OI noktasının ilk→son değişim oranı. Geçmiş yoksa None."""
    try:
        hist = _futures().fetch_open_interest_history(_norm(symbol), timeframe, limit=limit)
        vals: List[float] = []
        for h in (hist or []):
            v = h.get("openInterestAmount") or h.get("openInterestValue")
            if v:
                vals.append(float(v))
        if len(vals) < 2 or vals[0] == 0:
            return None
        return round((vals[-1] - vals[0]) / vals[0], 4)
    except Exception as e:
        logger.info("open_interest_change atlandı %s: %s", symbol, e)
        return None


def cross_asset_correlation(symbol: str, refs: Sequence[str] = ("BTC/USDT", "ETH/USDT"),
                            timeframe: str = "4h", days: int = 30) -> Optional[Dict]:
    """Sembolün BTC/ETH ile getiri korelasyonu. Ağ hatası → None. Backfill'lenebilir feature."""
    try:
        from data_fetcher import CryptoDataFetcher
        f = CryptoDataFetcher()
        base = f.fetch_ohlcv_data(_norm(symbol), timeframe=timeframe, days=days)
        if base is None or len(base) < 21:
            return None
        out: Dict[str, Optional[float]] = {}
        keymap = {"BTC/USDT": "btc_corr", "ETH/USDT": "eth_corr"}
        for ref in refs:
            try:
                rdf = f.fetch_ohlcv_data(ref, timeframe=timeframe, days=days)
                out[keymap.get(ref, ref)] = _return_corr(base["close"].values, rdf["close"].values) if rdf is not None else None
            except Exception:
                out[keymap.get(ref, ref)] = None
        return out if any(v is not None for v in out.values()) else None
    except Exception as e:
        logger.info("cross_asset_correlation atlandı %s: %s", symbol, e)
        return None


def build_v2_context(symbol: str) -> Dict:
    """v2 placeholder feature'ları için canlı bağlam. Eksik alanlar EKLENMEZ (builder 0-default kullanır)."""
    ctx: Dict[str, float] = {}
    ob = order_book_features(symbol)
    if ob:
        if ob.get("imbalance") is not None:
            ctx["ob_imbalance"] = ob["imbalance"]
        if ob.get("spread_percent") is not None:
            ctx["ob_spread_pct"] = ob["spread_percent"]
    fr = funding_rate(symbol)
    if fr and fr.get("funding_rate") is not None:
        ctx["funding_rate"] = fr["funding_rate"]
    oic = open_interest_change(symbol)
    if oic is not None:
        ctx["oi_change"] = oic
    corr = cross_asset_correlation(symbol)
    if corr:
        if corr.get("btc_corr") is not None:
            ctx["btc_corr"] = corr["btc_corr"]
        if corr.get("eth_corr") is not None:
            ctx["eth_corr"] = corr["eth_corr"]
    return ctx
