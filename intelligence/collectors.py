"""Haber toplayıcılar — RSS (feedparser) + Binance duyuru JSON + Reddit (sosyal).

İlkeler:
- Prod'da SAHTE veri YOK. Bir kaynağın hatası izole edilir (sistemi çökertmez).
- Her HTTP çağrısı timeout'lu. Sembol-alaka filtresi (base sembol + coin adı).
- `data_source` etiketlenir ('rss'|'binance_api'|'reddit'); is_mock YOK.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import source_registry as reg
from .config import IntelligenceConfig as C

logger = logging.getLogger("intelligence.collectors")

_UA = {"User-Agent": "Mozilla/5.0 (compatible; PaeraIntel/1.0; +https://paera.local)"}

# Sembol → coin adı (alaka filtresi için); bilinmeyen sembolde yalnız base kullanılır
_NAME = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "bnb", "XRP": "xrp",
    "ADA": "cardano", "DOGE": "dogecoin", "AVAX": "avalanche", "DOT": "polkadot",
    "MATIC": "polygon", "POL": "polygon", "LINK": "chainlink", "LTC": "litecoin",
    "TRX": "tron", "ATOM": "cosmos", "UNI": "uniswap", "ARB": "arbitrum", "OP": "optimism",
    "APT": "aptos", "SUI": "sui", "NEAR": "near", "FIL": "filecoin", "INJ": "injective",
    "FET": "fetch", "RNDR": "render", "RENDER": "render", "AAVE": "aave", "MKR": "maker",
    "SHIB": "shiba", "PEPE": "pepe", "TON": "toncoin", "XLM": "stellar", "ALGO": "algorand",
}


def _base_of(symbol: str) -> str:
    return (symbol or "").upper().split("/")[0].strip()


def relevant_to_symbol(text: str, symbol: str) -> bool:
    """Metin sembole alakalı mı? base sembol (kelime sınırı) veya coin adı geçiyorsa True."""
    if not symbol:
        return True
    base = _base_of(symbol)
    if not base:
        return True
    t = (text or "").lower()
    if re.search(rf"\b{re.escape(base.lower())}\b", t):
        return True
    name = _NAME.get(base)
    if name and name in t:
        return True
    return False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _within_lookback(dt: Optional[datetime], lookback_hours: int) -> bool:
    if dt is None:
        return True  # tarih yoksa ele — freshness=50 ile işlenir
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (_now() - dt).total_seconds() <= lookback_hours * 3600 + 3600


# ---------------------------------------------------------------------------- #
# RSS
# ---------------------------------------------------------------------------- #
def collect_rss(source: reg.Source, lookback_hours: int, max_items: int, timeout: int) -> List[Dict]:
    try:
        import requests
        import feedparser
    except Exception as e:
        logger.warning("RSS toplayıcı bağımlılığı yok (%s) — %s atlandı", e, source.name)
        return []
    try:
        resp = requests.get(source.url, headers=_UA, timeout=timeout)
        if resp.status_code != 200:
            logger.info("RSS %s HTTP %s — atlandı", source.name, resp.status_code)
            return []
        feed = feedparser.parse(resp.content)
    except Exception as e:
        logger.info("RSS %s hata: %s — atlandı", source.name, e)
        return []

    items: List[Dict] = []
    for entry in (feed.entries or [])[: max_items * 2]:
        published = None
        pp = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
        if pp:
            try:
                published = datetime.fromtimestamp(time.mktime(pp), tz=timezone.utc)
            except Exception:
                published = None
        if not _within_lookback(published, lookback_hours):
            continue
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
        summary = re.sub(r"<[^>]+>", " ", summary)  # HTML strip (kaba)
        items.append({
            "title": getattr(entry, "title", "").strip(),
            "url": getattr(entry, "link", ""),
            "summary": summary.strip()[:2000],
            "published_at": published,
            "source": source.name,
            "source_tier": source.tier,
            "category": source.category,
            "data_source": "rss",
        })
        if len(items) >= max_items:
            break
    return items


# ---------------------------------------------------------------------------- #
# Binance duyuruları (JSON API — best-effort; geo-block olabilir)
# ---------------------------------------------------------------------------- #
def collect_binance_ann(source: reg.Source, lookback_hours: int, max_items: int, timeout: int) -> List[Dict]:
    try:
        import requests
    except Exception:
        return []
    try:
        resp = requests.get(source.url, headers=_UA, timeout=timeout)
        if resp.status_code != 200:
            logger.info("Binance ann HTTP %s — atlandı", resp.status_code)
            return []
        data = resp.json()
    except Exception as e:
        logger.info("Binance ann hata: %s — atlandı", e)
        return []

    # Yanıt yapısı değişebilir → defensive: 'title' içeren dict listelerini topla
    articles = []
    d = data.get("data") if isinstance(data, dict) else None
    if isinstance(d, dict):
        if isinstance(d.get("catalogs"), list):
            for cat in d["catalogs"]:
                articles += (cat.get("articles") or [])
        if isinstance(d.get("articles"), list):
            articles += d["articles"]
    items: List[Dict] = []
    for a in articles[:max_items]:
        code = a.get("code")
        rel = a.get("releaseDate")
        published = None
        if isinstance(rel, (int, float)):
            try:
                published = datetime.fromtimestamp(rel / 1000.0, tz=timezone.utc)
            except Exception:
                published = None
        items.append({
            "title": (a.get("title") or "").strip(),
            "url": f"https://www.binance.com/en/support/announcement/{code}" if code else "",
            "summary": (a.get("title") or "").strip(),
            "published_at": published,
            "source": source.name,
            "source_tier": source.tier,
            "category": source.category,
            "data_source": "binance_api",
        })
    return items


# ---------------------------------------------------------------------------- #
# Reddit (sosyal — Faz 5'te momentum; bu pass'te news collection'a dahil değil)
# ---------------------------------------------------------------------------- #
def collect_reddit(symbol: str, max_items: int, timeout: int) -> List[Dict]:
    try:
        import requests
    except Exception:
        return []
    base = _base_of(symbol)
    out: List[Dict] = []
    for sub in ("cryptocurrency", "CryptoMarkets"):
        try:
            r = requests.get(f"https://www.reddit.com/r/{sub}/hot.json?limit=25",
                             headers=_UA, timeout=timeout)
            if r.status_code != 200:
                continue
            for ch in (r.json().get("data", {}).get("children", []) or []):
                d = ch.get("data", {})
                title = d.get("title", "")
                if base and not relevant_to_symbol(title, symbol):
                    continue
                out.append({
                    "title": title, "url": "https://reddit.com" + d.get("permalink", ""),
                    "summary": (d.get("selftext") or "")[:500], "published_at": None,
                    "source": "Reddit", "source_tier": 4, "category": "social",
                    "data_source": "reddit", "score": d.get("score", 0),
                })
                if len(out) >= max_items:
                    return out
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------- #
# Orkestrasyon
# ---------------------------------------------------------------------------- #
_DISPATCH = {"rss": collect_rss, "binance_ann": collect_binance_ann}


def collect_for_symbol(symbol: Optional[str] = None) -> List[Dict]:
    """Toplayıcısı olan (sosyal-olmayan) tüm kaynaklardan haber topla, sembole göre filtrele."""
    lookback = C.NEWS_LOOKBACK_HOURS
    per_src = C.MAX_ITEMS_PER_SOURCE
    timeout = C.COLLECT_TIMEOUT
    all_items: List[Dict] = []
    for src in reg.collectable_sources(include_social=False):
        fn = _DISPATCH.get(src.collector)
        if fn is None:
            continue
        try:
            items = fn(src, lookback, per_src, timeout)
        except Exception as e:
            logger.info("Toplayıcı %s hata: %s — atlandı", src.name, e)
            continue
        for it in items:
            text = f"{it.get('title', '')} {it.get('summary', '')}"
            if symbol and not relevant_to_symbol(text, symbol):
                continue
            all_items.append(it)
    return all_items
