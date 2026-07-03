"""Katmanlı kaynak kayıt defteri — KOD doğruluk kaynağıdır (DB'ye repository ile senkronlanır).

Her kaynak: tier (1 en güvenilir .. 4 sosyal), reputation (0-100), impact_weight (0-1), collector tipi.
`collector`:
- 'rss'         → RSS/Atom feed (feedparser)
- 'binance_ann' → Binance duyuru JSON API
- 'reddit'      → Reddit public JSON (sembol bazlı, sosyal)
- None          → ALTYAPI HAZIR ama bu pass'te toplayıcı yok (registry'de tanımlı; ileride eklenecek)

Yeni kaynak eklemek: bu listeye bir Source ekle. Toplayıcısı olan (collector != None) kaynaklar
otomatik toplanır; None olanlar dashboard/registry'de görünür ama veri çekilmez.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Source:
    name: str
    category: str           # exchange_official | regulator | crypto_news | security | social | regional
    tier: int               # 1..4
    reputation: float       # 0-100
    impact_weight: float    # 0-1
    collector: Optional[str]  # 'rss' | 'binance_ann' | 'reddit' | None
    url: str = ""
    regional: bool = False
    allow_trade_signal_boost: bool = True
    enabled: bool = True


# Not: RSS URL'leri değişebilir; toplayıcı hataları izole edilir (bir kaynağın düşmesi sistemi çökertmez).
REGISTRY: List[Source] = [
    # ── Tier 1 — Resmi borsa & regülasyon (en yüksek ağırlık) ──────────────────────────────
    Source("Binance Announcements", "exchange_official", 1, 98, 1.00, "binance_ann",
           "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&pageNo=1&pageSize=20"),
    Source("Coinbase Blog", "exchange_official", 1, 92, 0.90, "rss", "https://www.coinbase.com/blog/rss.xml"),
    Source("SEC Press Releases", "regulator", 1, 96, 0.95, "rss", "https://www.sec.gov/news/pressreleases.rss"),
    Source("CFTC Press Releases", "regulator", 1, 94, 0.92, "rss", "https://www.cftc.gov/RSS/RSSGP/rssgp.xml"),
    # Altyapı-hazır (collector=None): tier tanımlı, ileride toplayıcı eklenecek
    Source("OKX Announcements", "exchange_official", 1, 90, 0.85, None),
    Source("Bybit Announcements", "exchange_official", 1, 88, 0.82, None),
    Source("Kraken Blog", "exchange_official", 1, 90, 0.85, None),

    # ── Tier 2 — Güvenilir kripto/finans haberleri ────────────────────────────────────────
    Source("CoinDesk", "crypto_news", 2, 82, 0.75, "rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    Source("The Block", "crypto_news", 2, 80, 0.72, "rss", "https://www.theblock.co/rss.xml"),
    Source("Decrypt", "crypto_news", 2, 76, 0.68, "rss", "https://decrypt.co/feed"),
    Source("Bitcoin Magazine", "crypto_news", 2, 74, 0.66, "rss", "https://bitcoinmagazine.com/feed"),
    Source("Reuters", "crypto_news", 2, 85, 0.78, None),
    Source("Bloomberg", "crypto_news", 2, 85, 0.78, None),

    # ── Tier 3 — Hızlı ama kalite-kontrol isteyen (Ollama şart) ────────────────────────────
    Source("Cointelegraph", "crypto_news", 3, 58, 0.55, "rss", "https://cointelegraph.com/rss"),
    Source("BeInCrypto", "crypto_news", 3, 52, 0.48, "rss", "https://beincrypto.com/feed/"),
    Source("CryptoSlate", "crypto_news", 3, 55, 0.50, "rss", "https://cryptoslate.com/feed/"),

    # ── Güvenlik / hack (risk sinyali güçlü) ──────────────────────────────────────────────
    Source("Rekt News", "security", 1, 85, 0.90, "rss", "https://rekt.news/rss/feed.xml"),
    Source("PeckShield", "security", 1, 88, 0.92, None),
    Source("CertiK Alert", "security", 1, 86, 0.90, None),

    # ── Regional (TR) — global trade'de düşük ağırlık, dashboard akışı için ────────────────
    Source("CoinTürk", "regional", 4, 45, 0.20, "rss", "https://cointurk.com/feed", regional=True),
    Source("Uzmancoin", "regional", 4, 40, 0.18, "rss", "https://uzmancoin.com/feed/", regional=True),

    # ── Sosyal (Faz 5'te momentum/hype; haber değil) ──────────────────────────────────────
    Source("Reddit", "social", 4, 40, 0.30, "reddit", "", allow_trade_signal_boost=False),
]

_BY_NAME = {s.name: s for s in REGISTRY}


def all_sources() -> List[Source]:
    return list(REGISTRY)


def collectable_sources(include_social: bool = False) -> List[Source]:
    """Toplayıcısı olan + enabled kaynaklar (varsayılan: sosyal hariç — sosyal Faz 5)."""
    out = []
    for s in REGISTRY:
        if not s.enabled or s.collector is None:
            continue
        if s.category == "social" and not include_social:
            continue
        out.append(s)
    return out


def get(name: str) -> Optional[Source]:
    return _BY_NAME.get(name)


def reputation(name: str, default: float = 50.0) -> float:
    s = _BY_NAME.get(name)
    return float(s.reputation) if s else default


def impact_weight(name: str, default: float = 0.4) -> float:
    s = _BY_NAME.get(name)
    return float(s.impact_weight) if s else default
