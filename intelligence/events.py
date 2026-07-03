"""Event / catalyst tespiti — haber içeriğinden event_type + trading etkisi.

Ollama event_type verirse öncelik onda; yoksa anahtar-kelime kuralları (fallback).
Saf fonksiyonlar (DB/ağ yok).
"""
from __future__ import annotations

from typing import Dict, Optional

# event_type → (yön, risk_katkısı 0-100, not)
#   yön: +1 pozitif katalizör, -1 negatif, 0 nötr/karışık
EVENT_EFFECT = {
    "listing":       (+1, 0,  "Kısa vadeli pozitif; sonrasında sell-the-news riski"),
    "futures_listing": (+1, 5, "Vadeli listeleme — pozitif ama volatilite artar"),
    "delisting":     (-1, 70, "Güçlü negatif — likidite/erişim kaybı"),
    "regulatory":    (-1, 45, "Regülasyon/dava riski"),
    "enforcement_action": (-1, 60, "Yaptırım — güçlü negatif"),
    "hack":          (-1, 90, "Hack/exploit — güçlü negatif, alım engelle/pozisyon azalt"),
    "exploit":       (-1, 90, "Exploit — güçlü negatif"),
    "token_unlock":  (-1, 35, "Unlock arzı — negatif baskı riski"),
    "partnership":   (+1, 0,  "Ortaklık — pozitif (doğrulama gerek)"),
    "upgrade":       (+1, 5,  "Mainnet/upgrade — beklentiyle yükseliş, event sonrası satış riski"),
    "airdrop":       (+1, 10, "Airdrop — kısa vadeli ilgi"),
    "burn":          (+1, 0,  "Burn — arz azaltıcı"),
    "staking_update": (0, 5,  "Staking değişikliği"),
    "governance_vote": (0, 5, "Yönetişim oylaması"),
    "roadmap_milestone": (+1, 0, "Yol haritası kilometre taşı"),
    "etf":           (+1, 0,  "ETF/kurumsal — pozitif"),
    "general":       (0, 0,   ""),
    "unknown":       (0, 0,   ""),
}

# Anahtar-kelime fallback (Ollama yoksa)
_KEYWORDS = [
    ("hack", ["hack", "exploit", "drained", "stolen", "breach", "rug pull", "rugpull"]),
    ("delisting", ["delist", "delisting", "will remove", "trading pair removal", "suspend trading"]),
    ("listing", ["will list", "lists ", "listing", "new listing", "gets listed", "adds support for"]),
    ("regulatory", ["sec ", "cftc", "lawsuit", "regulator", "regulation", "sues", "charges", "mica"]),
    ("token_unlock", ["token unlock", "unlock", "vesting", "cliff unlock"]),
    ("upgrade", ["mainnet", "hard fork", "upgrade", "protocol upgrade"]),
    ("partnership", ["partners with", "partnership", "integrates", "collaborat"]),
    ("etf", ["etf", "spot etf"]),
    ("airdrop", ["airdrop"]),
    ("burn", ["token burn", "burns "]),
]


def detect_event(item: Dict, ollama: Optional[Dict]) -> str:
    """event_type belirle: önce Ollama, sonra anahtar-kelime; hiçbiri yoksa 'general'."""
    if ollama:
        et = (ollama.get("event_type") or "").strip().lower()
        if et and et in EVENT_EFFECT:
            return et
    text = f"{item.get('title', '')} {item.get('summary', '') or item.get('content', '')}".lower()
    for event_type, kws in _KEYWORDS:
        if any(k in text for k in kws):
            return event_type
    return "general"


def event_direction(event_type: str) -> int:
    return EVENT_EFFECT.get(event_type, (0, 0, ""))[0]


def event_risk(event_type: str) -> float:
    """Bu event tipinin risk skoruna katkısı (0-100)."""
    return float(EVENT_EFFECT.get(event_type, (0, 0, ""))[1])


def event_note(event_type: str) -> str:
    return EVENT_EFFECT.get(event_type, (0, 0, ""))[2]
