"""Duplicate haber tespiti (Faz 4: basit, ağır embedding YOK).

- content_hash: (source + normalize(title) + url) → aynı makalenin tekrar toplanmasını engeller (idempotent).
- duplicate_group_id: normalize(title) → FARKLI kaynakların AYNI haberi kopyalamasını gruplar.
  Böylece "20 haber bulundu ama 17'si aynı Binance duyurusunu kopyalamış → 3 bağımsız haber" hesaplanır.

Saf fonksiyonlar (DB/ağ yok) → test edilebilir.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
# Sık başlık ön-ekleri/gürültü (gruplamayı iyileştirir)
_PREFIX = re.compile(r"^(breaking|exclusive|report|update|just in|watch|opinion)\s*[:\-]\s*", re.IGNORECASE)


def normalize_title(title: str) -> str:
    t = (title or "").strip().lower()
    t = _PREFIX.sub("", t)
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def duplicate_group_id(title: str) -> str:
    """Normalize edilmiş başlık üzerinden kaynaklar-arası grup kimliği (16 hex)."""
    return _sha1(normalize_title(title))[:16]


def content_hash(item: Dict) -> str:
    """Bir makale örneğinin benzersiz kimliği (source+title+url) — 40 hex sha1."""
    src = (item.get("source") or "").strip().lower()
    url = (item.get("url") or "").strip().lower()
    return _sha1(f"{src}|{normalize_title(item.get('title', ''))}|{url}")


def annotate_and_group(items: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Her item'a content_hash + duplicate_group_id ekler; grup istatistiği döner.

    Dönüş: (items, stats) — stats: {total, independent, duplicates, groups:{gid:count}}
    Aynı content_hash (tam tekrar) elenir (bir kez tutulur).
    """
    seen_hashes = set()
    unique_items: List[Dict] = []
    groups: Dict[str, int] = {}
    for it in items:
        ch = content_hash(it)
        if ch in seen_hashes:
            continue
        seen_hashes.add(ch)
        gid = duplicate_group_id(it.get("title", ""))
        it = dict(it)
        it["content_hash"] = ch
        it["duplicate_group_id"] = gid
        unique_items.append(it)
        groups[gid] = groups.get(gid, 0) + 1

    total = len(unique_items)
    independent = len(groups)
    stats = {
        "total": total,
        "independent": independent,
        "duplicates": max(0, total - independent),
        "groups": groups,
    }
    return unique_items, stats


def representative_items(items: List[Dict]) -> List[Dict]:
    """Her duplicate grubundan bir temsilci (ilk görülen) — LLM'e yalnız bunlar gider."""
    seen = set()
    reps = []
    for it in items:
        gid = it.get("duplicate_group_id") or duplicate_group_id(it.get("title", ""))
        if gid in seen:
            continue
        seen.add(gid)
        reps.append(it)
    return reps
