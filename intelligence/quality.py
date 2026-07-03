"""Haber kalite/alaka/etki/tazelik/hype/risk skorları + sembol-bazlı toplulaştırma.

Skorlar kaynak reputation'ı + (varsa) Ollama çıktısı + kural tabanlı sinyallerin birleşimidir.
Saf fonksiyonlar (DB/ağ yok); vaderSentiment yalnız gerektiğinde lazy import edilir.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import events as ev

_HYPE_WORDS = re.compile(
    r"\b(moon|100x|1000x|to the moon|pump|ape in|next \w+ killer|guaranteed|explode|parabolic|"
    r"don'?t miss|last chance|gem|lambo|🚀|🌙)\b", re.IGNORECASE)


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def freshness_score(published_at, decay_hours: float = 48.0, now: Optional[datetime] = None) -> float:
    """0h → 100, decay_hours → 0 (lineer). Bilinmeyen tarih → 50 (nötr)."""
    if not isinstance(published_at, datetime):
        return 50.0
    now = now or datetime.now(timezone.utc)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    age_h = (now - published_at).total_seconds() / 3600.0
    if age_h < 0:
        age_h = 0.0
    return _clamp(100.0 - (age_h / decay_hours) * 100.0)


def vader_compound(text: str) -> float:
    """VADER compound (-1..1). vaderSentiment yoksa 0.0 (nötr)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return float(SentimentIntensityAnalyzer().polarity_scores(text or "")["compound"])
    except Exception:
        return 0.0


def _hype_from_text(text: str) -> float:
    hits = len(_HYPE_WORDS.findall(text or ""))
    return _clamp(hits * 22.0)


def item_scores(item: Dict, reputation: float, impact_weight: float,
                ollama: Optional[Dict], category: str = "crypto_news") -> Dict:
    """Bir haber öğesi için skor seti üretir. `ollama` None olabilir (graceful)."""
    text = f"{item.get('title', '')} {item.get('summary', '') or item.get('content', '')}"
    fresh = freshness_score(item.get("published_at"))
    event_type = ev.detect_event(item, ollama)

    # Sentiment (0-100): Ollama > VADER baseline
    if ollama and ollama.get("sentiment_score") is not None:
        sent = _clamp(float(ollama.get("sentiment_score", 50)))
    else:
        sent = _clamp(50.0 + vader_compound(text) * 50.0)
    sent_label = "positive" if sent >= 60 else ("negative" if sent <= 40 else "neutral")

    # Relevance: Ollama > heuristic
    if ollama and ollama.get("relevance_score") is not None:
        relevance = _clamp(float(ollama.get("relevance_score", 50)))
        if ollama.get("is_relevant_to_symbol") is False:
            relevance = min(relevance, 30.0)
    else:
        relevance = 60.0  # kaynak zaten sembol filtresinden geçti

    # Impact: Ollama + kaynak ağırlığı + event
    src_impact = impact_weight * 100.0
    if ollama and ollama.get("impact_score") is not None:
        impact = _clamp(0.6 * float(ollama.get("impact_score", 0)) + 0.4 * src_impact)
    else:
        impact = _clamp(src_impact)
    if ev.event_risk(event_type) >= 60 or event_type in ("listing", "delisting", "hack", "etf"):
        impact = _clamp(max(impact, 80.0))

    # Hype: Ollama > metin
    if ollama and ollama.get("hype_score") is not None:
        hype = _clamp(float(ollama.get("hype_score", 0)))
    else:
        hype = _hype_from_text(text)
    if category in ("regional",):
        hype = _clamp(hype + 5)

    # Risk: event riski + güvenlik kaynağı + negatif sentiment + hype
    risk = ev.event_risk(event_type)
    if category == "security":
        risk = _clamp(max(risk, 75.0))
    if sent <= 35:
        risk += 15
    risk += hype * 0.2
    if ollama and ollama.get("risk_flags"):
        risk += 10
    risk = _clamp(risk)

    # Quality: yüksek-reputation + alakalı + taze + düşük-hype
    quality = _clamp(0.4 * reputation + 0.3 * relevance + 0.2 * fresh + 0.1 * (100 - hype))

    return {
        "sentiment": sent_label,
        "sentiment_score": round(sent, 2),
        "relevance_score": round(relevance, 2),
        "impact_score": round(impact, 2),
        "freshness_score": round(fresh, 2),
        "hype_score": round(hype, 2),
        "risk_score": round(risk, 2),
        "quality_score": round(quality, 2),
        "event_type": event_type,
    }


def _wmean(pairs: List) -> Optional[float]:
    pairs = [(w, v) for (w, v) in pairs if v is not None and w > 0]
    tw = sum(w for w, _ in pairs)
    return (sum(w * v for w, v in pairs) / tw) if tw > 0 else None


def aggregate(scored: List[Dict], stats: Dict) -> Dict:
    """Skorlanmış (temsilci/deduped) öğeleri sembol-bazlı snapshot'a topla.

    scored öğeleri item_scores + {reputation, source, event_type, title, url} içerir.
    """
    if not scored:
        return {
            "news_quality_score": None, "hype_risk": None, "news_sentiment": "neutral",
            "news_risk": 0.0, "event_summary": [], "top_news": [], "source_breakdown": {},
        }

    # Reputation-ağırlıklı ortalamalar
    news_quality = _wmean([(s["reputation"], s["quality_score"]) for s in scored])
    news_sent = _wmean([(s["reputation"], s["sentiment_score"]) for s in scored]) or 50.0
    hypes = [s["hype_score"] for s in scored]
    hype_risk = _clamp(0.5 * (sum(hypes) / len(hypes)) + 0.5 * max(hypes)) if hypes else 0.0

    # En yüksek event riski + negatif haber → news_risk (scoring risk'e katkı)
    max_event_risk = max((ev.event_risk(s["event_type"]) for s in scored), default=0.0)
    neg_ratio = sum(1 for s in scored if s["sentiment_score"] <= 40) / len(scored)
    news_risk = _clamp(max_event_risk * 0.7 + neg_ratio * 30.0)

    # Kalite dup-cezası: bağımsız haber az ise (herkes aynı şeyi kopyalıyor) kaliteyi hafif düşür
    total = max(1, stats.get("total", len(scored)))
    independent = max(1, stats.get("independent", len(scored)))
    indep_ratio = independent / total
    if news_quality is not None and indep_ratio < 0.5:
        news_quality = _clamp(news_quality * (0.85 + 0.15 * indep_ratio))

    # Event özeti (tip → adet + risk)
    ev_counts: Dict[str, int] = {}
    for s in scored:
        et = s["event_type"]
        if et not in ("general", "unknown"):
            ev_counts[et] = ev_counts.get(et, 0) + 1
    event_summary = [
        {"event_type": et, "count": c, "risk": ev.event_risk(et),
         "direction": ev.event_direction(et), "note": ev.event_note(et)}
        for et, c in sorted(ev_counts.items(), key=lambda kv: -kv[1])
    ]

    # En dikkat çekici haberler (impact × freshness)
    top = sorted(scored, key=lambda s: (s["impact_score"] * 0.7 + s["freshness_score"] * 0.3), reverse=True)[:6]
    top_news = [{
        "title": s.get("title"), "url": s.get("url"), "source": s.get("source"),
        "sentiment": s["sentiment"], "impact_score": s["impact_score"],
        "event_type": s["event_type"], "quality_score": s["quality_score"],
        "summary": (s.get("ollama") or {}).get("summary"),
    } for s in top]

    # Kaynak dağılımı
    src_breakdown: Dict[str, int] = {}
    for s in scored:
        src_breakdown[s.get("source", "?")] = src_breakdown.get(s.get("source", "?"), 0) + 1

    return {
        "news_quality_score": round(news_quality, 2) if news_quality is not None else None,
        "hype_risk": round(hype_risk, 2),
        "news_sentiment": "positive" if news_sent >= 60 else ("negative" if news_sent <= 40 else "neutral"),
        "news_sentiment_score": round(news_sent, 2),
        "news_risk": round(news_risk, 2),
        "event_summary": event_summary,
        "top_news": top_news,
        "source_breakdown": src_breakdown,
    }
