"""Market Intelligence orkestrasyonu.

build_snapshot(symbol): collect → dedup → baseline sentiment (VADER) → Ollama (deduped top-N) →
kalite/event skorları → topla → persist (shared DB) → cache. Ağır (network+LLM) → best-effort + cache.

Graceful: kaynak yoksa/boşsa None; Ollama yoksa kural-tabanlı skorlarla devam (LLM atlanır).
Prod'da mock YOK. LLM yalnız yorumlar; trade kararı Decision Layer'da.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from . import collectors, dedup, quality, repository
from . import source_registry as reg
from .config import IntelligenceConfig as C

logger = logging.getLogger("intelligence.engine")

try:
    from api_cache import TTLCache
    _snap_cache = TTLCache(ttl_seconds=C.SNAPSHOT_TTL, max_items=512)
except Exception:  # pragma: no cover
    _snap_cache = None


def build_snapshot(symbol: str, persist: bool = True) -> Optional[Dict]:
    if not C.ENABLED:
        return None
    sym = (symbol or "").upper()
    if not sym:
        return None

    if _snap_cache is not None:
        hit = _snap_cache.get(sym)
        if hit is not None:
            return hit

    warnings: List[str] = []

    # 1) Topla
    try:
        raw_items = collectors.collect_for_symbol(sym)
    except Exception as e:
        logger.info("collect hata (%s): %s", sym, e)
        raw_items = []
    if not raw_items:
        warnings.append("no_news_collected")
        return None  # scoring bunu "veri yok" olarak işler (missing_intelligence riski)

    # 2) Dedup + grup
    unique_items, stats = dedup.annotate_and_group(raw_items)
    reps = dedup.representative_items(unique_items)

    # 3) Baseline sentiment (VADER, hafif) — raw_sentiment
    for it in unique_items:
        txt = f"{it.get('title', '')} {it.get('summary', '')}"
        it["raw_sentiment"] = round(quality.vader_compound(txt), 3)
        it["symbol"] = sym

    # 4) Ollama — yalnız deduped top-N (reputation × freshness)
    ollama_used = False
    llm = None
    if C.OLLAMA_ENABLED:
        try:
            from .ollama_client import get_client
            llm = get_client()
            if not llm.available():
                warnings.append("ollama_unavailable")
                llm = None
        except Exception:
            warnings.append("ollama_unavailable")
            llm = None

    def _priority(it):
        rep = reg.reputation(it.get("source", ""))
        fr = quality.freshness_score(it.get("published_at"))
        return rep * 0.6 + fr * 0.4

    reps_sorted = sorted(reps, key=_priority, reverse=True)
    llm_targets = set(id(x) for x in reps_sorted[: C.MAX_LLM_ITEMS])

    # 5) Skorla
    scored: List[Dict] = []
    analysis_rows: List[Dict] = []
    hash_to_item = {it.get("content_hash"): it for it in unique_items}
    for it in reps_sorted:
        src = reg.get(it.get("source", ""))
        rep = float(src.reputation) if src else 50.0
        iw = float(src.impact_weight) if src else 0.4
        ollama = None
        if llm is not None and id(it) in llm_targets:
            ollama = llm.analyze_news(it, sym)
            if ollama is not None:
                ollama_used = True
        sc = quality.item_scores(it, rep, iw, ollama, it.get("category", "crypto_news"))
        entry = {**sc, "reputation": rep, "source": it.get("source"),
                 "title": it.get("title"), "url": it.get("url"), "ollama": ollama}
        scored.append(entry)
        analysis_rows.append({
            "content_hash": it.get("content_hash"), "symbol": sym,
            "sentiment": sc["sentiment"], "sentiment_score": sc["sentiment_score"],
            "quality_score": sc["quality_score"], "relevance_score": sc["relevance_score"],
            "impact_score": sc["impact_score"], "freshness_score": sc["freshness_score"],
            "hype_score": sc["hype_score"], "risk_score": sc["risk_score"],
            "event_type": sc["event_type"],
            "ollama_model": (C.OLLAMA_MODEL if ollama else None),
            "ollama_confidence": (ollama or {}).get("confidence"),
            "analysis_json": ollama,
        })

    # 6) Topla
    agg = quality.aggregate(scored, stats)
    snapshot = {
        "symbol": sym,
        "news_quality_score": agg["news_quality_score"],
        "social_momentum_score": None,  # Faz 5
        "hype_risk": agg["hype_risk"],
        "news_risk": agg["news_risk"],
        "news_sentiment": agg["news_sentiment"],
        "news_sentiment_score": agg.get("news_sentiment_score"),
        "news_count": stats["total"],
        "independent_news_count": stats["independent"],
        "duplicate_count": stats["duplicates"],
        "event_summary": agg["event_summary"],
        "top_news": agg["top_news"],
        "source_breakdown": agg["source_breakdown"],
        "ollama_used": ollama_used,
        "warnings": warnings,
    }

    # 6b) Sosyal medya (Faz 5) — opsiyonel; social_momentum + hype/bot harmanı
    if C.SOCIAL_ENABLED:
        try:
            from . import social_media
            social = social_media.build_social_snapshot(sym, llm=llm)
            if social:
                snapshot["social_momentum_score"] = social.get("social_momentum_score")
                snapshot["social"] = social
                sh = social.get("hype_score")
                if sh is not None:
                    base_h = snapshot.get("hype_risk") or 0.0
                    snapshot["hype_risk"] = round(max(float(base_h), float(sh)), 2)
                if (social.get("bot_risk_score") or 0) >= 60:
                    warnings.append("high_bot_risk")
                if social.get("is_manipulative"):
                    warnings.append("social_manipulation_suspected")
        except Exception as e:
            logger.info("social snapshot atlandı (%s): %s", sym, e)

    # 7) Persist (best-effort)
    if persist:
        try:
            hash_to_id = repository.save_news_items(unique_items)
            for row in analysis_rows:
                row["news_item_id"] = hash_to_id.get(row.pop("content_hash", None))
            repository.save_analysis(analysis_rows)
            repository.upsert_snapshot(sym, snapshot)
        except Exception as e:
            logger.info("persist atlandı (%s): %s", sym, e)

    if _snap_cache is not None:
        _snap_cache.set(sym, snapshot)
    return snapshot


def get_status() -> Dict:
    """Dashboard/API için katman durumu (Ollama health + kaynak sayıları)."""
    ollama = {"enabled": C.OLLAMA_ENABLED, "ok": False}
    if C.OLLAMA_ENABLED:
        try:
            from .ollama_client import get_client
            ollama = get_client().health()
        except Exception as e:
            ollama = {"enabled": True, "ok": False, "error": str(e)}
    sources = reg.all_sources()
    by_tier: Dict[int, int] = {}
    for s in sources:
        by_tier[s.tier] = by_tier.get(s.tier, 0) + 1
    return {
        "enabled": C.ENABLED,
        "social_enabled": C.SOCIAL_ENABLED,
        "ollama": ollama,
        "model": C.OLLAMA_MODEL,
        "sources_total": len(sources),
        "sources_collectable": len(reg.collectable_sources(include_social=False)),
        "sources_by_tier": by_tier,
        "news_lookback_hours": C.NEWS_LOOKBACK_HOURS,
        "max_llm_items": C.MAX_LLM_ITEMS,
    }
