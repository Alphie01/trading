"""Market Intelligence veri erişimi — SHARED şema (evren-geneli; get_session).

automation/repository.py deseni: tenant bağlamı gerekmez; hatalar yutulur (best-effort),
market zekâsı tüm tenant'lara ortaktır. Tenant session'ları da shared tabloları okuyabilir
(search_path'te 'shared' var).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

logger = logging.getLogger("intelligence.repository")


def _f(v):
    return float(v) if isinstance(v, Decimal) else v


def sync_sources() -> None:
    """Kod registry'sini intelligence_sources tablosuna yansıt (upsert by name)."""
    try:
        from sqlalchemy import select
        from trading_db.models_shared import IntelligenceSource
        from trading_db.session import get_session
        from . import source_registry as reg
        with get_session() as s:
            existing = {r.name: r for r in s.execute(select(IntelligenceSource)).scalars().all()}
            for src in reg.all_sources():
                row = existing.get(src.name)
                if row is None:
                    row = IntelligenceSource(name=src.name)
                    s.add(row)
                row.category = src.category
                row.tier = src.tier
                row.reputation_score = src.reputation
                row.impact_weight = src.impact_weight
                row.allow_trade_signal_boost = src.allow_trade_signal_boost
                row.regional = src.regional
                row.enabled = src.enabled and src.collector is not None
    except Exception as e:
        logger.info("sync_sources atlandı: %s", e)


def save_news_items(items: List[Dict]) -> Dict[str, int]:
    """news_items upsert (content_hash unique). Dönüş: {content_hash: id} (mevcut + yeni)."""
    out: Dict[str, int] = {}
    if not items:
        return out
    try:
        from sqlalchemy import select
        from trading_db.models_shared import NewsItem
        from trading_db.session import get_session
        hashes = [it.get("content_hash") for it in items if it.get("content_hash")]
        with get_session() as s:
            existing = {
                r.content_hash: r.id
                for r in s.execute(select(NewsItem).where(NewsItem.content_hash.in_(hashes))).scalars().all()
            }
            out.update(existing)
            for it in items:
                ch = it.get("content_hash")
                if not ch or ch in out:
                    continue
                row = NewsItem(
                    source=it.get("source"), source_tier=it.get("source_tier"),
                    symbol=it.get("symbol"), title=(it.get("title") or "")[:8000],
                    url=it.get("url"), published_at=it.get("published_at"),
                    content_hash=ch, duplicate_group_id=it.get("duplicate_group_id"),
                    raw_sentiment=it.get("raw_sentiment"), data_source=it.get("data_source"),
                )
                s.add(row)
                s.flush()
                out[ch] = row.id
    except Exception as e:
        logger.info("save_news_items atlandı: %s", e)
    return out


def save_analysis(rows: List[Dict]) -> None:
    """news_analysis_results toplu insert."""
    if not rows:
        return
    try:
        from trading_db.models_shared import NewsAnalysisResult
        from trading_db.session import get_session
        with get_session() as s:
            for r in rows:
                s.add(NewsAnalysisResult(
                    news_item_id=r.get("news_item_id"), symbol=r.get("symbol"),
                    sentiment=r.get("sentiment"), sentiment_score=r.get("sentiment_score"),
                    quality_score=r.get("quality_score"), relevance_score=r.get("relevance_score"),
                    impact_score=r.get("impact_score"), freshness_score=r.get("freshness_score"),
                    hype_score=r.get("hype_score"), risk_score=r.get("risk_score"),
                    event_type=r.get("event_type"), ollama_model=r.get("ollama_model"),
                    ollama_confidence=r.get("ollama_confidence"), analysis_json=r.get("analysis_json"),
                ))
    except Exception as e:
        logger.info("save_analysis atlandı: %s", e)


def upsert_snapshot(symbol: str, snap: Dict) -> None:
    """market_intelligence_snapshots'a tarihsel kayıt (her hesaplama yeni satır)."""
    try:
        from trading_db.models_shared import MarketIntelligenceSnapshot
        from trading_db.session import get_session
        with get_session() as s:
            s.add(MarketIntelligenceSnapshot(
                symbol=(symbol or "").upper(),
                news_quality_score=snap.get("news_quality_score"),
                social_momentum_score=snap.get("social_momentum_score"),
                hype_risk=snap.get("hype_risk"),
                news_sentiment=snap.get("news_sentiment"),
                news_count=snap.get("news_count", 0),
                independent_news_count=snap.get("independent_news_count", 0),
                duplicate_count=snap.get("duplicate_count", 0),
                event_summary=snap.get("event_summary"),
                top_news=snap.get("top_news"),
                source_breakdown=snap.get("source_breakdown"),
                ollama_used=snap.get("ollama_used", False),
                warnings=snap.get("warnings"),
            ))
    except Exception as e:
        logger.info("upsert_snapshot atlandı: %s", e)


def get_snapshot(symbol: str) -> Optional[Dict]:
    """Bir sembol için en güncel snapshot."""
    try:
        from sqlalchemy import select
        from trading_db.models_shared import MarketIntelligenceSnapshot as M
        from trading_db.session import get_session
        with get_session() as s:
            row = s.execute(
                select(M).where(M.symbol == (symbol or "").upper()).order_by(M.computed_at.desc()).limit(1)
            ).scalar_one_or_none()
            if not row:
                return None
            return {
                "symbol": row.symbol, "news_quality_score": _f(row.news_quality_score),
                "social_momentum_score": _f(row.social_momentum_score), "hype_risk": _f(row.hype_risk),
                "news_sentiment": row.news_sentiment, "news_count": row.news_count,
                "independent_news_count": row.independent_news_count, "duplicate_count": row.duplicate_count,
                "event_summary": row.event_summary, "top_news": row.top_news,
                "source_breakdown": row.source_breakdown, "ollama_used": row.ollama_used,
                "warnings": row.warnings,
                "computed_at": row.computed_at.isoformat() if row.computed_at else None,
            }
    except Exception as e:
        logger.info("get_snapshot atlandı: %s", e)
        return None


def get_latest_news(symbol: str = None, limit: int = 30) -> List[Dict]:
    try:
        from sqlalchemy import select
        from trading_db.models_shared import NewsItem
        from trading_db.session import get_session
        with get_session() as s:
            q = select(NewsItem).order_by(NewsItem.published_at.desc().nullslast()).limit(limit)
            if symbol:
                q = select(NewsItem).where(NewsItem.symbol == symbol.upper()).order_by(
                    NewsItem.published_at.desc().nullslast()).limit(limit)
            rows = s.execute(q).scalars().all()
            return [{
                "source": r.source, "source_tier": r.source_tier, "symbol": r.symbol,
                "title": r.title, "url": r.url, "data_source": r.data_source,
                "raw_sentiment": _f(r.raw_sentiment), "duplicate_group_id": r.duplicate_group_id,
                "published_at": r.published_at.isoformat() if r.published_at else None,
            } for r in rows]
    except Exception as e:
        logger.info("get_latest_news atlandı: %s", e)
        return []


def get_sources() -> List[Dict]:
    """Kaynak registry'si (dashboard source health). DB gerekmez (kod = doğruluk kaynağı)."""
    from . import source_registry as reg
    return [{
        "name": s.name, "category": s.category, "tier": s.tier,
        "reputation": float(s.reputation), "impact_weight": float(s.impact_weight),
        "collector": s.collector, "regional": s.regional,
        "active": s.collector is not None and s.enabled,
    } for s in reg.all_sources()]


def get_recent_snapshots(limit: int = 50) -> List[Dict]:
    """Sembol başına en güncel snapshot (Market Intelligence genel bakış)."""
    try:
        from sqlalchemy import select
        from trading_db.models_shared import MarketIntelligenceSnapshot as M
        from trading_db.session import get_session
        with get_session() as s:
            rows = s.execute(select(M).order_by(M.symbol, M.computed_at.desc())).scalars().all()
            latest = {}
            for r in rows:
                if r.symbol not in latest:
                    latest[r.symbol] = r
            items = [{
                "symbol": r.symbol, "news_quality_score": _f(r.news_quality_score),
                "social_momentum_score": _f(r.social_momentum_score), "hype_risk": _f(r.hype_risk),
                "news_sentiment": r.news_sentiment, "news_count": r.news_count,
                "independent_news_count": r.independent_news_count, "ollama_used": r.ollama_used,
                "event_summary": r.event_summary,
                "computed_at": r.computed_at.isoformat() if r.computed_at else None,
            } for r in latest.values()]
            items.sort(key=lambda x: (x["computed_at"] or ""), reverse=True)
            return items[:limit]
    except Exception as e:
        logger.info("get_recent_snapshots atlandı: %s", e)
        return []
