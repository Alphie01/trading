"""Otomasyon veri erişimi — SHARED şema (evren-geneli market zekâsı).

trading_db paketini kullanır (tek kaynak). get_session() shared şemaya yazar;
tenant bağlamı GEREKMEZ (discovery/scoring tüm tenant'lara ortaktır).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import select

from trading_db.models_shared import AutomationRun, CoinScore, DiscoveryCandidate
from trading_db.session import get_session


def start_run(run_type: str) -> Optional[int]:
    try:
        with get_session() as s:
            run = AutomationRun(run_type=run_type, status="running")
            s.add(run)
            s.flush()
            return run.id
    except Exception as e:
        print(f"❌ automation start_run hatası: {e}")
        return None


def finish_run(run_id: Optional[int], status: str, scanned: int, passed: int,
               watchlisted: int, details: Dict = None) -> None:
    if run_id is None:
        return
    try:
        with get_session() as s:
            run = s.get(AutomationRun, run_id)
            if run:
                run.status = status
                run.scanned_count = scanned
                run.passed_count = passed
                run.watchlisted_count = watchlisted
                run.finished_at = datetime.now(timezone.utc)
                run.details = details or {}
    except Exception as e:
        print(f"❌ automation finish_run hatası: {e}")


def upsert_candidate(ticker: Dict, status: str, reason: str, scores: Dict = None) -> None:
    """discovery_candidates upsert (symbol unique)."""
    try:
        sym = (ticker.get("symbol") or "").upper()
        if not sym:
            return
        scores = scores or {}
        with get_session() as s:
            c = s.execute(
                select(DiscoveryCandidate).where(DiscoveryCandidate.symbol == sym)
            ).scalar_one_or_none()
            if c is None:
                c = DiscoveryCandidate(symbol=sym)
                s.add(c)
            c.base_asset = ticker.get("base")
            c.quote_asset = ticker.get("quote")
            c.volume_24h = ticker.get("quote_volume")
            c.price_change_24h = ticker.get("percentage")
            c.volatility_score = scores.get("volatility_score")
            c.liquidity_score = scores.get("volume_liquidity_score")
            c.discovery_reason = reason
            # Yalnız ileri statülere geçir; mevcut watchlist/hot'u geri düşürme
            if c.status in (None, "discovered", "rejected", "cooldown") or status in (
                "watchlist", "hot_candidate", "trade_candidate"
            ):
                c.status = status
    except Exception as e:
        print(f"❌ upsert_candidate hatası: {e}")


def save_score(score: Dict) -> None:
    """coin_scores tarihsel kayıt (her tarama yeni satır)."""
    try:
        with get_session() as s:
            s.add(CoinScore(
                symbol=(score.get("symbol") or "").upper(),
                opportunity_score=score.get("opportunity_score"),
                risk_score=score.get("risk_score"),
                confidence=score.get("confidence"),
                technical_score=score.get("technical_score"),
                volume_liquidity_score=score.get("volume_liquidity_score"),
                ai_prediction_score=score.get("ai_prediction_score"),
                sentiment_score=score.get("sentiment_score"),
                whale_score=score.get("whale_score"),
                recommendation=score.get("recommendation"),
                reasons=score.get("reasons"),
                warnings=score.get("warnings"),
            ))
    except Exception as e:
        print(f"❌ save_score hatası: {e}")


def get_candidates(status: str = None, limit: int = 100) -> List[Dict]:
    try:
        with get_session() as s:
            stmt = select(DiscoveryCandidate)
            if status:
                stmt = stmt.where(DiscoveryCandidate.status == status)
            stmt = stmt.order_by(DiscoveryCandidate.updated_at.desc()).limit(limit)
            rows = s.execute(stmt).scalars().all()
            return [{
                "symbol": r.symbol,
                "status": r.status,
                "volume_24h": float(r.volume_24h) if r.volume_24h is not None else None,
                "price_change_24h": float(r.price_change_24h) if r.price_change_24h is not None else None,
                "discovery_reason": r.discovery_reason,
                "discovered_at": r.discovered_at.isoformat() if r.discovered_at else None,
            } for r in rows]
    except Exception as e:
        print(f"❌ get_candidates hatası: {e}")
        return []


def get_coin_detail(symbol: str) -> Dict:
    """Bir coin'in aday kaydı + en son skoru."""
    sym = (symbol or "").upper()
    out = {"symbol": sym, "candidate": None, "score": None}
    try:
        with get_session() as s:
            c = s.execute(
                select(DiscoveryCandidate).where(DiscoveryCandidate.symbol == sym)
            ).scalar_one_or_none()
            if c:
                out["candidate"] = {
                    "symbol": c.symbol, "status": c.status, "base_asset": c.base_asset,
                    "quote_asset": c.quote_asset, "exchange": c.exchange,
                    "volume_24h": float(c.volume_24h) if c.volume_24h is not None else None,
                    "price_change_24h": float(c.price_change_24h) if c.price_change_24h is not None else None,
                    "volatility_score": float(c.volatility_score) if c.volatility_score is not None else None,
                    "liquidity_score": float(c.liquidity_score) if c.liquidity_score is not None else None,
                    "discovery_reason": c.discovery_reason,
                    "discovered_at": c.discovered_at.isoformat() if c.discovered_at else None,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                }
            sc = s.execute(
                select(CoinScore).where(CoinScore.symbol == sym)
                .order_by(CoinScore.scored_at.desc()).limit(1)
            ).scalar_one_or_none()
            if sc:
                out["score"] = {
                    "opportunity_score": float(sc.opportunity_score) if sc.opportunity_score is not None else None,
                    "risk_score": float(sc.risk_score) if sc.risk_score is not None else None,
                    "confidence": float(sc.confidence) if sc.confidence is not None else None,
                    "technical_score": float(sc.technical_score) if sc.technical_score is not None else None,
                    "volume_liquidity_score": float(sc.volume_liquidity_score) if sc.volume_liquidity_score is not None else None,
                    "ai_prediction_score": float(sc.ai_prediction_score) if sc.ai_prediction_score is not None else None,
                    "sentiment_score": float(sc.sentiment_score) if sc.sentiment_score is not None else None,
                    "whale_score": float(sc.whale_score) if sc.whale_score is not None else None,
                    "recommendation": sc.recommendation,
                    "reasons": sc.reasons or [],
                    "warnings": sc.warnings or [],
                    "scored_at": sc.scored_at.isoformat() if sc.scored_at else None,
                }
    except Exception as e:
        print(f"❌ get_coin_detail hatası: {e}")
    return out


def get_score_history(symbol: str, limit: int = 40) -> List[Dict]:
    """Bir coin'in tarihsel skor serisi (grafik için, eskiden yeniye)."""
    sym = (symbol or "").upper()
    try:
        with get_session() as s:
            rows = s.execute(
                select(CoinScore.opportunity_score, CoinScore.risk_score,
                       CoinScore.confidence, CoinScore.scored_at)
                .where(CoinScore.symbol == sym)
                .order_by(CoinScore.scored_at.desc()).limit(limit)
            ).all()
            rows = list(reversed(rows))
            return [{
                "opportunity_score": float(r[0]) if r[0] is not None else None,
                "risk_score": float(r[1]) if r[1] is not None else None,
                "confidence": float(r[2]) if r[2] is not None else None,
                "scored_at": r[3].isoformat() if r[3] else None,
            } for r in rows]
    except Exception as e:
        print(f"❌ get_score_history hatası: {e}")
        return []


def get_latest_scores(limit: int = 50) -> List[Dict]:
    try:
        with get_session() as s:
            rows = s.execute(
                select(CoinScore).order_by(CoinScore.scored_at.desc()).limit(limit)
            ).scalars().all()
            return [{
                "symbol": r.symbol,
                "opportunity_score": float(r.opportunity_score) if r.opportunity_score is not None else None,
                "risk_score": float(r.risk_score) if r.risk_score is not None else None,
                "confidence": float(r.confidence) if r.confidence is not None else None,
                "recommendation": r.recommendation,
                "reasons": r.reasons,
                "warnings": r.warnings,
                "scored_at": r.scored_at.isoformat() if r.scored_at else None,
            } for r in rows]
    except Exception as e:
        print(f"❌ get_latest_scores hatası: {e}")
        return []
