"""Sosyal medya momentum/hype/bot analizi (Faz 5) — Reddit + Google Trends.

Amaç: mention sayısı değil; hype mı gerçek momentum mu, ve bot/manipülasyon riski.
- Reddit: mention hacmi + upvote-ağırlıklı topluluk kalitesi + sentiment + tekrar (koordineli spam).
- Google Trends (pytrends): arama ilgisi hız (trend velocity).
- Opsiyonel Ollama sosyal analizi (hype/bot/manipülasyon) — varsa kuralları ezer.

Prod'da mock YOK. Kaynak/dep yoksa graceful (None). Sosyal skor trade kararı VERMEZ (destek sinyali).
"""
from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Optional

from .config import IntelligenceConfig as C

logger = logging.getLogger("intelligence.social")


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def google_trend_velocity(symbol: str) -> Optional[float]:
    """Google Trends arama ilgisi hızı (0-100; 50=sabit, >50 artıyor). pytrends yoksa None."""
    if not C.GOOGLE_TRENDS_ENABLED:
        return None
    try:
        from pytrends.request import TrendReq
        from .collectors import _base_of, _NAME
    except Exception:
        return None
    base = _base_of(symbol)
    kw = (_NAME.get(base) or base) + " crypto"
    try:
        py = TrendReq(hl="en-US", tz=0, timeout=(6, 10))
        py.build_payload([kw], timeframe="now 7-d")
        df = py.interest_over_time()
        if df is None or df.empty or kw not in df:
            return None
        vals = [float(x) for x in df[kw].values]
        if len(vals) < 6:
            return None
        k = max(1, len(vals) // 3)
        recent = sum(vals[-k:]) / k
        prior = sum(vals[:k]) / k
        if prior <= 0:
            return 60.0 if recent > 0 else 50.0
        return _clamp(50.0 * (recent / prior))  # ratio 1→50, 2→100, 0.5→25
    except Exception as e:
        logger.info("google_trends atlandı (%s): %s", symbol, e)
        return None


def _rule_signals(posts: List[Dict], trend: Optional[float]) -> Optional[Dict]:
    """Reddit gönderilerinden kural-tabanlı sosyal sinyaller."""
    texts = [(p.get("title") or "") for p in posts]
    n = len(texts)
    if n == 0:
        return None
    from .quality import vader_compound, _HYPE_WORDS
    from .dedup import normalize_title

    sents = [vader_compound(t) for t in texts]
    sent_score = _clamp(50.0 + (sum(sents) / n) * 50.0)

    hype_hits = sum(len(_HYPE_WORDS.findall(t)) for t in texts)
    hype = _clamp((hype_hits / n) * 60.0)

    norm = [normalize_title(t) for t in texts]
    uniq = len(set(norm))
    rep_ratio = 1.0 - (uniq / n)  # 0 hepsi farklı, →1 hepsi aynı (koordineli spam)

    scores = [float(p.get("score", 0) or 0) for p in posts]
    avg_score = (sum(scores) / n) if scores else 0.0
    community = _clamp(min(100.0, avg_score / 5.0))  # ~500 upvote → 100

    bot = _clamp(rep_ratio * 50.0 + hype * 0.3 + max(0.0, 40.0 - community) * 0.5)

    mention_vol = _clamp(math.log10(max(1, n)) * 33.0)  # 1→0, 10→33, 100→66
    momentum = _clamp(0.4 * mention_vol + 0.35 * sent_score + 0.25 * (trend if trend is not None else 50.0))
    momentum *= (1.0 - bot / 200.0)  # bot cezası (max -%50)

    return {
        "mention_count": n,
        "social_sentiment_score": round(sent_score, 2),
        "hype_score": round(hype, 2),
        "bot_risk_score": round(bot, 2),
        "community_quality_score": round(community, 2),
        "repetition_ratio": round(rep_ratio, 3),
        "trend_velocity": round(trend, 2) if trend is not None else None,
        "social_momentum_score": round(_clamp(momentum), 2),
    }


def build_social_snapshot(symbol: str, llm=None) -> Optional[Dict]:
    """Sembol için sosyal snapshot. Kaynak yoksa None (graceful)."""
    if not C.SOCIAL_ENABLED:
        return None
    try:
        from . import collectors
        posts = collectors.collect_reddit(symbol, max_items=C.SOCIAL_MAX_POSTS, timeout=C.COLLECT_TIMEOUT)
    except Exception:
        posts = []
    trend = google_trend_velocity(symbol)
    if not posts and trend is None:
        return None

    sig = _rule_signals(posts, trend) or {
        "mention_count": 0, "social_sentiment_score": 50.0, "hype_score": 0.0,
        "bot_risk_score": 0.0, "community_quality_score": 50.0, "repetition_ratio": 0.0,
        "trend_velocity": trend, "social_momentum_score": _clamp((trend or 50.0)),
    }

    # Opsiyonel Ollama sosyal analizi (kuralları ezer)
    ollama = None
    if llm is not None and posts:
        try:
            from .prompts import social_prompt
            ollama = llm.generate_json(social_prompt(symbol, [p.get("title", "") for p in posts]))
        except Exception:
            ollama = None
    if ollama:
        for src_k, dst_k in [("hype_score", "hype_score"), ("bot_risk_score", "bot_risk_score"),
                             ("social_sentiment_score", "social_sentiment_score"),
                             ("community_quality_score", "community_quality_score")]:
            v = ollama.get(src_k)
            if v is not None:
                sig[dst_k] = _clamp(float(v))
        sig["manipulation_type"] = ollama.get("manipulation_type", "none")
        sig["is_manipulative"] = bool(ollama.get("is_manipulative", False))
        # momentum'u güncel sentiment/bot ile yeniden hesapla
        mv = _clamp(math.log10(max(1, sig["mention_count"])) * 33.0)
        mom = _clamp(0.4 * mv + 0.35 * sig["social_sentiment_score"] + 0.25 * (trend if trend is not None else 50.0))
        sig["social_momentum_score"] = round(_clamp(mom * (1.0 - sig["bot_risk_score"] / 200.0)), 2)

    sig["ollama_used"] = ollama is not None
    return sig
