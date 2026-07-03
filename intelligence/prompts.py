"""Ollama prompt şablonları — YALNIZ geçerli JSON döndürmesi istenir (markdown/açıklama YOK).

Not: JSON zorlaması `ollama_client` tarafında `format=json` ile de pekiştirilir; ayrıca yanıt
parse edilmeden önce `<think>...</think>` ve ```json çitleri temizlenir.
"""
from __future__ import annotations

from typing import Dict, List


def news_prompt(item: Dict, symbol: str) -> str:
    """Bir haber öğesi için trading-alaka analizi prompt'u (brief §7.2)."""
    title = (item.get("title") or "").replace("\n", " ").strip()
    source = item.get("source") or "unknown"
    published = item.get("published_at") or ""
    content = (item.get("content") or item.get("summary") or "").replace("\n", " ").strip()[:1800]
    return f"""You are a crypto market news analyst.
Analyze the following news item for trading relevance.
Return ONLY valid JSON. Do not include markdown. Do not include any explanation outside JSON.

News:
Title: {title}
Source: {source}
Published At: {published}
Content: {content}
Target Symbol: {symbol}

Return exactly this JSON shape:
{{
  "summary": "one concise sentence",
  "mentioned_assets": [],
  "is_relevant_to_symbol": true,
  "relevance_score": 0,
  "sentiment": "positive|neutral|negative",
  "sentiment_score": 0,
  "impact_level": "low|medium|high|critical",
  "impact_score": 0,
  "time_horizon": "short_term|medium_term|long_term",
  "is_hype": false,
  "hype_score": 0,
  "is_duplicate_or_rewrite": false,
  "risk_flags": [],
  "event_type": "listing|delisting|regulatory|hack|partnership|token_unlock|upgrade|general|unknown",
  "trade_interpretation": "one concise sentence",
  "confidence": 0.0
}}
Scores relevance_score/sentiment_score/impact_score/hype_score are integers 0-100. confidence is 0.0-1.0."""


def social_prompt(symbol: str, posts: List[str]) -> str:
    """Sosyal medya gönderileri için risk/hype/bot analizi prompt'u (brief §7.3, Faz 5)."""
    joined = "\n".join(f"- {p.strip()}" for p in posts[:40] if p and p.strip())
    return f"""You are a crypto social media risk analyst.
Analyze the following social media posts for market relevance, hype, bot risk and trading usefulness.
Return ONLY valid JSON. Do not include markdown.

Target Symbol: {symbol}
Posts:
{joined}

Return exactly this JSON shape:
{{
  "summary": "one concise sentence",
  "mention_quality": "low|medium|high",
  "social_sentiment": "positive|neutral|negative",
  "social_sentiment_score": 0,
  "hype_score": 0,
  "bot_risk_score": 0,
  "manipulation_risk_score": 0,
  "community_quality_score": 0,
  "influencer_impact_score": 0,
  "is_manipulative": false,
  "manipulation_type": "none|pump_hype|coordinated_spam|fake_news|unknown",
  "risk_flags": [],
  "trade_interpretation": "one concise sentence",
  "confidence": 0.0
}}
All *_score fields are integers 0-100. confidence is 0.0-1.0."""
