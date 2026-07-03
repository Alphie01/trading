"""Skorlama — opportunity_score / risk_score / confidence.

Faz 4 (bu aşama): YALNIZ ticker verisinden ÖN skor (hacim/likidite/volatilite/momentum).
Derin teknik analiz (RSI/MACD/Yigit) + AI (LSTM/DQN) + sentiment + whale skorları Faz 5'te
`data_preprocessor.add_technical_indicators` ve optimize predictor ile eklenecek.
Bu yüzden Faz 4 confidence'ı bilinçli olarak DÜŞÜK tutulur (AI henüz katkı vermiyor).

Saf fonksiyonlar (ağ/DB yok) → test edilebilir.
"""
from __future__ import annotations

import math
from typing import Dict, List

from .config import AutomationConfig as C


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _volume_liquidity_score(quote_volume: float) -> float:
    """Log-ölçekli hacim skoru. ~1M→~40, ~10M→~60, ~100M→~80, ~1B→~100."""
    if quote_volume <= 0:
        return 0.0
    return _clamp((math.log10(quote_volume) - 5.0) * 20.0)  # 10^5=0, 10^10=100


def _volatility(ticker: Dict) -> float:
    high, low = ticker.get("high"), ticker.get("low")
    if high and low and float(low) > 0:
        return (float(high) - float(low)) / float(low) * 100.0
    return 0.0


def score_ticker(ticker: Dict, warnings: List[str] = None) -> Dict:
    """Ön (Faz 4) skor üretir. Faz 5 bu alanları teknik/AI ile zenginleştirir."""
    warnings = list(warnings or [])
    reasons: List[str] = []

    quote_volume = float(ticker.get("quote_volume") or 0)
    pct = ticker.get("percentage")
    pct = float(pct) if pct is not None else 0.0

    vol_liq = _volume_liquidity_score(quote_volume)
    if vol_liq >= 60:
        reasons.append("healthy_24h_volume")

    volatility = _volatility(ticker)
    # Momentum skoru: 24h % değişimi 0-100'e eşle (+%10 → 100, -%10 → 0)
    momentum_score = _clamp(50.0 + pct * 5.0)
    if pct > 3:
        reasons.append(f"positive_24h_momentum:{pct:+.1f}%")
    elif pct < -3:
        reasons.append(f"negative_24h_momentum:{pct:+.1f}%")

    # --- Opportunity (Faz 4 ÖN skor: hacim/likidite %40 + momentum %60) ---
    # AI/teknik/sentiment/whale bileşenleri Faz 5'te eklenecek; şimdilik skora katılmıyor.
    opportunity = _clamp(0.4 * vol_liq + 0.6 * momentum_score)

    # --- Risk ---
    risk = 0.0
    if volatility > 30:
        risk += 30
        warnings.append(f"high_volatility_24h:{volatility:.1f}%")
    elif volatility > 15:
        risk += 15
    if quote_volume < C.MIN_24H_VOLUME_USDT * 2:
        risk += 15
        warnings.append("thin_volume_risk")
    if "elevated_spread" in " ".join(warnings) or any("spread" in w for w in warnings):
        risk += 10
    if abs(pct) > 25:
        risk += 25
        warnings.append("pump_dump_risk")
    # Faz 4: AI confidence yok → model_uncertainty_risk baz eklenir
    risk += 15  # model_uncertainty (AI henüz devrede değil)
    risk = _clamp(risk)

    # --- Confidence (Faz 4 düşük; AI yok) ---
    confidence = 0.35 if opportunity >= C.WATCHLIST_MIN_OPPORTUNITY else 0.25

    # --- Recommendation (ön; Faz 5'te AI ensemble ile netleşir) ---
    if opportunity >= C.HOT_MIN_OPPORTUNITY and risk <= C.MAX_RISK_SCORE:
        recommendation = "BUY"
    elif opportunity >= C.WATCHLIST_MIN_OPPORTUNITY and risk <= C.MAX_RISK_SCORE:
        recommendation = "HOLD"
    else:
        recommendation = "HOLD"

    warnings.append("ai_scoring_pending_phase5")

    return {
        "symbol": ticker.get("symbol"),
        "opportunity_score": round(opportunity, 2),
        "risk_score": round(risk, 2),
        "confidence": round(confidence, 4),
        "technical_score": None,          # Faz 5
        "volume_liquidity_score": round(vol_liq, 2),
        "ai_prediction_score": None,      # Faz 5
        "sentiment_score": None,          # Faz 5
        "whale_score": None,              # Faz 5
        "recommendation": recommendation,
        "reasons": reasons,
        "warnings": sorted(set(warnings)),
    }


def score_full(research: Dict) -> Dict:
    """Faz 5 tam skor: research bileşenlerini (hacim+teknik+AI+sentiment+whale) birleştirir.

    Eksik bileşenler için ağırlıklar YENİDEN NORMALİZE edilir (skoru yapay şişirmez/düşürmez);
    ayrıca eksik veri riski (missing_data_risk / model_uncertainty) eklenir.
    """
    symbol = research.get("symbol")
    ticker = research.get("ticker") or {}
    tech = research.get("technical") or {}
    ai = research.get("ai")
    sent = research.get("sentiment")
    whale = research.get("whale")

    reasons: List[str] = list(tech.get("reasons", []) if isinstance(tech, dict) else [])
    warnings: List[str] = list(tech.get("warnings", []) if isinstance(tech, dict) else [])

    # Bileşen skorları (mevcut olanlar)
    vol_liq = _volume_liquidity_score(float(ticker.get("quote_volume") or 0)) if ticker else None
    technical = tech.get("technical_score") if isinstance(tech, dict) else None
    ai_score = ai.get("ai_prediction_score") if ai else None
    sent_score = sent.get("sentiment_score") if sent else None
    whale_score = whale.get("whale_score") if whale else None

    components = [
        (C.W_VOLUME_LIQUIDITY, vol_liq),
        (C.W_TECHNICAL, technical),
        (C.W_AI_PREDICTION, ai_score),
        (C.W_SENTIMENT, sent_score),
        (C.W_WHALE, whale_score),
    ]
    present = [(w, v) for (w, v) in components if v is not None]
    total_w = sum(w for w, _ in present)
    if total_w > 0:
        opportunity = _clamp(sum(w * v for w, v in present) / total_w)
    else:
        opportunity = 0.0

    # Risk: teknik risk + eksik veri/model belirsizliği
    risk = float(tech.get("technical_risk", 0) if isinstance(tech, dict) else 0)
    missing = 0
    if ai_score is None:
        risk += 15; missing += 1; warnings.append("ai_unavailable")
    if sent_score is None:
        risk += 5; missing += 1; warnings.append("sentiment_unavailable")
    if whale_score is None:
        risk += 5; missing += 1; warnings.append("whale_unavailable")
    if vol_liq is not None and vol_liq < 40:
        risk += 15; warnings.append("low_liquidity_risk")
    risk = _clamp(risk)

    # Confidence: bileşen kapsamı + AI confidence
    coverage = len(present) / 5.0
    ai_conf = float(ai.get("ai_confidence", 0)) if ai else 0.0
    confidence = round(_clamp(0.2 + 0.5 * coverage + 0.3 * ai_conf, 0.0, 1.0), 4)

    # Recommendation
    if opportunity >= C.HOT_MIN_OPPORTUNITY and risk <= C.MAX_RISK_SCORE:
        recommendation = "BUY"
    elif opportunity <= 30 and risk >= C.MAX_RISK_SCORE:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    if ai and ai.get("predicted_change_percent", 0) is not None:
        chg = ai.get("predicted_change_percent", 0)
        if chg is not None and chg > 3 and recommendation == "HOLD" and risk <= C.MAX_RISK_SCORE:
            reasons.append(f"ai_predicts_upside:{chg:+.1f}%")

    return {
        "symbol": symbol,
        "opportunity_score": round(opportunity, 2),
        "risk_score": round(risk, 2),
        "confidence": confidence,
        "technical_score": technical,
        "volume_liquidity_score": round(vol_liq, 2) if vol_liq is not None else None,
        "ai_prediction_score": ai_score,
        "sentiment_score": sent_score,
        "whale_score": whale_score,
        "recommendation": recommendation,
        "reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings)),
    }
