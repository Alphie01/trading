"""Derin coin araştırması (Faz 5): teknik analiz + (opsiyonel) AI + sentiment + whale.

Tasarım:
- Teknik skor HER ZAMAN hesaplanır (RSI/MACD/EMA/BB/Yigit — data_preprocessor).
- AI (LSTM/DQN), sentiment, whale OPSİYONELDİR: TF/model/key yoksa GÜVENLE atlanır
  (None döner, skora katılmaz). Böylece TF olmayan ortamda da çalışır; production'da
  optimize predictor devredeyse AI skoru eklenir.
- Sessiz mock YOK: sentiment/whale api_cache.demo_mode/skip kurallarına uyar.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional


def _last_valid(series):
    try:
        s = series.dropna()
        return float(s.iloc[-1]) if len(s) else None
    except Exception:
        return None


def technical_snapshot(df) -> Dict:
    """OHLCV DataFrame'den teknik skor + göstergeler. df: open/high/low/close/volume."""
    reasons: List[str] = []
    warnings: List[str] = []
    try:
        from data_preprocessor import CryptoDataPreprocessor
        pp = CryptoDataPreprocessor()
        tdf = pp.add_technical_indicators(df.copy())
    except Exception as e:
        return {"technical_score": None, "error": f"technical_failed: {e}",
                "reasons": reasons, "warnings": ["technical_unavailable"]}

    last = tdf.iloc[-1]
    close = _last_valid(tdf["close"])
    rsi = _last_valid(tdf["rsi"])
    macd = _last_valid(tdf["macd"])
    macd_sig = _last_valid(tdf["macd_signal"])
    ema12 = _last_valid(tdf["ema_12"])
    sma25 = _last_valid(tdf["sma_25"])
    bb_upper = _last_valid(tdf["bb_upper"])
    bb_lower = _last_valid(tdf["bb_lower"])
    yigit_pos = _last_valid(tdf["yigit_position"]) if "yigit_position" in tdf else None
    yigit_strength = _last_valid(tdf["yigit_trend_strength"]) if "yigit_trend_strength" in tdf else None

    score = 50.0
    risk_add = 0.0

    # RSI
    if rsi is not None:
        if 45 <= rsi <= 62:
            score += 10; reasons.append(f"rsi_healthy:{rsi:.0f}")
        elif rsi < 30:
            score += 12; risk_add += 8; reasons.append(f"rsi_oversold:{rsi:.0f}")
        elif rsi > 72:
            score -= 10; risk_add += 12; warnings.append(f"rsi_overbought:{rsi:.0f}")

    # MACD
    if macd is not None and macd_sig is not None:
        if macd > macd_sig:
            score += 12; reasons.append("macd_bullish")
        else:
            score -= 8

    # Trend (EMA/SMA)
    if close is not None and ema12 is not None and sma25 is not None:
        if close > ema12 and close > sma25:
            score += 12; reasons.append("uptrend_above_ema_sma")
        elif close < ema12 and close < sma25:
            score -= 12; warnings.append("downtrend")

    # Bollinger
    if close is not None and bb_upper is not None and bb_lower is not None and bb_upper > bb_lower:
        if close >= bb_upper:
            risk_add += 8; warnings.append("above_bb_upper")
        elif close <= bb_lower:
            reasons.append("near_bb_lower_value")

    # Yigit ATR trailing stop
    if yigit_pos is not None:
        if yigit_pos > 0:
            score += 12; reasons.append("yigit_long")
        elif yigit_pos < 0:
            score -= 12; warnings.append("yigit_short")

    score = max(0.0, min(100.0, score))
    risk = max(0.0, min(100.0, risk_add))

    return {
        "technical_score": round(score, 2),
        "technical_risk": round(risk, 2),
        "rsi": rsi,
        "macd_bullish": (macd is not None and macd_sig is not None and macd > macd_sig),
        "trend_up": (close is not None and ema12 is not None and close > ema12),
        "yigit_position": yigit_pos,
        "yigit_trend_strength": yigit_strength,
        "reasons": reasons,
        "warnings": warnings,
    }


def _lstm_ai_snapshot(df, symbol: str) -> Optional[Dict]:
    """OPSİYONEL LSTM tabanlı AI tahmini (optimize predictor). TF/model yoksa None.

    AUTO_USE_AI=false ile kapatılabilir. Bellek-cache sayesinde model yeniden eğitilmez.
    Not: AVX'siz prod sunucuda TensorFlow SIGILL edebilir → burası None döner; tree modeli
    (varsa) yine ai skoru üretebilir (bkz. _tree_ai_snapshot).
    """
    if os.getenv("AUTO_USE_AI", "true").lower() not in ("1", "true", "yes"):
        return None
    try:
        from data_preprocessor import CryptoDataPreprocessor
        from predictor import CryptoPricePredictor
        pp = CryptoDataPreprocessor()
        processed = pp.prepare_data(df, use_technical_indicators=True)
        if processed is None or len(processed) < 100:
            return None
        predictor = CryptoPricePredictor(model=None, preprocessor=pp)
        pred = predictor.predict_next_price(df, sequence_length=60, coin_symbol=symbol)
        if not pred:
            return None
        pct = pred.get("price_change_percent", 0) or 0
        conf = pred.get("confidence", 0) or 0
        ai_score = max(0.0, min(100.0, 50.0 + pct * 5.0))
        return {
            "ai_prediction_score": round(ai_score, 2),
            "ai_confidence": round(float(conf), 4),
            "predicted_change_percent": round(float(pct), 2),
        }
    except Exception as e:
        # TF yok / model hatası → sessizce atla (sahte skor üretme)
        print(f"ℹ️ AI snapshot atlandı ({symbol}): {e.__class__.__name__}")
        return None


def _tree_ai_snapshot(df, symbol: str) -> Optional[Dict]:
    """OPSİYONEL tree/ensemble tabanlı AI skoru (Faz 3-4). Default KAPALI; model yoksa None.

    Faz 4: birden çok tree tipini (ENSEMBLE_TREE_TYPES) model_weights ağırlıklarıyla oylar.
    Tree modelleri AVX-güvenli → LSTM'in çöktüğü ortamda bile ai skoru üretebilir.
    """
    try:
        from .config import AutomationConfig as C
        if not getattr(C, "TREE_MODELS_ENABLED", False):
            return None
        from models.ensemble import predict_ensemble
        types = getattr(C, "ENSEMBLE_TREE_TYPES", None) or [getattr(C, "TREE_MODEL_TYPE", "random_forest")]
        ens = predict_ensemble(symbol, df, model_types=types, feature_set_version="v2_ensemble_advanced")
        if ens is None:
            return None
        p_up = float(ens["p_up"])
        ai_score = round(max(0.0, min(100.0, 100.0 * p_up)), 2)  # 50 = nötr
        return {
            "ai_prediction_score": ai_score,
            "ai_confidence": round(float(ens["confidence"]), 4),
            "predicted_change_percent": None,
            "source": "ensemble",
            "ensemble": {"n_models": ens["n_models"], "recommendation": ens["recommendation"],
                         "model_contributions": ens["model_contributions"]},
        }
    except Exception as e:
        print(f"ℹ️ tree/ensemble AI snapshot atlandı ({symbol}): {e.__class__.__name__}")
        return None


def ai_snapshot(df, symbol: str) -> Optional[Dict]:
    """LSTM + (opsiyonel) tree tabanlı AI skorunu birleştirir → research['ai'].

    Tree KAPALI/eğitilmemiş (DEFAULT) → LSTM sonucunu AYNEN döndürür (byte-aynı davranış).
    Yalnız LSTM None + tree var → tree; ikisi de var → confidence×W_TREE ile harman.
    scoring.score_full DEĞİŞMEZ (ai bileşenini opsiyonel okur).
    """
    lstm_ai = _lstm_ai_snapshot(df, symbol)
    tree_ai = _tree_ai_snapshot(df, symbol)

    if tree_ai is None:
        return lstm_ai  # DEFAULT yol → eskisiyle birebir
    if lstm_ai is None:
        return tree_ai

    from .config import AutomationConfig as C
    w_tree = float(getattr(C, "W_TREE", 0.0))
    lw = float(lstm_ai.get("ai_confidence", 0) or 0) * (1.0 - w_tree)
    tw = float(tree_ai.get("ai_confidence", 0) or 0) * w_tree
    tot = lw + tw
    if tot <= 0:
        return lstm_ai
    score = (float(lstm_ai["ai_prediction_score"]) * lw
             + float(tree_ai["ai_prediction_score"]) * tw) / tot
    return {
        "ai_prediction_score": round(score, 2),
        "ai_confidence": round(max(float(lstm_ai.get("ai_confidence", 0) or 0),
                                   float(tree_ai.get("ai_confidence", 0) or 0)), 4),
        "predicted_change_percent": lstm_ai.get("predicted_change_percent"),
        "source": "blend",
    }


def sentiment_snapshot(symbol: str) -> Optional[Dict]:
    """OPSİYONEL haber sentiment. Key yoksa/DEMO kapalıysa None (news_analyzer skip eder)."""
    try:
        from api_cache import demo_mode
        if not os.getenv("NEWSAPI_KEY") and not demo_mode():
            return None  # news_analyzer zaten skip+uyarı yapar
        from news_analyzer import CryptoNewsAnalyzer
        na = CryptoNewsAnalyzer()
        news = na.fetch_all_news(symbol, days=3)
        if not news:
            return None
        sdf = na.analyze_news_sentiment_batch(news)
        if sdf is None or len(sdf) == 0:
            return None
        avg = float(sdf.get("sentiment_score", sdf.iloc[:, 0]).mean()) if hasattr(sdf, "mean") else 0.0
        score = max(0.0, min(100.0, 50.0 + avg * 50.0))
        return {"sentiment_score": round(score, 2)}
    except Exception:
        return None


def whale_snapshot(symbol: str) -> Optional[Dict]:
    """OPSİYONEL whale skoru. Key yoksa/DEMO kapalıysa None (whale_tracker skip eder)."""
    try:
        from api_cache import demo_mode
        if not os.getenv("WHALE_ALERT_API_KEY") and not demo_mode():
            return None
        from whale_tracker import CryptoWhaleTracker
        wt = CryptoWhaleTracker()
        txs = wt.fetch_whale_alert_transactions(symbol, hours=24)
        if not txs:
            return None
        analysis = wt.analyze_whale_transactions(txs)
        score = float(analysis.get("whale_activity_score", 50)) if isinstance(analysis, dict) else 50.0
        return {"whale_score": round(max(0.0, min(100.0, score)), 2)}
    except Exception:
        return None


def market_intel_snapshot(symbol: str) -> Optional[Dict]:
    """OPSİYONEL market zekâsı (haber/sosyal/LLM). Kapalıysa (default) veya veri yoksa None.

    intelligence.engine.build_snapshot cache'li + best-effort; Ollama/kaynak yoksa graceful skip eder.
    """
    try:
        from .config import AutomationConfig as C
        if not C.MARKET_INTEL_ENABLED:
            return None
        from intelligence.engine import build_snapshot
        snap = build_snapshot(symbol)
        if not snap:
            return None
        return {
            "news_quality_score": snap.get("news_quality_score"),
            "social_momentum_score": snap.get("social_momentum_score"),
            "hype_risk": snap.get("hype_risk"),
            "news_risk": snap.get("news_risk"),
            "news_sentiment": snap.get("news_sentiment"),
            "event_summary": snap.get("event_summary"),
        }
    except Exception as e:
        print(f"ℹ️ market_intel snapshot atlandı ({symbol}): {e.__class__.__name__}")
        return None


def research_coin(symbol: str, ticker: Dict = None, df=None, fetcher=None,
                  days: int = None) -> Dict:
    """Bir coin için tam araştırma: teknik + (opsiyonel) AI/sentiment/whale snapshot'ları.

    df verilirse ağ atlanır (test). ticker verilirse hacim/likidite skoru da eklenir.
    Dönüş: research bileşenleri (scoring.score_full ile birleştirilir).
    """
    if df is None:
        if fetcher is None:
            from data_fetcher import CryptoDataFetcher
            fetcher = CryptoDataFetcher()
        d = days or int(os.getenv("LSTM_TRAINING_DAYS", "200"))
        df = fetcher.fetch_ohlcv_data(symbol, days=d)
    if df is None or len(df) < 60:
        return {"symbol": symbol, "error": "insufficient_data", "technical": None}

    try:
        current_price = float(df["close"].iloc[-1])
    except Exception:
        current_price = None

    technical = technical_snapshot(df)

    # Faz 5: regime + anomaly (default KAPALI). Anomali YALNIZ RİSK'i artırır (BUY üretmez).
    regime = None
    anomaly = None
    try:
        from .config import AutomationConfig as C
        if getattr(C, "REGIME_ANOMALY_ENABLED", False):
            from decision.anomaly import detect_anomaly
            from decision.regime import classify_regime
            regime = classify_regime(df, symbol)
            anomaly = detect_anomaly(df, symbol)
            if anomaly and isinstance(technical, dict):
                bump = float(anomaly.get("risk_contribution", 0) or 0)
                if bump > 0:
                    technical["technical_risk"] = min(
                        100.0, float(technical.get("technical_risk", 0) or 0) + bump)
                    warns = technical.setdefault("warnings", [])
                    for w in (anomaly.get("warnings") or []):
                        if w not in warns:
                            warns.append(w)
            try:
                from decision import repository as drepo
                drepo.save_regime_snapshot(symbol, regime, anomaly)
            except Exception:
                pass
    except Exception as e:
        print(f"ℹ️ regime/anomaly atlandı ({symbol}): {e.__class__.__name__}")

    return {
        "symbol": symbol,
        "ticker": ticker,
        "current_price": current_price,
        "technical": technical,
        "ai": ai_snapshot(df, symbol),
        "sentiment": sentiment_snapshot(symbol),
        "whale": whale_snapshot(symbol),
        "market_intel": market_intel_snapshot(symbol),
        "regime": regime,
        "anomaly": anomaly,
    }
