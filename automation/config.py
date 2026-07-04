"""Otomasyon motoru konfigürasyonu (AUTO_* env değişkenleri).

Tümü .env ile yönetilebilir. Güvenlik: AUTO_TRADE_ENABLED default KAPALI.
"""
from __future__ import annotations

import os


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def _csv(name: str, default: str) -> list:
    return [s.strip().upper() for s in os.getenv(name, default).split(",") if s.strip()]


class AutomationConfig:
    # Discovery / scan
    DISCOVERY_ENABLED = _bool("AUTO_DISCOVERY_ENABLED", "false")
    SCAN_INTERVAL_MINUTES = int(os.getenv("AUTO_DISCOVERY_INTERVAL_MINUTES", "15"))
    MIN_24H_VOLUME_USDT = float(os.getenv("AUTO_MIN_24H_VOLUME_USDT", "10000000"))
    MAX_SYMBOLS_PER_SCAN = int(os.getenv("AUTO_MAX_SYMBOLS_PER_SCAN", "50"))
    ALLOWED_QUOTES = _csv("AUTO_ALLOWED_QUOTES", "USDT")
    EXCLUDED_SYMBOLS = set(_csv("AUTO_EXCLUDED_SYMBOLS", "USDC/USDT,TUSD/USDT,FDUSD/USDT"))
    MAX_SPREAD_PERCENT = float(os.getenv("AUTO_MAX_SPREAD_PERCENT", "1.0"))

    # Modlar / güvenlik
    DEMO_MODE = _bool("AUTO_DEMO_MODE", "false")
    TRADE_ENABLED = _bool("AUTO_TRADE_ENABLED", "false")  # canlı trade — Faz 7, default KAPALI
    # Market Intelligence (haber/sosyal/LLM) skorlamaya bağlansın mı — default KAPALI (opt-in)
    MARKET_INTEL_ENABLED = _bool("AUTO_MARKET_INTEL_ENABLED", "false")
    # Tree modelleri (Faz 3) ai_prediction_score'a blend edilsin mi — default KAPALI (opt-in).
    # KAPALI iken ai_snapshot davranışı AYNEN eskisi (byte-aynı).
    TREE_MODELS_ENABLED = _bool("AUTO_TREE_MODELS_ENABLED", "false")
    TREE_MODEL_TYPE = os.getenv("AUTO_TREE_MODEL_TYPE", "random_forest")
    W_TREE = float(os.getenv("AUTO_W_TREE", "0.0"))  # LSTM↔tree harman ağırlığı (0 → tree katkısı yok)
    # Ensemble (Faz 4) — birden çok tree tipi + dinamik ağırlık. W_ENSEMBLE Faz 7 (score_full_v2) için.
    W_ENSEMBLE = float(os.getenv("AUTO_W_ENSEMBLE", "0.0"))
    ENSEMBLE_TREE_TYPES = [s.strip() for s in os.getenv("AUTO_ENSEMBLE_TREE_TYPES", "random_forest").split(",") if s.strip()]
    WEIGHT_EWMA_ALPHA = float(os.getenv("AUTO_WEIGHT_EWMA_ALPHA", "0.3"))  # feedback ağırlık güncelleme hızı
    # Regime + anomaly (Faz 5) — research'e rejim/anomali ekler; anomali yalnız RİSK'i artırır.
    # Default KAPALI (opt-in). KAPALI iken research davranışı AYNEN eskisi.
    REGIME_ANOMALY_ENABLED = _bool("AUTO_REGIME_ANOMALY_ENABLED", "false")

    # Stablecoin base'leri (discovery dışı)
    STABLE_BASES = set(_csv(
        "AUTO_STABLE_BASES",
        "USDC,TUSD,FDUSD,BUSD,DAI,USDP,USDD,PYUSD,GUSD,EUR,TRY,GBP,AUD,BRL",
    ))

    # Skorlama ağırlıkları (opportunity_score bileşenleri)
    W_TECHNICAL = float(os.getenv("AUTO_W_TECHNICAL", "0.35"))
    W_VOLUME_LIQUIDITY = float(os.getenv("AUTO_W_VOLUME_LIQUIDITY", "0.20"))
    W_AI_PREDICTION = float(os.getenv("AUTO_W_AI_PREDICTION", "0.25"))
    W_SENTIMENT = float(os.getenv("AUTO_W_SENTIMENT", "0.10"))
    W_WHALE = float(os.getenv("AUTO_W_WHALE", "0.10"))
    # Market Intelligence bileşen ağırlıkları (score_full mevcut olanlar üzerinden normalize eder)
    W_NEWS_QUALITY = float(os.getenv("AUTO_W_NEWS_QUALITY", "0.10"))
    W_SOCIAL_MOMENTUM = float(os.getenv("AUTO_W_SOCIAL_MOMENTUM", "0.10"))

    # Watchlist eşiği
    WATCHLIST_MIN_OPPORTUNITY = float(os.getenv("AUTO_WATCHLIST_MIN_OPPORTUNITY", "60"))
    HOT_MIN_OPPORTUNITY = float(os.getenv("AUTO_HOT_MIN_OPPORTUNITY", "75"))
    MAX_RISK_SCORE = float(os.getenv("AUTO_MAX_RISK_SCORE", "70"))

    @classmethod
    def as_dict(cls) -> dict:
        return {
            k: getattr(cls, k)
            for k in dir(cls)
            if k.isupper() and not k.startswith("_")
        }

    # Settings formundan ayarlanabilen alanlar: form-anahtarı → (attr, tip)
    _OVERRIDE_MAP = {
        "AUTO_DISCOVERY_ENABLED": ("DISCOVERY_ENABLED", "bool"),
        "AUTO_DISCOVERY_INTERVAL_MINUTES": ("SCAN_INTERVAL_MINUTES", "int"),
        "AUTO_MIN_24H_VOLUME_USDT": ("MIN_24H_VOLUME_USDT", "float"),
        "AUTO_MAX_SYMBOLS_PER_SCAN": ("MAX_SYMBOLS_PER_SCAN", "int"),
        "AUTO_MAX_SPREAD_PERCENT": ("MAX_SPREAD_PERCENT", "float"),
        "AUTO_DEMO_MODE": ("DEMO_MODE", "bool"),
        "AUTO_TRADE_ENABLED": ("TRADE_ENABLED", "bool"),
        "AUTO_MARKET_INTEL_ENABLED": ("MARKET_INTEL_ENABLED", "bool"),
        "AUTO_TREE_MODELS_ENABLED": ("TREE_MODELS_ENABLED", "bool"),
        "AUTO_W_TREE": ("W_TREE", "float"),
        "AUTO_REGIME_ANOMALY_ENABLED": ("REGIME_ANOMALY_ENABLED", "bool"),
        "AUTO_W_NEWS_QUALITY": ("W_NEWS_QUALITY", "float"),
        "AUTO_W_SOCIAL_MOMENTUM": ("W_SOCIAL_MOMENTUM", "float"),
        "AUTO_WATCHLIST_MIN_OPPORTUNITY": ("WATCHLIST_MIN_OPPORTUNITY", "float"),
        "AUTO_HOT_MIN_OPPORTUNITY": ("HOT_MIN_OPPORTUNITY", "float"),
        "AUTO_MAX_RISK_SCORE": ("MAX_RISK_SCORE", "float"),
    }

    @classmethod
    def current_settings(cls) -> dict:
        """Settings formunda gösterilecek ayarlanabilir ayarların güncel değerleri."""
        out = {}
        for key, (attr, typ) in cls._OVERRIDE_MAP.items():
            out[key] = getattr(cls, attr)
        return out

    @classmethod
    def apply_overrides(cls, data: dict) -> dict:
        """Kaydedilen ayarları class attribute'larına uygular (runtime override).

        DEMO_MODE ve TRADE_ENABLED gibi anahtarlar ilgili modüllerin os.getenv okumasını
        etkilemez; bu yüzden ortam değişkenleri de (aynı process için) güncellenir.
        """
        if not data:
            return cls.current_settings()
        applied = {}
        for key, (attr, typ) in cls._OVERRIDE_MAP.items():
            if key not in data or data[key] is None:
                continue
            raw = data[key]
            try:
                if typ == "bool":
                    val = str(raw).lower() in ("1", "true", "yes", "on")
                elif typ == "int":
                    val = int(float(raw))
                elif typ == "float":
                    val = float(raw)
                else:
                    val = raw
            except (ValueError, TypeError):
                continue
            setattr(cls, attr, val)
            applied[key] = val
            # DEMO_MODE / AUTO_TRADE_ENABLED aynı process içindeki os.getenv okuyanlar için de yaz
            if key == "AUTO_DEMO_MODE":
                os.environ["DEMO_MODE"] = "true" if val else "false"
            if key == "AUTO_TRADE_ENABLED":
                os.environ["AUTO_TRADE_ENABLED"] = "true" if val else "false"
        return applied
