"""Market Intelligence konfigürasyonu (INTELLIGENCE_*/OLLAMA_*/INTEL_* env).

automation/config.py stilinde; tümü .env ile yönetilebilir. Güvenlik: default'lar KAPALI/güvenli.
"""
from __future__ import annotations

import os


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


class IntelligenceConfig:
    # Katmanın tamamı (collect + analyze + scoring hook) bu bayrakla açılır.
    ENABLED = _bool("INTELLIGENCE_ENABLED", "false")

    # --- Ollama (LLM) ---
    OLLAMA_ENABLED = _bool("OLLAMA_ENABLED", "false")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
    OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
    OLLAMA_REQUIRE_JSON = _bool("OLLAMA_REQUIRE_JSON", "true")

    # --- Toplama (collectors) ---
    NEWS_LOOKBACK_HOURS = int(os.getenv("INTEL_NEWS_LOOKBACK_HOURS", "48"))
    MAX_ITEMS_PER_SOURCE = int(os.getenv("INTEL_MAX_ITEMS_PER_SOURCE", "30"))
    COLLECT_TIMEOUT = int(os.getenv("INTEL_COLLECT_TIMEOUT", "10"))
    # Ollama'ya yalnız dedup sonrası en alakalı N haber gider (latency/maliyet kontrolü)
    MAX_LLM_ITEMS = int(os.getenv("INTEL_MAX_LLM_ITEMS", "8"))

    # --- Sosyal medya (Faz 5) — Reddit + Google Trends (momentum/hype/bot) ---
    SOCIAL_ENABLED = _bool("INTEL_SOCIAL_ENABLED", "false")
    SOCIAL_MAX_POSTS = int(os.getenv("INTEL_SOCIAL_MAX_POSTS", "40"))
    GOOGLE_TRENDS_ENABLED = _bool("INTEL_GOOGLE_TRENDS_ENABLED", "true")

    # --- Cache ---
    SNAPSHOT_TTL = int(os.getenv("INTEL_SNAPSHOT_TTL_SECONDS", "900"))  # 15 dk
    OLLAMA_CACHE_TTL = int(os.getenv("INTEL_OLLAMA_CACHE_TTL_SECONDS", "21600"))  # 6 saat

    @classmethod
    def as_dict(cls) -> dict:
        return {k: getattr(cls, k) for k in dir(cls) if k.isupper() and not k.startswith("_")}
