"""Ollama (LLM) istemcisi — sync, `requests` tabanlı (monolith'te httpx yok).

Özellikler: timeout, retry, format=json zorlaması, JSON validate + 1 kez repair (<think>/```json temizleme),
content-hash cache, model availability (/api/tags), health(), ve GRACEFUL fallback (Ollama yoksa None).

Ollama YALNIZ yorumlar; trade kararı vermez. Bozuk çıktı trade skoruna DAHİL EDİLMEZ (None döner).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Dict, Optional

from .config import IntelligenceConfig as C

logger = logging.getLogger("intelligence.ollama")

# Cache: api_cache.TTLCache (proje geneli yardımcı)
try:
    from api_cache import TTLCache
    _cache = TTLCache(ttl_seconds=C.OLLAMA_CACHE_TTL, max_items=1024)
except Exception:  # pragma: no cover
    _cache = None

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _extract_json(text: str) -> Optional[Dict]:
    """LLM metninden JSON çıkar: <think>/```json temizle → json.loads; olmazsa ilk dengeli {…}."""
    if not text:
        return None
    cleaned = _THINK.sub("", text)
    cleaned = _FENCE.sub("", cleaned).strip()
    # 1) doğrudan
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # 2) repair: ilk dengeli {…} bloğunu bul
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


class OllamaClient:
    def __init__(self, base_url: str = None, model: str = None, timeout: int = None):
        self.base_url = (base_url or C.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or C.OLLAMA_MODEL
        self.timeout = timeout or C.OLLAMA_TIMEOUT
        self.max_retries = C.OLLAMA_MAX_RETRIES

    # ---- durum ----
    def health(self) -> Dict:
        if not C.OLLAMA_ENABLED:
            return {"ok": False, "enabled": False, "error": "OLLAMA_ENABLED=false", "base_url": self.base_url}
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m.get("name") for m in (r.json().get("models") or [])]
                return {"ok": True, "enabled": True, "base_url": self.base_url, "model": self.model,
                        "models": models, "model_present": self.model in models}
            return {"ok": False, "enabled": True, "base_url": self.base_url, "error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"ok": False, "enabled": True, "base_url": self.base_url, "error": str(e)}

    def available(self) -> bool:
        return bool(self.health().get("ok"))

    # ---- üretim ----
    def generate_json(self, prompt: str, cache_key: str = None) -> Optional[Dict]:
        """format=json ile JSON yanıt üret. Başarısızsa None (graceful, sistem çökmez)."""
        if not C.OLLAMA_ENABLED:
            return None
        if cache_key and _cache is not None:
            hit = _cache.get(cache_key)
            if hit is not None:
                return hit
        try:
            import requests
        except Exception:
            return None

        # qwen3 "thinking" yumuşak kapatma (diğer modeller için zararsız ek metin)
        p = prompt + ("\n/no_think" if self.model.lower().startswith("qwen") else "")
        payload = {
            "model": self.model,
            "prompt": p,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 700},
        }
        if C.OLLAMA_REQUIRE_JSON:
            payload["format"] = "json"

        last_err = None
        for attempt in range(1, self.max_retries + 2):  # 1 ilk + max_retries
            try:
                r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}"
                    continue
                parsed = _extract_json(r.json().get("response", ""))
                if parsed is not None:
                    if cache_key and _cache is not None:
                        _cache.set(cache_key, parsed)
                    return parsed
                last_err = "json_parse_failed"
            except Exception as e:  # Timeout dahil
                last_err = e.__class__.__name__

        logger.warning("⚠️ Ollama generate_json başarısız (%s) — LLM analizi atlandı.", last_err)
        return None

    def analyze_news(self, item: Dict, symbol: str) -> Optional[Dict]:
        from .prompts import news_prompt
        key = "news:" + hashlib.sha1(
            f"{self.model}|{symbol}|{item.get('content_hash') or item.get('title', '')}".encode("utf-8", "ignore")
        ).hexdigest()
        return self.generate_json(news_prompt(item, symbol), cache_key=key)


_singleton: Optional[OllamaClient] = None


def get_client() -> OllamaClient:
    global _singleton
    if _singleton is None:
        _singleton = OllamaClient()
    return _singleton
