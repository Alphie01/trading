"""Dış API çağrıları için yardımcılar: TTL response cache, retry + exponential
backoff, ve DEMO_MODE kontrolü.

İlke (veri doğruluğu):
- Production'da API key yoksa **mock veri ÜRETİLMEZ**; ilgili adım atlanır + açık uyarı.
- Mock yalnız `DEMO_MODE=true` iken üretilebilir.
- Üretilen/mock veri `data_source`/`is_mock` ile etiketlenmelidir (sinyal kirlenmesini önler).
"""
from __future__ import annotations

import functools
import logging
import os
import threading
import time
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger("api")


def demo_mode() -> bool:
    """Mock/demo veri üretimine YALNIZ bu true iken izin verilir."""
    return os.getenv("DEMO_MODE", "false").lower() == "true"


def warn_missing_key(key_name: str, feature: str) -> None:
    """API key eksikse standart uyarı (sessiz mock yerine)."""
    msg = f"⚠️ {key_name} missing. {feature} skipped."
    logger.warning(msg)
    print(msg)


class TTLCache:
    """Basit, thread-safe TTL cache (dış API yanıtları için)."""

    def __init__(self, ttl_seconds: int, max_items: int = 256):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: "dict[str, Tuple[float, Any]]" = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            ent = self._store.get(key)
            if ent and (time.time() - ent[0]) < self.ttl:
                return ent[1]
            if ent:
                self._store.pop(key, None)
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._store) >= self.max_items:
                oldest = min(self._store.items(), key=lambda kv: kv[1][0])[0]
                self._store.pop(oldest, None)
            self._store[key] = (time.time(), value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


def retry(
    times: int = 3,
    base_delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    label: str = "",
):
    """Exponential backoff ile retry decorator'ı.

    Not: HTTP timeout'u çağrının kendisinde ayarlanmalıdır (ör. requests timeout=).
    """

    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            delay = base_delay
            last_exc = None
            name = label or getattr(fn, "__name__", "api_call")
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:  # noqa: BLE001
                    last_exc = e
                    if attempt < times:
                        logger.warning(
                            "%s attempt %d/%d failed: %s — retry in %.1fs",
                            name, attempt, times, e, delay,
                        )
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        logger.error("%s failed after %d attempts: %s", name, times, e)
            if last_exc:
                raise last_exc
        return wrap

    return deco


def cached_call(cache: TTLCache, key: str, producer: Callable[[], Any]) -> Any:
    """key cache'te varsa döndür; yoksa producer() çağır, cache'le, döndür."""
    hit = cache.get(key)
    if hit is not None:
        return hit
    val = producer()
    if val is not None:
        cache.set(key, val)
    return val
