"""Hafif analiz pipeline zamanlama/instrumentation aracı (config ile aç/kapa).

Kullanım:
    from perf_timing import analysis_timer, step, cache_event

    with analysis_timer("BTC/USDT"):
        with step("fetch_ohlcv"):
            df = fetch(...)
        with step("lstm_predict"):
            ...
    # analiz sonunda tek satır breakdown loglanır (yalnız ANALYSIS_TIMING_ENABLED=true iken)

Kapalıyken (varsayılan) hiçbir yan etki/log spam yok — step() no-op'a düşer.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger("analysis.timing")

_current: ContextVar[Optional["TimingCollector"]] = ContextVar("timing_collector", default=None)


def timing_enabled() -> bool:
    return os.getenv("ANALYSIS_TIMING_ENABLED", "false").lower() == "true"


class TimingCollector:
    """Bir analiz isteği boyunca adım sürelerini biriktirir."""

    def __init__(self, label: str):
        self.label = label
        self.steps: "dict[str, float]" = {}
        self._order: "list[str]" = []
        self.start = time.perf_counter()

    def add(self, name: str, dur: float):
        if name not in self.steps:
            self._order.append(name)
        self.steps[name] = self.steps.get(name, 0.0) + dur

    def report(self) -> str:
        total = time.perf_counter() - self.start
        parts = " | ".join(f"{n}: {self.steps[n]:.2f}s" for n in self._order)
        return f"⏱️ Analysis Timing [{self.label}]\n  {parts} | total: {total:.2f}s"


@contextmanager
def analysis_timer(label: str):
    """Bir analiz isteğini sarar; çıkışta breakdown loglar (etkinse)."""
    if not timing_enabled():
        yield None
        return
    col = TimingCollector(label)
    token = _current.set(col)
    try:
        yield col
    finally:
        _current.reset(token)
        try:
            logger.info(col.report())
        except Exception:
            pass
        print(col.report())


@contextmanager
def step(name: str):
    """Tek bir adımı ölçer. Aktif analysis_timer yoksa no-op."""
    col = _current.get()
    if col is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        col.add(name, time.perf_counter() - t0)


def record(name: str, dur: float):
    """Manuel süre kaydı (harici ölçüm için)."""
    col = _current.get()
    if col is not None:
        col.add(name, dur)


def cache_event(kind: str, key: str, hit: bool):
    """Cache hit/miss logu (yalnız timing etkinse)."""
    if timing_enabled():
        logger.info("%s %s: %s", "✅ HIT" if hit else "❌ MISS", kind, key)
        print(f"{'✅ CACHE HIT' if hit else '❌ CACHE MISS'} [{kind}] {key}")
