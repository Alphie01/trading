"""Otomasyon zamanlayıcı — periyodik discovery + research (durdurulabilir daemon thread).

Mevcut `schedule` + daemon-thread desenini (training_scheduler/CoinMonitor) izler.
AUTO_DISCOVERY_ENABLED=false ise başlatılmaz. Canlı trade YOK (Faz 7).
"""
from __future__ import annotations

import threading
import time

from .config import AutomationConfig as C


class AutomationScheduler:
    def __init__(self):
        self.running = False
        self.thread = None
        self.last_run = None

    def _job(self):
        if not C.DISCOVERY_ENABLED:
            return
        try:
            from .engine import run_discovery, run_research
            print("🤖 Automation turu başlıyor (discovery + research + simülasyon işleme)...")
            d = run_discovery(persist=True)
            r = run_research(persist=True)
            print(f"✅ Automation turu: scanned={d['scanned']} passed={d['passed']} "
                  f"researched={r['researched']} watchlisted={r['watchlisted']}")
        except Exception as e:
            print(f"❌ Automation job hatası: {e}")

        # Faz 4: model ağırlıklarını değerlendirmelerden EWMA ile güncelle (best-effort).
        # Yalnız model_weights yazar → ensemble KAPALIYKEN (default) canlı akışa etkisi YOK.
        try:
            from . import feedback
            fb = feedback.run_feedback()
            if fb.get("updated"):
                print(f"⚖️ Model ağırlıkları güncellendi: {fb['updated']} model")
        except Exception as e:
            print(f"⚠️ feedback atlandı: {e}")

    def _loop(self):
        # Manuel zamanlayıcı (schedule dep'i gerekmez) → DISCOVERY_ENABLED / SCAN_INTERVAL RUNTIME değişebilir
        # (Settings'ten açılınca yeniden başlatmaya gerek yok; thread hep döner, iş sadece açıkken çalışır).
        while self.running:
            try:
                if C.DISCOVERY_ENABLED:
                    now = time.time()
                    if self.last_run is None or (now - self.last_run) >= max(1, C.SCAN_INTERVAL_MINUTES) * 60:
                        self.last_run = now
                        self._job()
            except Exception as e:
                print(f"⚠️ scheduler loop hatası: {e}")
            time.sleep(10)

    def start(self):
        # Thread HER ZAMAN başlar (self-gating); AUTO_DISCOVERY_ENABLED runtime'da açılınca otomatik koşar.
        if self.running:
            return True
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"✅ Automation scheduler thread aktif (enabled={C.DISCOVERY_ENABLED}, her {C.SCAN_INTERVAL_MINUTES} dk)")
        return True

    def stop(self):
        self.running = False
        print("🛑 Automation scheduler durduruldu")

    def status(self) -> dict:
        return {
            "running": self.running,
            "enabled": C.DISCOVERY_ENABLED,
            "interval_minutes": C.SCAN_INTERVAL_MINUTES,
            "trade_enabled": C.TRADE_ENABLED,
        }


_scheduler = None


def get_scheduler() -> AutomationScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AutomationScheduler()
    return _scheduler
