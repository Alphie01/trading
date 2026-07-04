"""ai-worker launcher — RQ worker'ı 'trading' kuyruğunu dinler; ağır işleri (train/analyze/scan/intel)
web'de değil BURADA koşar. Docker `command: python worker/run_worker.py` ile başlatılır (SERVICE_ROLE=ai-worker).

Redis erişilebilir olana kadar bekler (worker, web'den bağımsız ayakta kalır).
"""
from __future__ import annotations

import os
import sys
import time

# Repo kökünü path'e ekle (import jobs / worker.handlers / comprehensive_trainer ...)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


def _wait_for_redis(jobq, tries=60, delay=2):
    for i in range(1, tries + 1):
        if jobq.queue_available():
            return True
        print(f"⏳ Redis bekleniyor ({i}/{tries}) — {jobq.REDIS_URL}")
        time.sleep(delay)
    return False


def main():
    import jobs as jobq
    from rq import Queue, Worker

    print("👷 AI worker başlatılıyor...")
    if not _wait_for_redis(jobq):
        print("❌ Redis erişilemedi — worker çıkıyor (container restart edecek)")
        sys.exit(1)

    conn = jobq.get_redis()
    queue = Queue(jobq.QUEUE_NAME, connection=conn)
    worker = Worker([queue], connection=conn, name=os.getenv("WORKER_ID"))

    # Periyodik otomasyon (discovery→research→sinyal→simülasyon) WORKER'da koşar (web'de DEĞİL).
    # Daemon thread; iş yalnız AUTO_DISCOVERY_ENABLED=true iken çalışır (self-gating).
    try:
        from automation.scheduler import get_scheduler
        get_scheduler().start()
        print("✅ Automation scheduler worker'da aktif (AUTO_DISCOVERY_ENABLED ile çalışır)")
    except Exception as e:
        print(f"⚠️ Automation scheduler başlatılamadı: {e}")

    # Haftalık fine-tune scheduler de WORKER'da (comprehensive_trainer → TF; web'de import edilmez).
    try:
        import os as _os
        from training_scheduler import init_scheduler
        _ts = init_scheduler(
            schedule_day=_os.getenv('TRAINING_SCHEDULE_DAY', 'sunday'),
            schedule_time=_os.getenv('TRAINING_SCHEDULE_TIME', '02:00'),
            enable_notifications=True,
        )
        _ts.start_scheduler()
        print("✅ Training scheduler worker'da aktif (haftalık fine-tune)")
    except Exception as e:
        print(f"⚠️ Training scheduler başlatılamadı: {e}")

    print(f"✅ AI worker hazır — queue='{jobq.QUEUE_NAME}' redis='{jobq.REDIS_URL}'")
    # burst=False → sürekli dinle; SIGTERM'de düzgün kapanır (Docker restart uyumlu)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
