"""RQ worker job handler'ları.

`run_job(job_id)` RQ tarafından çağrılır: jobs tablosundan işi okur, tipe göre AĞIR işi çalıştırır
(train/analyze/scan/intelligence), ilerleme + sonuç/hata'yı jobs tablosuna yazar. Her işi izole eder —
bir job çökse worker ayakta kalır, diğer işleri alır.
"""
from __future__ import annotations

import os
import socket
import traceback

import jobs as jobq  # root jobs.py (worker cwd = /app)

WORKER_ID = os.getenv("WORKER_ID") or f"ai-worker-{socket.gethostname()}"


def run_job(job_id: str):
    """RQ entrypoint. jobs tablosundaki işi tipine göre yürütür."""
    j = jobq.get_job(job_id)
    if not j:
        print(f"⚠️ Job bulunamadı: {job_id}")
        return
    job_type = j["job_type"]
    symbol = j.get("symbol")
    payload = j.get("payload") or {}
    tenant_schema = j.get("tenant_schema")

    print(f"📥 Job alındı: {job_type} {symbol or ''} ({job_id})")
    jobq.mark_running(job_id, worker_id=WORKER_ID)

    if tenant_schema:
        try:
            from trading_db import set_current_tenant
            set_current_tenant(tenant_schema)
        except Exception as e:
            print(f"⚠️ tenant bağlamı set edilemedi ({tenant_schema}): {e}")

    try:
        handler = HANDLERS.get(job_type)
        if handler is None:
            jobq.mark_failed(job_id, f"Handler yok: {job_type}")
            print(f"❌ Handler yok: {job_type}")
            return
        result = handler(job_id, symbol, payload)
        jobq.mark_succeeded(job_id, result=result or {})
        print(f"✅ Job tamamlandı: {job_type} {symbol or ''} ({job_id})")
    except Exception as e:
        traceback.print_exc()
        jobq.mark_failed(job_id, str(e))
        print(f"❌ Job başarısız: {job_type} {symbol or ''} — {e}")
    finally:
        try:
            from trading_db import clear_current_tenant
            clear_current_tenant()
        except Exception:
            pass


# ── Handler'lar ───────────────────────────────────────────────────────────────
def _train_coin(job_id, symbol, payload):
    if not symbol:
        raise ValueError("symbol gerekli")
    force = bool(payload.get("force"))
    jobq.update_progress(job_id, 10, "Eğitim başlıyor (veri + LSTM/DQN/Hybrid)")
    from comprehensive_trainer import ComprehensiveTrainer
    res = ComprehensiveTrainer().train_coin_sync(symbol, is_fine_tune=force)
    jobq.update_progress(job_id, 95, "Model kaydedildi")
    # Ham sonuç büyük/JSON-güvensiz olabilir → özet
    if isinstance(res, dict):
        return {"ok": True, "models": list(res.keys()), "summary": jobq._json_safe(res)}
    return {"ok": True, "result": str(res)[:2000]}


def _analyze_coin(job_id, symbol, payload):
    if not symbol:
        raise ValueError("symbol gerekli")
    jobq.update_progress(job_id, 20, "Araştırma + skorlama")
    from automation.engine import run_research
    out = run_research(symbols=[symbol.upper()], persist=True)
    from automation import repository as arepo
    sym = symbol.upper() if "/" in symbol else f"{symbol.upper()}/USDT"
    jobq.update_progress(job_id, 90, "Skor hazır")
    return {"detail": jobq._json_safe(arepo.get_coin_detail(sym)),
            "signal": jobq._json_safe((out.get("signals") or [None])[0])}


def _automation_scan(job_id, symbol, payload):
    from automation.engine import run_discovery, run_research
    jobq.update_progress(job_id, 20, "Discovery (tarama)")
    d = run_discovery(persist=True)
    jobq.update_progress(job_id, 60, "Research (araştırma + sinyal + simülasyon)")
    r = run_research(persist=True)
    return {"discovery": jobq._json_safe(d), "research": jobq._json_safe(r)}


def _market_intelligence(job_id, symbol, payload):
    if not symbol:
        raise ValueError("symbol gerekli")
    jobq.update_progress(job_id, 30, "Haber/sosyal toplama + LLM analizi")
    from intelligence.engine import build_snapshot
    snap = build_snapshot(symbol, persist=True)
    return {"snapshot": jobq._json_safe(snap)}


HANDLERS = {
    "TRAIN_COIN": _train_coin,
    "ANALYZE_COIN": _analyze_coin,
    "AUTOMATION_SCAN": _automation_scan,
    "RUN_MARKET_INTELLIGENCE": _market_intelligence,
}
