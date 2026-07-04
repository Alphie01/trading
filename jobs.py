"""Redis + RQ tabanlı asenkron iş kuyruğu — web ENQUEUE eder, ai-worker YÜRÜTÜR.

Ağır işler (eğitim/analiz/intelligence/scan) web process'ini/thread'ini bloklamaz; ayrı worker
container'da koşar. Job durumu shared `jobs` tablosunda (progress/step/status/result/error).

Tasarım:
- redis/rq LAZY import edilir → web boot'u ağırlaştırmaz, kurulu değilse import patlamaz.
- Redis erişilemezse `enqueue_job` graceful (None, mesaj) döner → web çökmez, "kuyruk erişilemez" gösterir.
- RQ, `worker.handlers.run_job(job_id)`'i çağırır; worker durum güncellemelerini buradan yapar.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = os.getenv("RQ_QUEUE", "trading")
DEFAULT_TIMEOUT = 1800

# Job tipi başına timeout (saniye) — env ile ayarlanabilir
JOB_TIMEOUTS = {
    "TRAIN_COIN": int(os.getenv("TRAIN_COIN_TIMEOUT_SECONDS", "3600")),
    "ANALYZE_COIN": int(os.getenv("ANALYZE_COIN_TIMEOUT_SECONDS", "300")),
    "AUTOMATION_SCAN": int(os.getenv("AUTOMATION_SCAN_TIMEOUT_SECONDS", "1800")),
    "RUN_MARKET_INTELLIGENCE": int(os.getenv("MARKET_INTELLIGENCE_TIMEOUT_SECONDS", "600")),
    "RUN_BACKTEST": int(os.getenv("BACKTEST_TIMEOUT_SECONDS", "7200")),
}

VALID_JOB_TYPES = set(JOB_TIMEOUTS) | {"OLLAMA_NEWS_ANALYSIS", "RUN_SIMULATION_STEP", "DISCOVER_COINS"}

_redis = None


def get_redis():
    global _redis
    if _redis is None:
        import redis  # lazy
        _redis = redis.Redis.from_url(REDIS_URL)
    return _redis


def get_queue():
    from rq import Queue  # lazy
    return Queue(QUEUE_NAME, connection=get_redis(), default_timeout=DEFAULT_TIMEOUT)


def queue_available() -> bool:
    """Redis ping — erişilemezse False (web çökmeden 'kuyruk yok' gösterebilsin)."""
    try:
        return bool(get_redis().ping())
    except Exception:
        return False


def _now():
    return datetime.now(timezone.utc)


def _new_job_id(job_type: str, symbol) -> str:
    base = (symbol or "").replace("/", "").upper() or "GEN"
    return f"{job_type.lower()}_{base}_{_now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"


# ── Enqueue (web tarafı) ──────────────────────────────────────────────────────
def enqueue_job(job_type: str, payload=None, symbol=None, created_by=None, tenant_schema=None):
    """Job satırı oluştur (queued) + RQ'ya at. (job_id, error) döner. Redis yoksa (None, mesaj)."""
    payload = payload or {}
    if job_type not in VALID_JOB_TYPES:
        return None, f"Bilinmeyen job tipi: {job_type}"
    if not queue_available():
        return None, "İş kuyruğu (Redis) şu an erişilemez"

    job_id = _new_job_id(job_type, symbol)
    from trading_db.session import get_session
    from trading_db.models_shared import Job
    try:
        with get_session() as s:
            s.add(Job(job_id=job_id, job_type=job_type, status="queued", symbol=symbol,
                      payload=payload, created_by=created_by, tenant_schema=tenant_schema,
                      progress_percent=0, current_step="Kuyruğa alındı"))
    except Exception as e:
        return None, f"Job kaydı oluşturulamadı: {e}"

    try:
        get_queue().enqueue(
            "worker.handlers.run_job", job_id,
            job_id=job_id, job_timeout=JOB_TIMEOUTS.get(job_type, DEFAULT_TIMEOUT),
        )
    except Exception as e:
        _update(job_id, status="failed", error=f"Enqueue hatası: {e}", finished_at=_now())
        return None, str(e)
    return job_id, None


# ── Sorgu (web tarafı) ────────────────────────────────────────────────────────
def get_job(job_id: str):
    from trading_db.session import get_session
    from trading_db.models_shared import Job
    with get_session() as s:
        j = s.query(Job).filter(Job.job_id == job_id).first()
        return _to_dict(j) if j else None


def list_jobs(limit: int = 50, status: str = None, job_type: str = None):
    from trading_db.session import get_session
    from trading_db.models_shared import Job
    with get_session() as s:
        q = s.query(Job)
        if status:
            q = q.filter(Job.status == status)
        if job_type:
            q = q.filter(Job.job_type == job_type)
        rows = q.order_by(Job.created_at.desc()).limit(min(int(limit), 200)).all()
        return [_to_dict(r) for r in rows]


# ── Durum güncelleme (worker tarafı) ─────────────────────────────────────────
def mark_running(job_id: str, worker_id=None):
    _update(job_id, status="running", started_at=_now(), worker_id=worker_id,
            progress_percent=1, current_step="Başladı")


def update_progress(job_id: str, percent: int = None, step: str = None):
    fields = {}
    if percent is not None:
        fields["progress_percent"] = int(percent)
    if step is not None:
        fields["current_step"] = step[:160]
    if fields:
        _update(job_id, **fields)


def mark_succeeded(job_id: str, result=None):
    _update(job_id, status="succeeded", finished_at=_now(), progress_percent=100,
            current_step="Tamamlandı", result=_json_safe(result or {}))


def mark_failed(job_id: str, error: str):
    _update(job_id, status="failed", finished_at=_now(), error=str(error)[:4000],
            current_step="Hata")


def cancel_job(job_id: str) -> bool:
    """Kuyruktaki (queued) işi iptal işaretle. Çalışan iş için best-effort."""
    j = get_job(job_id)
    if not j:
        return False
    try:
        from rq.job import Job as RQJob
        RQJob.fetch(job_id, connection=get_redis()).cancel()
    except Exception:
        pass
    _update(job_id, status="cancelled", finished_at=_now(), current_step="İptal edildi")
    return True


def retry_job(job_id: str):
    """Aynı tip/payload/symbol ile YENİ bir job enqueue eder (başarısız/iptal işi tekrar dene). (job_id, error)."""
    j = get_job(job_id)
    if not j:
        return None, "Job bulunamadı"
    return enqueue_job(j["job_type"], payload=j.get("payload") or {}, symbol=j.get("symbol"),
                       created_by=j.get("created_by"), tenant_schema=j.get("tenant_schema"))


def list_workers():
    """RQ'da kayıtlı canlı worker'lar (heartbeat/durum). Redis yoksa boş liste."""
    out = []
    try:
        from rq import Worker
        for w in Worker.all(connection=get_redis()):
            info = {"worker_id": getattr(w, "name", "?")}
            for attr, key in (("get_state", "state"), ("successful_job_count", "succeeded"),
                              ("failed_job_count", "failed")):
                try:
                    v = getattr(w, attr)
                    info[key] = v() if callable(v) else v
                except Exception:
                    info[key] = None
            try:
                cur = w.get_current_job()
                info["current_job"] = cur.id if cur else None
            except Exception:
                info["current_job"] = None
            try:
                hb = w.last_heartbeat
                info["last_heartbeat"] = hb.isoformat() if hb else None
            except Exception:
                info["last_heartbeat"] = None
            out.append(info)
    except Exception as e:
        print(f"⚠️ worker listesi alınamadı: {e}")
    return out


def _update(job_id: str, **fields):
    from trading_db.session import get_session
    from trading_db.models_shared import Job
    try:
        with get_session() as s:
            j = s.query(Job).filter(Job.job_id == job_id).first()
            if not j:
                return
            for k, v in fields.items():
                setattr(j, k, v)
    except Exception as e:
        print(f"⚠️ job {job_id} güncellenemedi: {e}")


def _json_safe(obj):
    """JSONB'ye yazılabilir hale getir (numpy/datetime vb. → str)."""
    import json
    try:
        json.dumps(obj)
        return obj
    except Exception:
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return {"repr": str(obj)[:2000]}


def _iso(dt):
    return dt.isoformat() if dt else None


def _to_dict(j):
    return {
        "job_id": j.job_id,
        "job_type": j.job_type,
        "status": j.status,
        "symbol": j.symbol,
        "payload": j.payload,
        "progress_percent": j.progress_percent or 0,
        "current_step": j.current_step,
        "result": j.result,
        "error": j.error,
        "created_by": j.created_by,
        "tenant_schema": j.tenant_schema,
        "worker_id": j.worker_id,
        "created_at": _iso(j.created_at),
        "started_at": _iso(j.started_at),
        "finished_at": _iso(j.finished_at),
    }
