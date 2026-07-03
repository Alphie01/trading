"""Tenant-özel otomasyon çıktıları (sinyal/uyarı/watchlist) — aktif/default tenant'a yazılır.

Global engine (onaylanan karar): keşif/skorlar SHARED; sinyaller/watchlist/kararlar TENANT.
Engine bir request bağlamında olmadığından, default tenant şeması çözülüp o şemaya yazılır.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import select

from trading_db.models_shared import Tenant
from trading_db.models_tenant import AutomationAlert, AutomationSignal, Watchlist
from trading_db.session import get_session, get_tenant_session


def default_tenant_schema() -> Optional[str]:
    """DEFAULT_TENANT_SLUG (varsayılan 'default') tenant'ının aktif şemasını döndürür."""
    slug = os.getenv("DEFAULT_TENANT_SLUG", "default")
    try:
        with get_session() as s:
            return s.execute(
                select(Tenant.schema_name).where(
                    Tenant.slug == slug, Tenant.status == "active"
                )
            ).scalar_one_or_none()
    except Exception as e:
        print(f"⚠️ default_tenant_schema hatası: {e}")
        return None


def save_signal(schema: str, score: Dict) -> None:
    signal = {"STRONG_BUY": "BUY", "BUY": "BUY", "SELL": "SELL", "STRONG_SELL": "SELL"}.get(
        score.get("recommendation", "HOLD"), "HOLD"
    )
    try:
        with get_tenant_session(schema) as s:
            s.add(AutomationSignal(
                symbol=(score.get("symbol") or "").upper(),
                signal=signal,
                recommendation=score.get("recommendation"),
                opportunity_score=score.get("opportunity_score"),
                risk_score=score.get("risk_score"),
                confidence=score.get("confidence"),
                reasons=score.get("reasons"),
                warnings=score.get("warnings"),
                source="automation",
            ))
    except Exception as e:
        print(f"❌ save_signal hatası: {e}")


def save_alert(schema: str, symbol: Optional[str], level: str, title: str, message: str) -> None:
    try:
        with get_tenant_session(schema) as s:
            s.add(AutomationAlert(
                symbol=(symbol or "").upper() or None,
                level=level, title=title, message=message,
            ))
    except Exception as e:
        print(f"❌ save_alert hatası: {e}")


def add_to_watchlist(schema: str, symbol: str) -> None:
    """Aday coin'i tenant izleme listesine ekler (upsert, is_active=True)."""
    try:
        sym = symbol.upper()
        with get_tenant_session(schema) as s:
            w = s.execute(
                select(Watchlist).where(Watchlist.coin_symbol == sym)
            ).scalar_one_or_none()
            if w is None:
                s.add(Watchlist(coin_symbol=sym, is_active=True))
            else:
                w.is_active = True
    except Exception as e:
        print(f"❌ add_to_watchlist hatası: {e}")


def recent_signals(schema: str, limit: int = 20) -> List[Dict]:
    try:
        with get_tenant_session(schema) as s:
            rows = s.execute(
                select(AutomationSignal).order_by(AutomationSignal.created_at.desc()).limit(limit)
            ).scalars().all()
            return [{
                "symbol": r.symbol, "signal": r.signal, "recommendation": r.recommendation,
                "opportunity_score": float(r.opportunity_score) if r.opportunity_score is not None else None,
                "risk_score": float(r.risk_score) if r.risk_score is not None else None,
                "confidence": float(r.confidence) if r.confidence is not None else None,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            } for r in rows]
    except Exception as e:
        print(f"❌ recent_signals hatası: {e}")
        return []


def recent_alerts(schema: str, limit: int = 20) -> List[Dict]:
    try:
        with get_tenant_session(schema) as s:
            rows = s.execute(
                select(AutomationAlert).order_by(AutomationAlert.created_at.desc()).limit(limit)
            ).scalars().all()
            return [{
                "symbol": r.symbol, "level": r.level, "title": r.title, "message": r.message,
                "is_read": r.is_read,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            } for r in rows]
    except Exception as e:
        print(f"❌ recent_alerts hatası: {e}")
        return []
