"""Otomasyon ayarlarının kalıcılığı — default tenant'ın system_state'inde tutulur.

Settings formu → save_automation_settings → system_state('automation_settings') + apply_overrides.
Engine her çalışmada load_and_apply() ile kaydedilmiş ayarları uygular (env üzerine).
"""
from __future__ import annotations

from typing import Dict

from .config import AutomationConfig

_KEY = "automation_settings"


def _schema():
    from . import tenant_repo
    return tenant_repo.default_tenant_schema()


def load_automation_settings() -> Dict:
    schema = _schema()
    if not schema:
        return {}
    try:
        from trading_db import TradingDatabase, clear_current_tenant, set_current_tenant
        tok = set_current_tenant(schema)
        try:
            return TradingDatabase().load_system_state(_KEY, {}) or {}
        finally:
            clear_current_tenant(tok)
    except Exception as e:
        print(f"⚠️ load_automation_settings hatası: {e}")
        return {}


def save_automation_settings(data: Dict) -> bool:
    schema = _schema()
    if not schema:
        print("⚠️ default tenant yok — otomasyon ayarları kaydedilemedi")
        return False
    try:
        from trading_db import TradingDatabase, clear_current_tenant, set_current_tenant
        tok = set_current_tenant(schema)
        try:
            TradingDatabase().save_system_state(_KEY, data)
        finally:
            clear_current_tenant(tok)
        AutomationConfig.apply_overrides(data)  # anında etkinleştir
        return True
    except Exception as e:
        print(f"❌ save_automation_settings hatası: {e}")
        return False


def load_and_apply() -> Dict:
    """Kaydedilmiş ayarları yükle + uygula (engine çalışmadan önce çağrılır)."""
    data = load_automation_settings()
    if data:
        AutomationConfig.apply_overrides(data)
    return data
