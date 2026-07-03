"""trading_db — Crypto Trading platformu paylaşılan veri katmanı.

PostgreSQL + SQLAlchemy + Alembic, schema-per-tenant multi-tenancy.

Uygulama katmanı için ana giriş noktaları:
- ``TradingDatabase``     : mevcut DB API'sinin tenant-aware muadili (drop-in)
- ``AuthManager, User, setup_login_manager`` : auth (eski auth.py yerine)
- ``SystemPersistence``   : oturum/durum kalıcılığı (eski system_persistence.py yerine)
- tenancy: ``set_current_tenant / get_current_tenant / clear_current_tenant``
- ``provision_tenant``    : yeni tenant + şema
"""
from __future__ import annotations

from .auth import AuthManager, User, hash_password, setup_login_manager, verify_password
from .provisioning import deprovision_tenant, list_tenants, provision_tenant
from .repository import TradingDatabase
from .session import (
    dispose_engine,
    get_engine,
    get_session,
    get_tenant_session,
    tenant_session,
)
from .system_persistence import SystemPersistence
from .tenancy import (
    clear_current_tenant,
    get_current_tenant,
    require_current_tenant,
    schema_for_slug,
    set_current_tenant,
    slugify_tenant,
)

__all__ = [
    "TradingDatabase",
    "AuthManager",
    "User",
    "setup_login_manager",
    "hash_password",
    "verify_password",
    "SystemPersistence",
    "get_engine",
    "get_session",
    "get_tenant_session",
    "tenant_session",
    "dispose_engine",
    "set_current_tenant",
    "get_current_tenant",
    "clear_current_tenant",
    "require_current_tenant",
    "schema_for_slug",
    "slugify_tenant",
    "provision_tenant",
    "deprovision_tenant",
    "list_tenants",
]
