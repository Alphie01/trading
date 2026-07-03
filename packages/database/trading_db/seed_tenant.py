"""Tenant seed — idempotent (provisioning sırasında çağrılır).

Oluşturur:
- Tenant admin kullanıcısı (shared users, tenant_id = ilgili tenant)
- Tenant system_state varsayılanları (monitoring defaults)

Kullanıcı adı global UNIQUE olduğundan tenant admin varsayılanı ``admin_<slug>``.
Şifre env'den okunur, log'a YAZILMAZ.
"""
from __future__ import annotations

import os
import secrets

from sqlalchemy import select

from .auth import hash_password
from .models_shared import Tenant, User
from .session import get_session
from .tenancy import clear_current_tenant, set_current_tenant


def _tenant_id_for_schema(schema: str):
    with get_session() as s:
        return s.execute(
            select(Tenant.id).where(Tenant.schema_name == schema)
        ).scalar_one_or_none()


def seed_tenant_admin(schema: str, slug: str, admin_username=None, admin_password=None, admin_email=None):
    tenant_id = _tenant_id_for_schema(schema)
    username = admin_username or os.getenv("ADMIN_USERNAME", f"admin_{slug}")
    password = admin_password or os.getenv("ADMIN_PASSWORD", "trading123")
    email = admin_email or os.getenv("ADMIN_EMAIL", f"admin@{slug}.local")

    with get_session() as s:
        existing = s.execute(select(User).where(User.username == username)).scalar_one_or_none()
        if existing is None:
            s.add(
                User(
                    id=secrets.token_urlsafe(16),
                    username=username,
                    password_hash=hash_password(password),
                    email=email,
                    role="admin",
                    tenant_id=tenant_id,
                )
            )
            print(f"✅ Tenant admin oluşturuldu: {username} → {schema}")
        else:
            # Var olan kullanıcıyı bu tenant'a bağla (idempotent onarım)
            if existing.tenant_id is None:
                existing.tenant_id = tenant_id
            print(f"ℹ️ Tenant admin zaten mevcut: {username}")


def seed_tenant_defaults(schema: str):
    """Tenant system_state varsayılanları (aktif tenant bağlamında)."""
    from .repository import TradingDatabase

    token = set_current_tenant(schema)
    try:
        db = TradingDatabase()
        # Yalnız yoksa yaz (idempotent)
        if db.load_system_state("dashboard_monitoring_interval", None) is None:
            db.save_system_state("dashboard_monitoring_interval", 15)
        if db.load_system_state("dashboard_active_monitoring", None) is None:
            db.save_system_state("dashboard_active_monitoring", False)
        if db.load_system_state("trading_auto_enabled", None) is None:
            db.save_system_state("trading_auto_enabled", False)
    finally:
        clear_current_tenant(token)


def seed_tenant(schema: str, slug: str, admin_username=None, admin_password=None, admin_email=None):
    seed_tenant_admin(schema, slug, admin_username, admin_password, admin_email)
    seed_tenant_defaults(schema)
    print(f"🌱 Tenant seed tamamlandı: {slug}")
