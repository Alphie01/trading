"""Tenant provisioning (schema-per-tenant).

Akış (transactional/güvenli):
  1. tenants kaydı (status=provisioning)
  2. CREATE SCHEMA tenant_<slug>
  3. tenant Alembic migration (upgrade head)
  4. tenant seed + tenant admin kullanıcısı (shared users)
  5. status=active
Hata: status=failed (+ opsiyonel DROP SCHEMA cleanup).

CLI: python -m trading_db.provisioning <slug> [--name] [--admin-user] [--admin-password]
Admin API: web katmanı bu fonksiyonu korumalı endpoint'ten çağırır.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from sqlalchemy import select, text

from .migrate import run_tenant_migrations
from .models_shared import Tenant
from .session import get_engine, get_session
from .tenancy import schema_for_slug, slugify_tenant


def _set_status(slug: str, status: str):
    try:
        with get_session() as s:
            t = s.execute(select(Tenant).where(Tenant.slug == slug)).scalar_one_or_none()
            if t:
                t.status = status
    except Exception as e:
        print(f"⚠️ Status güncellenemedi ({slug}={status}): {str(e)}")


def provision_tenant(
    slug: str,
    name: str = None,
    admin_username: str = None,
    admin_password: str = None,
    admin_email: str = None,
    drop_on_failure: bool = False,
) -> Dict:
    """Yeni tenant oluşturur ve şemasını hazırlar. İdempotent (var olan tenant'ı tamamlar)."""
    slug = slugify_tenant(slug)
    schema = schema_for_slug(slug)
    engine = get_engine()

    # 1) tenants kaydı (provisioning)
    try:
        with get_session() as s:
            existing = s.execute(select(Tenant).where(Tenant.slug == slug)).scalar_one_or_none()
            if existing is None:
                s.add(
                    Tenant(
                        slug=slug,
                        schema_name=schema,
                        name=name or slug,
                        status="provisioning",
                    )
                )
            else:
                existing.status = "provisioning"
    except Exception as e:
        return {"success": False, "slug": slug, "error": f"tenant kaydı: {str(e)}"}

    try:
        # 2) CREATE SCHEMA (identifier güvenli: schema_for_slug regex doğruluyor)
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        print(f"✅ Şema oluşturuldu: {schema}")

        # 3) tenant migration
        run_tenant_migrations(schema)

        # 4) tenant seed + admin
        from .seed_tenant import seed_tenant

        seed_tenant(
            schema,
            slug=slug,
            admin_username=admin_username,
            admin_password=admin_password,
            admin_email=admin_email,
        )

        # 5) active
        _set_status(slug, "active")
        print(f"🎉 Tenant hazır: {slug} ({schema})")
        return {"success": True, "slug": slug, "schema_name": schema, "status": "active"}

    except Exception as e:
        print(f"❌ Provisioning hatası ({slug}): {str(e)}")
        _set_status(slug, "failed")
        if drop_on_failure:
            try:
                with engine.begin() as conn:
                    conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
                print(f"🧹 Başarısız şema temizlendi: {schema}")
            except Exception as ce:
                print(f"⚠️ Şema temizlenemedi: {str(ce)}")
        return {"success": False, "slug": slug, "schema_name": schema, "status": "failed", "error": str(e)}


def deprovision_tenant(slug: str, drop_schema: bool = False) -> Dict:
    """Tenant'ı suspend eder (opsiyonel şema drop). Veri kaybı riskli — dikkatli kullan."""
    slug = slugify_tenant(slug)
    schema = schema_for_slug(slug)
    try:
        _set_status(slug, "suspended")
        if drop_schema:
            with get_engine().begin() as conn:
                conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            print(f"🧹 Şema silindi: {schema}")
        return {"success": True, "slug": slug, "status": "suspended"}
    except Exception as e:
        return {"success": False, "slug": slug, "error": str(e)}


def list_tenants() -> list:
    try:
        with get_session() as s:
            rows = s.execute(select(Tenant.slug, Tenant.schema_name, Tenant.status, Tenant.name)).all()
            return [
                {"slug": r[0], "schema_name": r[1], "status": r[2], "name": r[3]} for r in rows
            ]
    except Exception as e:
        print(f"⚠️ Tenant listesi alınamadı: {str(e)}")
        return []


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Tenant provisioning")
    parser.add_argument("slug", help="tenant slug (ör. acme)")
    parser.add_argument("--name")
    parser.add_argument("--admin-user")
    parser.add_argument("--admin-password")
    parser.add_argument("--admin-email")
    parser.add_argument("--drop-on-failure", action="store_true")
    args = parser.parse_args()
    get_engine()
    result = provision_tenant(
        args.slug,
        name=args.name,
        admin_username=args.admin_user,
        admin_password=args.admin_password,
        admin_email=args.admin_email,
        drop_on_failure=args.drop_on_failure,
    )
    print(result)
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
