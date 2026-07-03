"""Alembic migration çalıştırıcıları (programatik).

İki track:
- shared: shared şema (tenants/users/coins/prediction_cache/model_registry)
- tenant: her tenant şeması (trades/positions/analysis_results/...)

Entrypoint ve provisioning bu fonksiyonları kullanır.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import List

from alembic import command
from alembic.config import Config

from .session import get_engine, get_session
from .tenancy import is_valid_schema_name

# alembic dizini paket İÇİNDE (wheel ile birlikte gelir)
_PKG_DIR = Path(__file__).resolve().parent  # .../trading_db/
_ALEMBIC_INI = _PKG_DIR / "alembic.ini"
_ALEMBIC_DIR = _PKG_DIR / "alembic"


def _config(scope: str, schema: str = None) -> Config:
    cfg = Config(str(_ALEMBIC_INI))
    cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
    # Her track kendi versiyon dizinini kullanır (ayrı lineer geçmiş)
    subdir = "versions_shared" if scope == "shared" else "versions_tenant"
    cfg.set_main_option("version_locations", str(_ALEMBIC_DIR / subdir))
    x_args = [f"scope={scope}"]
    if schema:
        x_args.append(f"schema={schema}")
    # env.py get_x_argument() bunu okur
    cfg.cmd_opts = SimpleNamespace(x=x_args)
    return cfg


def run_shared_migrations():
    """Shared şema migration'larını uygular (upgrade head)."""
    print("🗄️  Shared migration'lar uygulanıyor...")
    command.upgrade(_config("shared"), "head")
    print("✅ Shared migration tamamlandı")


def run_tenant_migrations(schema: str):
    """Belirli bir tenant şemasına tenant migration'larını uygular."""
    if not is_valid_schema_name(schema):
        raise ValueError(f"Geçersiz tenant şeması: {schema!r}")
    print(f"🏢 Tenant migration uygulanıyor: {schema}")
    command.upgrade(_config("tenant", schema), "head")
    print(f"✅ Tenant migration tamamlandı: {schema}")


def list_tenant_schemas() -> List[str]:
    """Kayıtlı (active/provisioning) tüm tenant şemalarını döndürür."""
    from sqlalchemy import select

    from .models_shared import Tenant

    try:
        with get_session() as s:
            rows = s.execute(select(Tenant.schema_name)).scalars().all()
            return [r for r in rows if is_valid_schema_name(r)]
    except Exception as e:
        print(f"⚠️ Tenant listesi alınamadı: {str(e)}")
        return []


def run_all_tenant_migrations():
    """Kayıtlı tüm tenant şemalarına tenant migration'larını uygular (deploy'da)."""
    schemas = list_tenant_schemas()
    if not schemas:
        print("ℹ️ Kayıtlı tenant yok, tenant migration atlandı")
        return
    for schema in schemas:
        try:
            run_tenant_migrations(schema)
        except Exception as e:
            # Bir tenant'ın hatası diğerlerini durdurmasın
            print(f"❌ {schema} migration hatası: {str(e)}")


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="trading-db migration çalıştırıcı")
    parser.add_argument("scope", choices=["shared", "tenant", "all-tenants"])
    parser.add_argument("--schema", help="tenant scope için hedef şema")
    args = parser.parse_args()
    get_engine()
    if args.scope == "shared":
        run_shared_migrations()
    elif args.scope == "tenant":
        if not args.schema:
            raise SystemExit("--schema gerekli")
        run_tenant_migrations(args.schema)
    else:
        run_all_tenant_migrations()


if __name__ == "__main__":
    _cli()
