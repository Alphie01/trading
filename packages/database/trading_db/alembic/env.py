"""Alembic environment — iki track (shared / tenant), search_path tabanlı.

Kullanım (migrate.py üzerinden):
  scope=shared              → shared şemayı hedefler (search_path=shared)
  scope=tenant  schema=X    → X tenant şemasını hedefler (search_path="X")

DATABASE_URL env'den okunur (alembic.ini'de URL yoktur).
"""
from __future__ import annotations

from alembic import context
from sqlalchemy import text

# trading_db paketini import edilebilir kıl (kurulu değilse de çalışsın)
import os
import sys

# trading_db paketini import edilebilir kıl (kurulu değilse: packages/database dizini)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from trading_db.base import SHARED_SCHEMA, shared_metadata, tenant_metadata  # noqa: E402
import trading_db.models_shared  # noqa: E402,F401  (tabloları metadata'ya kaydeder)
import trading_db.models_tenant  # noqa: E402,F401
from trading_db.session import get_engine  # noqa: E402
from trading_db.tenancy import is_valid_schema_name  # noqa: E402

config = context.config

_x = context.get_x_argument(as_dictionary=True)
_scope = _x.get("scope", "shared")
_schema = _x.get("schema")

if _scope == "shared":
    target_metadata = shared_metadata
    _version_schema = SHARED_SCHEMA
elif _scope == "tenant":
    if not _schema or not is_valid_schema_name(_schema):
        raise SystemExit(f"tenant scope için geçerli -x schema gerekli (verilen: {_schema!r})")
    target_metadata = tenant_metadata
    _version_schema = _schema
else:
    raise SystemExit(f"Bilinmeyen scope: {_scope!r}")


def run_migrations_online():
    connectable = get_engine()
    with connectable.connect() as connection:
        # Hedef şemayı oluştur ve search_path'i ayarla; Alembic'in kendi transaction'ından
        # ÖNCE commit et (aksi halde DDL bağlantı kapanışında geri alınır).
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{_version_schema}"'))
        if _scope == "tenant":
            connection.execute(text(f'SET search_path TO "{_version_schema}"'))
        else:
            connection.execute(text(f"SET search_path TO {SHARED_SCHEMA}"))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=_version_schema,
            include_schemas=False,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


def run_migrations_offline():
    # Offline mod desteklenmez (search_path/şema oluşturma canlı bağlantı gerektirir)
    raise SystemExit("Offline migration desteklenmiyor; online kullanın.")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
