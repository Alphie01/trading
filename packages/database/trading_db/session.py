"""SQLAlchemy engine + session yönetimi (PostgreSQL, schema-per-tenant).

- ``get_session()``          → shared şema (login, katalog, cache, tenant yönetimi)
- ``get_tenant_session(s)``  → belirli tenant şeması (search_path = "<s>", shared)
- ``tenant_session()``       → aktif tenant contextvar'ından şema alır

İzolasyon: her session AÇILIŞTA search_path'i açıkça set eder → connection-pool
üzerinden tenant sızıntısı olmaz. Shared session tenant şemasını görmez (fail-closed).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .base import SHARED_SCHEMA
from .tenancy import get_current_tenant, is_valid_schema_name

load_dotenv()

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def get_database_url() -> str:
    """DATABASE_URL'i döndürür (psycopg v3 sürücüsü)."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL tanımlı değil. Örn: "
            "postgresql+psycopg://postgres:postgres@localhost:5432/app_db"
        )
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_engine() -> Engine:
    global _engine, _SessionFactory
    if _engine is None:
        is_dev = os.getenv("APP_ENV", os.getenv("FLASK_ENV", "production")).lower() in (
            "development",
            "dev",
        )
        _engine = create_engine(
            get_database_url(),
            echo=is_dev,
            pool_pre_ping=True,
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
            future=True,
        )
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def _factory() -> sessionmaker:
    if _SessionFactory is None:
        get_engine()
    return _SessionFactory  # type: ignore[return-value]


def _set_search_path(session: Session, path: str):
    session.execute(text(f"SET search_path TO {path}"))


@contextmanager
def get_session() -> Iterator[Session]:
    """Shared-only session (search_path = shared). Tenant tabloları görünmez → fail-closed."""
    session = _factory()()
    try:
        _set_search_path(session, SHARED_SCHEMA)
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_tenant_session(schema_name: str) -> Iterator[Session]:
    """Tenant session (search_path = "<schema>", shared) → shared tablolar da erişilebilir."""
    if not is_valid_schema_name(schema_name):
        raise ValueError(f"Geçersiz tenant şeması: {schema_name!r}")
    session = _factory()()
    try:
        _set_search_path(session, f'"{schema_name}", {SHARED_SCHEMA}')
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def tenant_session() -> Iterator[Session]:
    schema = get_current_tenant()
    if not schema:
        raise RuntimeError("Aktif tenant bağlamı yok (tenant_session).")
    with get_tenant_session(schema) as s:
        yield s


def dispose_engine():
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
