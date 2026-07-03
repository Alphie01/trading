"""SQLAlchemy declarative base'leri.

Multi-tenancy: schema-per-tenant, PostgreSQL ``search_path`` ile.
- Modeller şema-agnostiktir (Table'larda schema set edilmez).
- Çalışma anında session ``search_path``'i set eder:
  * shared işlemler   → ``search_path = shared``
  * tenant işlemleri  → ``search_path = "tenant_<slug>", shared``
  Böylece shared tablolar (coins/users/...) shared şemada, tenant tabloları
  (trades/positions/...) ilgili tenant şemasında çözülür.

İki ayrı Base/MetaData tutulur ki Alembic her track'i (shared/tenant) ayrı
hedefleyebilsin.
"""
from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

SHARED_SCHEMA = "shared"
SCHEMA_PREFIX = "tenant_"

# Deterministik index/constraint adları (Alembic autogenerate güvenli)
_NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Şema-agnostik metadata (schema search_path ile çözülür)
shared_metadata = MetaData(naming_convention=_NAMING)
tenant_metadata = MetaData(naming_convention=_NAMING)

SharedBase = declarative_base(metadata=shared_metadata)
TenantBase = declarative_base(metadata=tenant_metadata)
