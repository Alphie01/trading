"""Aktif tenant bağlamı yönetimi.

Çalışma anında hangi tenant şemasına yazılacağı bir contextvar'da tutulur.
Flask uygulamalarında ``before_request`` içinde ``set_current_tenant`` çağrılır;
istek bitince ``clear_current_tenant`` ile temizlenir (pool sızıntısı olmaz).

İzolasyon ilkesi: tenant bağlamı set edilmeden tenant-tablosu erişimi
sessizce yanlış şemaya yazmaz — repository fail-safe davranır (yazma atlanır),
okuma boş/None döner. Cross-tenant sızıntı engellenir.
"""
from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Optional

_current_tenant_schema: ContextVar[Optional[str]] = ContextVar(
    "current_tenant_schema", default=None
)

_SLUG_RE = re.compile(r"^[a-z0-9_]{2,50}$")
SCHEMA_PREFIX = "tenant_"


def slugify_tenant(slug: str) -> str:
    """Tenant slug'ını normalize eder (küçük harf, alfanumerik + alt çizgi)."""
    s = re.sub(r"[^a-z0-9_]+", "_", (slug or "").strip().lower()).strip("_")
    if not s:
        raise ValueError("Geçersiz tenant slug")
    if s[0].isdigit():
        s = f"t_{s}"
    return s[:50]


def schema_for_slug(slug: str) -> str:
    """Slug'dan gerçek PostgreSQL şema adını üretir (SQL-injection'a karşı güvenli)."""
    s = slugify_tenant(slug)
    schema = f"{SCHEMA_PREFIX}{s}"
    if not re.match(r"^[a-z0-9_]{1,63}$", schema):
        raise ValueError(f"Geçersiz şema adı: {schema}")
    return schema


def is_valid_schema_name(schema: str) -> bool:
    return bool(schema) and bool(re.match(r"^[a-z0-9_]{1,63}$", schema))


def set_current_tenant(schema_name: Optional[str]):
    """Aktif tenant şemasını set eder. Döndürülen token clear için kullanılabilir."""
    if schema_name is not None and not is_valid_schema_name(schema_name):
        raise ValueError(f"Geçersiz tenant şeması: {schema_name!r}")
    return _current_tenant_schema.set(schema_name)


def get_current_tenant() -> Optional[str]:
    """Aktif tenant şemasını döndürür (yoksa None)."""
    return _current_tenant_schema.get()


def clear_current_tenant(token=None):
    """Aktif tenant bağlamını temizler."""
    if token is not None:
        try:
            _current_tenant_schema.reset(token)
            return
        except (ValueError, LookupError):
            pass
    _current_tenant_schema.set(None)


def require_current_tenant() -> str:
    """Aktif tenant zorunlu olan yerlerde kullanılır; yoksa hata verir."""
    schema = get_current_tenant()
    if not schema:
        raise RuntimeError(
            "Aktif tenant bağlamı yok — tenant-özel bir işlem tenant seçilmeden çağrıldı."
        )
    return schema
