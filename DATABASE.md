# DATABASE.md — Veritabanı Rehberi

> Veri katmanı **PostgreSQL + SQLAlchemy + Alembic** ve **schema-per-tenant multi-tenancy**
> kullanır. Tüm kod paylaşılan `packages/database/` (`trading_db`) paketindedir.
> Geçiş özeti için ayrıca **DATABASE_MIGRATION.md**'ye bakın.

---

## 1. Teknoloji ve Katmanlar

| Katman | Konum |
|---|---|
| ORM / models | `packages/database/trading_db/models_shared.py`, `models_tenant.py` |
| Engine / session | `trading_db/session.py` (search_path tabanlı tenant izolasyonu) |
| Tenant bağlamı | `trading_db/tenancy.py` (contextvar) |
| Veri erişim API'si | `trading_db/repository.py` (`TradingDatabase` — eski API korunur) |
| Migration | `trading_db/alembic/` (iki track: shared + tenant), `trading_db/migrate.py` |
| Seed | `trading_db/seed_shared.py`, `seed_tenant.py` |
| Provisioning | `trading_db/provisioning.py` |
| Auth | `trading_db/auth.py` |

- **Tek DB**: PostgreSQL. Eski SQLite/MSSQL katmanları **kaldırıldı**.
- **ORM**: SQLAlchemy 2.0 (declarative). Ham SQL yok (yalnız `execute_query` deprecated kaçış-deliği).

---

## 2. Şema Yerleşimi (schema-per-tenant)

```
PostgreSQL app_db
├── shared            # ortak: tenants, users, coins(katalog), prediction_cache, model_registry
└── tenant_<slug>     # her tenant: watchlist, trades, positions, analysis_results,
                      #             portfolio_summary, system_state
```

**Shared tablolar:** `tenants`, `users` (tenant_id FK), `coins` (katalog), `prediction_cache`, `model_registry`.
**Tenant tablolar:** `watchlist`, `trades`, `positions`, `analysis_results`, `portfolio_summary`, `system_state`.

> "Eğitimler ortak": coin kataloğu + tahmin cache + model metadata shared'dadır; tenant'lar
> arası paylaşılır. Kullanıcının işlemleri/pozisyonları/analizleri tenant şemasında izoledir.

---

## 3. Tenant İzolasyonu (search_path)

- Her session açılışta `search_path` set eder (`session.py`):
  - shared: `SET search_path TO shared`
  - tenant: `SET search_path TO "tenant_<slug>", shared`
- Aktif tenant `tenancy.set_current_tenant(schema)` ile belirlenir. Flask `before_request`
  bunu `current_user.tenant_schema`'dan doldurur, `teardown_request` temizler.
- **Fail-safe**: aktif tenant yoksa tenant-tablosu yazımı atlanır, okuma boş döner
  → cross-tenant sızıntı olmaz. Connection-pool güvenli (her session search_path'i yeniden set eder).

---

## 4. Modeller (naming & tipler)

- **Tablo/kolon adları**: snake_case (eski adlar korundu — @map gerekmedi).
- **PK**: Integer autoincrement; **`users.id` = String** (eski token formatı korunur, uuid'ye çevrilmedi).
- **FK**: `users.tenant_id → tenants.id` (aynı şema). Tenant tablolarındaki `coin_symbol`
  shared `coins`'e **mantıksal** referanstır (DB FK'sı yok — eski davranış).
- **Para/miktar/fiyat**: `Numeric(18,8)` (Decimal). **Yüzde/confidence**: `Numeric(10,4)`.
- **JSON**: `features_used`, `system_state.state_value`, `prediction_cache.*` → `JSONB`.
- **Timestamp**: tabloya özel adlar korundu (`added_date`, `timestamp`, `analysis_timestamp`,
  `last_updated`, `cache_timestamp` ...). `server_default=now()`.

> **Sınır kuralı (API uyumu):** `repository.TradingDatabase` metodları dış katmana Decimal'i
> `float`, DateTime'ı ISO string döndürür → eski response formatı ve Flask `jsonify` uyumu korunur.

---

## 5. Migration Kuralları (Alembic — AI agent MUTLAKA uymalı)

İki track, ayrı lineer geçmiş ve ayrı `alembic_version` (şema başına):
- **shared**: `versions_shared/` — shared şema tabloları.
- **tenant**: `versions_tenant/` — tenant şema tabloları (her tenant şemasına uygulanır).

Komutlar:
```bash
python -m trading_db.migrate shared                       # shared upgrade head
python -m trading_db.migrate tenant --schema tenant_acme  # tek tenant
python -m trading_db.migrate all-tenants                  # kayıtlı tüm tenant'lar (deploy)
```

Kurallar:
1. **Prod'da yalnız `upgrade`** (deploy). `revision --autogenerate` ve otomatik `db push` **YOK**.
2. Şema değişikliğinde ilgili track'e yeni revision ekle; tenant değişikliğini **tüm tenant
   şemalarına** `all-tenants` ile yay.
3. Modelleri (`models_shared/tenant.py`) ve migration'ı birlikte güncelle.
4. Veri kaybı riskli downgrade'lerden kaçın; önce yedek al.
5. Yeni migration'ları elle gözden geçir (search_path/şema-agnostik `op.create_table(schema=None)` deseni).

---

## 6. Seed & Provisioning

```bash
python -m trading_db.seed_shared                # coin kataloğu + platform admin (idempotent)
python -m trading_db.provisioning acme --name "Acme" --admin-user admin_acme --admin-password '***'
```
- Seed **idempotent** (upsert; duplicate yok). Şifreler env'den, log'a yazılmaz.
- Provisioning: `tenants` kaydı → `CREATE SCHEMA` → tenant migration → tenant seed (admin +
  system_state) → `status=active`. Hata → `status=failed` (+opsiyonel drop).
- Admin API: `POST /api/tenants` (platform_admin yetkisi).

---

## 7. Soft Delete vs Hard Delete

- **Soft delete** korundu: `watchlist.is_active`, `positions.is_open`, `prediction_cache.is_valid`.
- Tek hard delete: `cleanup_expired_cache` (7 günden eski prediction_cache). Diğer tablolarda silme yok.
- **`add_coin` veri kaybı bug'ı DÜZELTİLDİ**: eski SQLite `INSERT OR REPLACE` yerine artık
  katalog + watchlist upsert (istatistik sıfırlanmaz).

---

## 8. ⚠️ Veri Kaybı Riski / Dikkat

- `deprovision_tenant(slug, drop_schema=True)` → tenant şemasını **CASCADE siler** (tüm tenant verisi). Onaysız kullanma.
- `DROP SCHEMA`/`DROP TABLE` yalnız provisioning cleanup ve downgrade'de; production DB'sine karşı otomatik önerme.
- Mevcut MSSQL/SQLite verisinin taşınması geçişe dahil DEĞİL (bkz. DATABASE_MIGRATION.md §10).

---

## 9. Kritik / Bozulmaması Gereken Yapılar

1. `search_path` set etmeden tenant tablosuna erişilmez (fail-safe) — bu davranışı kaldırma.
2. `coins.symbol` UNIQUE (katalog anahtarı); `coin_symbol.upper()` normalizasyonu.
3. Shared vs tenant tablo bölümü (coins/prediction_cache shared; trades/positions/... tenant).
4. `repository.TradingDatabase` public metod adları/imzaları ve float/ISO-string sınır dönüşümü (API uyumu).
5. Password hash PBKDF2 formatı (auth uyumu).
6. İki Alembic track ve şema-agnostik migration deseni.
