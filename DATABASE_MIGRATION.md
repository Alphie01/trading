# DATABASE_MIGRATION.md — MSSQL/SQLite → PostgreSQL + SQLAlchemy/Alembic + Multi-Tenancy

Bu doküman, veri katmanının eski **MSSQL + SQLite (ham SQL)** yapısından yeni
**PostgreSQL + SQLAlchemy + Alembic (schema-per-tenant multi-tenancy)** yapısına
geçişini özetler.

---

## 1. Geçiş Özeti

| | Eski | Yeni |
|---|---|---|
| Veritabanı | MSSQL (pyodbc) **veya** SQLite (sqlite3) | **PostgreSQL** (tek) |
| Erişim | Ham SQL string'leri (2 ayrı sınıf) | **SQLAlchemy ORM** (tek `trading_db` paketi) |
| Migration | Yok (`CREATE TABLE IF NOT EXISTS`) | **Alembic** (iki track: shared + tenant) |
| Seed | `create_mssql_database.py` içi ad-hoc | **Idempotent** `seed_shared` + `seed_tenant` |
| Çok kiracılılık | Yok | **Schema-per-tenant** (`tenant_<slug>` şemaları) |
| Kod tekrarı | `database.py`/`auth.py`/... 2 kopya | Tek paylaşılan paket `packages/database/` |
| Deploy | Elle | Docker + entrypoint (bekle→migrate→seed→başlat) |

**Neden Prisma değil?** Proje %100 Python'dır; Prisma bir Node/TS ORM'idir. Python'ın
muadili olan **SQLAlchemy (Prisma Client) + Alembic (Prisma Migrate) + Python seed**
kullanıldı. Tüm hedefler (otomatik migrate, idempotent seed, container wait-for-db,
prod-güvenli akış) karşılanır.

---

## 2. Kaldırılan / Eklenen Paketler

**Kaldırıldı:** `pyodbc` (MSSQL), örtük `sqlite3` kullanımı, `bcrypt`/`PyJWT` (kullanılmıyordu).

**Eklendi:** `SQLAlchemy>=2.0.25`, `alembic>=1.13`, `psycopg[binary]>=3.1` (üçü de
`packages/database` paketinin bağımlılığı; kök ve `web-service` requirements'a da eklendi).

**Silinen dosyalar:** `database.py`(×2), `mssql_database.py`(×2), `auth.py`(×2),
`system_persistence.py`(×2), `create_mssql_database.py`, `test_env_mssql.py`.

**Yeni paket:** `packages/database/` (kurulabilir `trading-db`).

---

## 3. Yeni Şema (Prisma-benzeri model açıklaması)

Tek PostgreSQL veritabanı, çok şema:

**`shared` şeması (ortak / eğitim / katalog):**
- `tenants` — tenant kayıt defteri (slug, schema_name, status).
- `users` — global kullanıcılar; `tenant_id` FK → tenants (platform admin için NULL).
- `coins` — coin **kataloğu** (symbol, name, current_price, price_change_24h).
- `prediction_cache` — model tahmin cache'i (tenant'lar arası ortak).
- `model_registry` — eğitilmiş model metadata'sı (model_cache/ dosyalarına referans).

**`tenant_<slug>` şeması (her tenant için ayrı):**
- `watchlist` — tenant'ın izleme listesi (coin_symbol, is_active, analysis_count, last_analysis).
- `trades`, `positions`, `portfolio_summary`, `analysis_results`, `system_state`.

Modeller: `packages/database/trading_db/models_shared.py`, `models_tenant.py`.
- **PK:** Integer autoincrement (users.id = String — eski token formatı korunur).
- **Para/miktar/fiyat:** `Numeric(18,8)` (Decimal). Yüzde/confidence: `Numeric(10,4)/(5,2)`.
- **JSON:** `features_used`, `state_value`, `prediction_cache.*` → `JSONB`.
- Sınır kuralı: repository, Decimal'i dış katmana `float`, DateTime'ı ISO string döndürür
  (eski API/response formatı korunur, Flask jsonify uyumlu).

---

## 4. Tenant Mekaniği

- **İzolasyon:** PostgreSQL `search_path`. Her session açılışta set edilir:
  - shared: `SET search_path TO shared`
  - tenant: `SET search_path TO "tenant_<slug>", shared`
- **Çözümleme:** Login sonrası `User.tenant_schema` doldurulur → Flask `before_request`
  `set_current_tenant(schema)` çağırır → repository o şemaya yazar; `teardown_request` temizler.
- **Eğitim ortak:** Model/scheduler kodu yalnız shared tabloları (coins/prediction_cache/
  model_registry) kullanır → tenant bağlamı gerekmez.
- **Fail-safe:** Aktif tenant yoksa tenant-tablosu yazımı atlanır, okuma boş döner
  (cross-tenant sızıntı olmaz).

---

## 5. Migration Komutları

```bash
# Paketi kur
pip install -e packages/database         # veya: pip install packages/database

# Shared şema (tenants/users/coins/prediction_cache/model_registry)
python -m trading_db.migrate shared

# Belirli bir tenant şeması
python -m trading_db.migrate tenant --schema tenant_acme

# Kayıtlı TÜM tenant şemaları (deploy'da şema değişikliği yayma)
python -m trading_db.migrate all-tenants

# (Geliştirme) yeni migration üretme — dikkatle, elle gözden geçir
cd packages/database
alembic -x scope=shared upgrade head
# tenant için: alembic -x scope=tenant -x schema=tenant_acme upgrade head
```

> **Prod kuralı:** yalnız `upgrade` (deploy). `revision --autogenerate` ve otomatik
> `db push` prod'da KULLANILMAZ.

---

## 6. Seed & Provisioning Komutları

```bash
# Shared seed (coin kataloğu + platform admin) — idempotent
python -m trading_db.seed_shared

# Yeni tenant provision (şema + tenant migration + tenant seed)
python -m trading_db.provisioning acme --name "Acme" --admin-user admin_acme --admin-password '***'
```
Admin API: `POST /api/tenants` (platform_admin yetkisi) — body `{slug, name, admin_username, admin_password}`.

---

## 7. Docker Compose ile Çalıştırma

```bash
cp .env.example .env        # değerleri düzenle (gerçek secret'ları .env'e)
docker compose up --build
```
Açılış akışı (entrypoint): PostgreSQL bekle → shared migrate → tenant migrate (tüm) →
shared seed → default tenant provision → Flask başlat.

Servisler: `postgres` (16-alpine, healthcheck), `web` (Flask, `depends_on: service_healthy`).
`ai` servisi DB'siz ve ML-ağır olduğundan opsiyoneldir (compose'da yorumlu).

---

## 8. Production Deploy Akışı

1. `DATABASE_URL` ve admin secret'ları güvenli biçimde (secret manager) sağla.
2. Görüntüyü build/push et.
3. Container açılışında entrypoint: `migrate shared` → `migrate all-tenants` → `seed_shared`
   → default tenant provision → uygulama. (Hepsi idempotent.)
4. Yeni tenant: admin API veya CLI `provisioning`.
5. Şema değişikliği: yeni Alembic revision (elle gözden geçir) → `migrate shared` +
   `migrate all-tenants` sıralı.

---

## 9. Rollback Önerileri

- **Migration:** `alembic -x scope=shared downgrade -1` / tenant için schema hedefli downgrade.
  Veri kaybı riskli downgrade'lerden kaçının; önce yedek alın.
- **Tenant:** hatalı provisioning `status=failed` bırakır; `deprovision_tenant(slug, drop_schema=True)`
  ile şema temizlenebilir (veri kaybı — dikkat).
- **Uygulama:** eski MSSQL/SQLite katmanı silindiği için geri dönüş, git revert + eski
  container imajı gerektirir.

---

## 10. Bilinen Riskler / Notlar

- **Mevcut MSSQL/SQLite verisinin taşınması bu geçişe DAHİL DEĞİLDİR.** Yeni model
  (tenant + katalog/watchlist ayrımı) farklıdır; üretim verisi varsa ayrı bir ETL +
  "hangi veri hangi tenant'a ait" kararı gerekir.
- **`coins` ayrımı:** eski tekil `coins` tablosu → shared `coins` (katalog) + tenant
  `watchlist` (izleme durumu). `add_coin` her ikisini de günceller.
- **`analysis_results` tenant-özeldir:** model kodundan gelen doğrudan `save_*_analysis`
  çağrıları yalnız aktif tenant bağlamında yazar; bağlam yoksa güvenle atlanır.
- **Decimal↔JSON:** çözüldü — repository sınırda `float`/ISO string döndürür.
- **Şifre uyumu:** PBKDF2 formatı korunmuştur; eski hash'ler ve yeni seed uyumludur.
- **ai-service:** DB kullanmaz; `trading_db` importu try/except ile opsiyoneldir.

---

## 11. Manuel Kontrol Listesi

- [ ] `docker compose up --build` → postgres healthy, web migrate+seed+provision logları, Flask ayakta.
- [ ] `shared`, `tenant_default` şemaları oluştu (`\dn` psql).
- [ ] Login: seed admin ile giriş çalışıyor; `tenant_schema` doğru.
- [ ] Tenant izolasyonu: farklı tenant kullanıcıları birbirinin trade/position/analiz'ini görmüyor.
- [ ] Ortak katalog/tahmin (coins/prediction_cache) tüm tenant'lara görünür.
- [ ] `POST /api/tenants` (platform_admin) yeni tenant + şema açıyor; yetkisiz 403.
- [ ] Seed/provision iki kez → duplicate yok.
- [ ] API response formatları (portfolio, recent_trades, analyze) değişmemiş; Decimal alanlar sayı.
- [ ] `grep -ri "pyodbc\|mssql\|GETDATE" --include=*.py` → yalnız doküman.
- [ ] `.env` gerçek secret içermiyor commit'te; `.env.example` placeholder.
