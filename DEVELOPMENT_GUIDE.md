# DEVELOPMENT_GUIDE.md — Geliştirme Rehberi

> Lokal kurulum, çalıştırma ve yaygın geliştirme senaryoları. Komutlar repo kökünden (`trading/`) verilmiştir.

---

## 1. Gerekli Bağımlılıklar

- **Python** (monolith 3.9 hedefli; mikroservisler 3.13 uyumlu). Uyumsuzluk yaşarsan ilgili `requirements.txt`'in sürüm kısıtlarına bak.
- **pip + venv**.
- **PostgreSQL** (16+ önerilir) veya Docker. Sürücü: `psycopg[binary]` (`packages/database` ile gelir). ODBC/pyodbc **gerekmez** (MSSQL kaldırıldı).
- **Mac M1/M2**: Ek kurulum gerekmez; `tf_config.py` Metal/CPU ayarını otomatik yapar.
- İnternet: Frontend CDN'lere (Bootstrap/Font Awesome/Chart.js/Socket.IO) ve Binance/News/Whale API'lerine erişim.

---

## 2. Lokal Kurulum

Her uygulamanın **kendi venv + requirements + .env**'i vardır. İhtiyacına göre birini veya hepsini kur.

### Monolith
```bash
cd trading
python -m venv trading_env && source trading_env/bin/activate
pip install -r requirements.txt
cp .env.example .env   # ⚠️ .env.example gerçek sırlar içeriyor — düzenle & yeni değerler koy
```

### AI Service
```bash
cd trading/ai-service
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # ai-service/.env.example ŞU AN BOŞ — anahtarları ai-service/CLAUDE.md'den al
```

### Web Service
```bash
cd trading/web-service
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # web-service/.env.example ŞU AN BOŞ — anahtarları web-service/CLAUDE.md'den al
```

---

## 3. Environment Kurulumu

- Değerleri `.env`'e koy; **koda/dokümana/log'a yazma**.
- Anahtar grupları: `DATABASE_URL`, `POSTGRES_HOST/PORT`, `PLATFORM_ADMIN_*`/`ADMIN_*`/`DEFAULT_TENANT_SLUG`, `BINANCE_API_KEY/SECRET/TESTNET`, `NEWSAPI_KEY`, `WHALE_ALERT_API_KEY`, `FLASK_SECRET_KEY`/`SECRET_KEY`, `WEB_PORT`, `AI_SERVICE_URL`, `AI_SERVICE_API_KEY`, `LSTM_EPOCHS`, `LSTM_TRAINING_DAYS`, `DQN_EPISODES`. (`MSSQL_*` kaldırıldı.)
- LSTM eğitim ayarları senaryoları için `LSTM_CONFIG_README.md`'ye bak (hızlı test / dengeli / yüksek accuracy profilleri).
- API key'ler opsiyoneldir ama yoksa haber/whale **mock veri** üretir (production'da sahte sinyal riski).

---

## 4. Database Kurulumu (PostgreSQL)

**En kolay:** `docker compose up --build` (postgres + web; migrate/seed/provision otomatik).

**Elle:**
```bash
pip install -e packages/database
export DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/app_db
python -m trading_db.migrate shared        # shared şema
python -m trading_db.seed_shared           # coin kataloğu + platform admin
python -m trading_db.provisioning default --name Default   # default tenant + şema
```
> Şema/tenancy kuralları için **DATABASE.md**, geçiş özeti için **DATABASE_MIGRATION.md**.

## 5. Migration Çalıştırma (Alembic — iki track)

```bash
python -m trading_db.migrate shared                       # shared şema
python -m trading_db.migrate tenant --schema tenant_acme  # tek tenant
python -m trading_db.migrate all-tenants                  # kayıtlı tüm tenant'lar
```
- Yeni kolon: `models_shared.py`/`models_tenant.py` + ilgili track'e Alembic revision ekle;
  tenant değişikliğini `all-tenants` ile yay. **Prod'da yalnız `upgrade`** (autogenerate/db-push yok).

## 6. Seed Çalıştırma

- `python -m trading_db.seed_shared` → coin kataloğu (BTC/ETH/BNB) + platform admin (idempotent).
- Tenant seed provisioning'in parçasıdır (`provisioning <slug>` → tenant admin + system_state defaults).
- Tümü **idempotent** (upsert; tekrar çalıştırınca duplicate yok).

---

## 7. Development Server Başlatma

```bash
# --- Monolith ---
python run_dashboard.py           # Flask + SocketIO dashboard
python main.py                    # interaktif CLI analizi
python main.py --demo             # BTC otomatik demo

# --- Mikroservisler (birlikte) ---
bash start_microservices.sh       # ai-service (8000) → web-service; loglar *.log

# --- Mikroservisler (ayrı) ---
cd ai-service && python run_ai_service.py    # http://localhost:8000  (Swagger: /docs)
cd web-service && python run_web_service.py  # http://localhost:<WEB_PORT>
```
> `web-service` portu: `config.py` varsayılanı **25629**; `.env`'deki `WEB_PORT` gerçek değeri belirler. README/MICROSERVICES_README'deki "5000" güncel olmayabilir.

## 8. Test Çalıştırma

```bash
python test_ai_service.py             # AI service çalışırken
python test_web_service.py            # Web service çalışırken
python quick_test.py                  # hızlı LSTM/pipeline dumanı testi
python test_comprehensive_system.py   # uçtan uca
python test_database_cache.py         # DB cache testi
```
> `pytest`/`tests/` yapısı yok; hepsi çalıştırılabilir script'ler ve çoğu ilgili servisin ayakta olmasını gerektirir.

## 9. Build Alma

- Derleme/bundle adımı **yoktur** (Python + CDN frontend). "Build" = venv + `pip install`. Frontend için transpile/bundler yok.

---

## 10. Yeni Feature Ekleme Adımları

1. **Nerede?** Monolith mi, ai-service mi, web-service mi çalışıyor belirle.
2. İlgili `CLAUDE.md`'yi ve **benzer mevcut örneği** oku (aynı pattern'i izle).
3. Duplication tablosunu (`CLAUDE.md §6.1`) kontrol et — dosyanın kaç kopyası var, hangisini düzenliyorsun.
4. Contract'lara dokunuyor musun? (AI API / Jinja2 `analysis.*` / SocketIO event / DB şeması) — dokunuyorsan tüm tüketicileri güncelle.
5. Gerekiyorsa `requirements.txt`'e (doğru servisinkine) dependency ekle.
6. İlgili `test_*.py` ile veya servisi çalıştırıp doğrula.

## 11. Yeni API Endpoint Ekleme

**AI Service (FastAPI)** — `ai-service/ai_service.py`:
1. `@app.post("/yeni")` veya `@app.get(...)` ekle; gerekiyorsa Pydantic request/response modeli tanımla.
2. `success`/`error` alanlarını koru; gerçek hatalarda `HTTPException`.
3. `web-service`'in tüketmesi gerekiyorsa `AIServiceClient`'a (`web_service.py`) karşılık gelen metodu + timeout ekle.

**Web Service / Monolith (Flask)**:
1. `@app.route('/api/yeni', methods=[...])`; korumalıysa `@login_required`.
2. `jsonify({success, data/error})` döndür; handler adına `api_` öneki.
3. AI service çağrısı gerekiyorsa mevcut `asyncio.new_event_loop()` köprü desenini kullan.

## 12. Yeni Frontend Sayfası Ekleme

1. `templates/` (monolith, Flask'ın render ettiği) — ve gerekiyorsa `web-service/templates/` kopyasına — yeni `.html` ekle.
2. Bootstrap 5 + Font Awesome 6 CDN'lerini head'e koy (mevcut sayfadan kopyala); stil için tek inline `<style>`.
3. Route ekle: `@app.route('/sayfa')` → `render_template('sayfa.html', ...)` (gerekiyorsa `@login_required`).
4. JS'i inline `<script>` içine yaz (`fetch().then(...)` veya form POST). Grafik gerekiyorsa Chart.js CDN'i (portfolio örneğine bak).
5. Türkçe UI metni kullan; `url_for(...)` ile link/form action ver.
6. Detaylı kurallar: **web-service/CLAUDE.md**.

## 13. Yeni Database Alanı Ekleme

1. **DATABASE.md §5**'i oku.
2. `packages/database/trading_db/models_shared.py` veya `models_tenant.py`'de kolonu ekle (shared mı tenant mı doğru track).
3. İlgili Alembic revision'ı ekle (`versions_shared/` veya `versions_tenant/`) ve `repository.py`'deki metodları güncelle.
4. Deploy'da `migrate shared` + (tenant değişikliği ise) `migrate all-tenants` çalıştır.
5. Yeni alan bir model feature'ı ise: **feature sırası ve `n_features` (25) değişir → tüm model cache geçersiz olur**; bunu belirt (bkz. ai-service/CLAUDE.md).

## 14. Yeni Model Feature Ekleme (özel dikkat)

1. `data_preprocessor.py`'de göstergeyi ekle; OHLCV'nin ilk 5 sırasını ve **close=index 3** kuralını bozma.
2. `feature_columns` mutasyon bug'ına dikkat (`prepare_data` her çağrıda listeyi extend eder).
3. Feature sayısı değişince eski `.h5`/`_scaler.pkl` cache **uyumsuz** olur → yeniden eğitim gerekir.
4. DQN state boyutu (~31) değişirse kayıtlı model yüklenmez, agent sıfırdan kurulur.

---

## 15. Debug Yöntemleri

- **Loglar**: `dashboard.log`, `training_scheduler.log`, mikroservis `ai-service/ai_service.log` / `web-service/web_service.log`.
- **AI service**: `http://localhost:8000/docs` (Swagger) ile endpoint'leri elle dene; `curl http://localhost:8000/health`.
- **TF cihaz sorunları**: `tf_config.print_training_device_info()` / `get_current_device()`; M1/M2 çökmelerinde `tf_config` import sırasını doğrula.
- **DB**: `docker compose exec postgres psql -U postgres -d app_db` ile bağlan; şemalar `\dn`, tablolar `\dt shared.*` / `\dt tenant_default.*`. Bağlantı testi: `python -c "from trading_db import TradingDatabase; print(TradingDatabase().test_connection())"`.
- **Web**: Flask `WEB_DEBUG=True` (varsayılan) ile hata izleri; tarayıcı console'unda `fetch`/SocketIO hataları.

## 16. Sık Karşılaşılan Hatalar

| Belirti | Olası neden / çözüm |
|---|---|
| AI service bağlantı hatası (web) | ai-service ayakta değil → `curl :8000/health`; `AI_SERVICE_URL` doğru mu |
| TensorFlow Metal çökmesi (M1/M2) | `tf_config` TF'den önce import edilmemiş; import sırasını düzelt |
| `.h5` yüklenemiyor | `directional_accuracy` custom metriği register edilmemiş; feature sayısı/model uyumsuz |
| Tahminler saçma/yanlış ölçekli | Scaler `fit_scaler=True/False` karışmış; `_scaler.pkl` eksik |
| Tenant verisi görünmüyor/boş | Aktif tenant bağlamı yok — login gerekli veya `set_current_tenant(schema)` (fail-safe boş döner) |
| `DATABASE_URL tanımlı değil` | `.env`'de `DATABASE_URL` set et (postgresql+psycopg://...) |
| UI değişikliği yansımıyor | Yanlış template kopyası düzenlendi (`templates/` vs `web-service/`) |
| Kod değişikliği etkisiz | Dosyanın başka bir kopyası çalışıyor (bkz. CLAUDE.md §6.1 duplication) |
| Haber/whale sonuçları "sahte" | API key yok → mock/demo veri fallback'i devrede |
| Web portu beklenenden farklı | `config.py` default 25629; `.env` `WEB_PORT` kontrol et |
| Migration'da tablo yanlış şemada | `env.py` search_path'i commit'ten sonra set eder; `op.create_table(schema=None)` deseni korunmalı |
