# CLAUDE.md — Crypto Trading AI System

> Bu dosya, geliştiricilere ve AI coding agent'lara bu projede **doğru ve güvenli** çalışmaları için rehberdir.
> Kod yazmadan önce bu dosyayı ve ilgili alt-CLAUDE.md dosyasını oku.

---

## 1. Proje Nedir?

Kripto para ticareti için **hibrit deep-learning** tabanlı bir analiz/otomasyon sistemi. LSTM (fiyat tahmini) + DQN (aksiyon seçimi) + teknik analiz + haber sentiment + whale tracking'i birleştirir; sonuçları bir web dashboard'da gösterir ve opsiyonel olarak Binance üzerinde işlem açar.

Proje **iki mimari nesil içerir ve bu iki nesil aynı repoda bir arada durur**:

1. **Monolith** (repo kökü) — Orijinal sistem. Flask `web_app.py` + CLI `main.py` + tüm modeller + DB + trading tek klasörde.
2. **Mikroservisler** (`ai-service/`, `web-service/`) — Monolitten ayrıştırılmış yeni nesil. AI işleme (FastAPI) ile web arayüzü (Flask) ayrı servisler.

> **Not:** `ai-service/` ve `web-service/` klasörleri, monolitteki dosyaların **kopyalarını** içerir. Bu, en önemli bakım riskidir — bkz. [§6 Kritik Kurallar](#6-ai-agent-için-kesin-kurallar-mutlaka-uyulmalı).

Dış workspace'te bir de **boş** `trading-ai-service/` klasörü var (kullanılmayan artık). Bu klasöre kod yazma.

---

## 2. Teknoloji Stack (yalnızca gerçekten kullanılanlar)

| Katman | Teknoloji |
|---|---|
| Dil | Python (monolith: 3.9 hedefli `requirements.txt`; mikroservisler: 3.13 uyumlu sürümler) |
| ML / DL | TensorFlow 2.x + Keras (LSTM, custom DQN, Hybrid), scikit-learn (`MinMaxScaler`), NumPy, Pandas |
| RL | Custom Double-DQN (`gym`/`gymnasium`, `stable-baselines3` requirements'ta var ama DQN **elle** yazılmış) |
| Web API | **FastAPI** + Uvicorn + Pydantic (`ai-service`) |
| Web UI backend | **Flask** + Flask-SocketIO + Flask-Login + Werkzeug (`web-service` ve monolith) |
| HTTP client | `httpx` (async), `requests` |
| Borsa | `ccxt` (Binance spot + futures) |
| Veritabanı | **PostgreSQL** (SQLAlchemy 2.0 ORM + Alembic, **schema-per-tenant** multi-tenancy). Paylaşılan paket: `packages/database/` (`trading_db`). Bkz. `DATABASE.md`, `DATABASE_MIGRATION.md` |
| Haber/Sentiment | `transformers` (FinBERT), `vaderSentiment`, `textblob`, `beautifulsoup4`, `newsapi-python` |
| Whale | `web3` + Whale Alert API |
| Frontend | Jinja2 server-render + **Bootstrap 5 (CDN)** + Font Awesome 6 + Chart.js + Socket.IO client + vanilla JS |
| Zamanlama | `schedule` (haftalık fine-tune) |
| Config | `python-dotenv` (`.env`) |
| Mac M1/M2 | `tf_config.py` (TensorFlow'dan **önce** import edilmeli) |
| Container | Docker + docker-compose (postgres + web) + `docker/entrypoint.sh` (bekle→migrate→seed→provision→başlat) |

> **Dokümanda olmayan hiçbir teknolojiyi "kullanılıyor" diye ekleme.** Örn. proje React/Vue/TypeScript/Node/**Prisma** **kullanmaz** (Prisma Node ORM'idir; Python muadili SQLAlchemy+Alembic kullanılır). PostgreSQL/SQLAlchemy/Alembic/Docker **artık gerçek dosyalar olarak vardır** (`packages/database/`, `docker/`, `docker-compose.yml`).

---

## 3. Klasör Yapısı

```
trading/                          # ← GERÇEK proje kökü (git repo burada)
├── web_app.py                    # MONOLİT Flask app (2435 satır, 23 route, aktif SocketIO)
├── main.py                       # MONOLİT CLI uygulaması (interaktif coin analizi)
├── run_dashboard.py              # Monolith dashboard launcher
├── training_scheduler.py         # Haftalık fine-tune scheduler (schedule lib, Pazar 02:00)
│
├── lstm_model.py                 # LSTM fiyat tahmini (CryptoLSTMModel)
├── dqn_trading_model.py          # DQN aksiyon seçimi (9 aksiyon, Double-DQN)
├── hybrid_trading_model.py       # LSTM+DQN+teknik ensemble
├── predictor.py                  # Çok-modelli tahmin orkestratörü (CryptoPricePredictor)
├── data_preprocessor.py          # 20 teknik gösterge + scaling (25 feature)
├── data_fetcher.py               # Binance OHLCV çekme (ccxt)
├── model_cache.py                # Model cache (ModelCache, CachedModelManager)
├── tf_config.py                  # M1/M2 TensorFlow config (TF'den ÖNCE import et)
│
├── news_analyzer.py              # Haber sentiment (FinBERT/VADER, çoklu kaynak)
├── whale_tracker.py              # Whale Alert + on-chain hareket analizi
├── binance_trader.py             # ccxt trading motoru (spot+futures)
├── auto_trader_integration.py    # AI sinyali → trade köprüsü
│
│   # NOT: eski database.py/mssql_database.py/auth.py/system_persistence.py KALDIRILDI
│   # → hepsi packages/database/ (trading_db) paketine taşındı.
│
├── packages/database/            # ← PAYLAŞILAN VERİ KATMANI (kurulabilir "trading-db")
│   ├── trading_db/               #    models_shared/tenant, session, tenancy, repository,
│   │                             #    auth, system_persistence, provisioning, seed_*, migrate
│   │   └── alembic/              #    iki track: versions_shared/ + versions_tenant/
│   └── pyproject.toml
│
├── docker/                       # Dockerfile, entrypoint.sh (migrate+seed+provision), wait_for_db.py
├── docker-compose.yml            # postgres + web servisleri
│
├── templates/                    # MONOLİT Jinja2 template'leri (Flask bunları render eder)
├── static/                       # (pratikte boş; CSS/JS inline)
├── model_cache/                  # Eğitilmiş model artefaktları (.h5/.pkl/.json) — gitignore'lu
│
├── ai-service/                   # ← MİKROSERVİS: FastAPI AI servisi (port 8000)
│   ├── CLAUDE.md                 #    (bu servise özel kurallar)
│   ├── ai_service.py             #    FastAPI app + endpoint'ler
│   ├── comprehensive_trainer.py  #    Async çoklu-model eğitim
│   ├── config.py                 #    Config sınıfı (DİKKAT: ai_service.py bunu import ETMİYOR)
│   ├── run_ai_service.py         #    Launcher
│   └── (predictor.py, *_model.py, ... monolitten KOPYA)
│
├── web-service/                  # ← MİKROSERVİS: Flask web servisi (port config'te 25629)
│   ├── CLAUDE.md                 #    (bu servise + frontend'e özel kurallar)
│   ├── web_service.py            #    Flask app + AIServiceClient (httpx)
│   ├── config.py                 #    WebConfig sınıfı
│   ├── run_web_service.py        #    Launcher
│   ├── *.html + templates/       #    Frontend (Jinja2) — monolit templates'in KOPYASI
│   └── (binance_trader.py monolitten KOPYA; DB/auth artık trading_db paketinden)
│
├── ARCHITECTURE.md               # Sistem mimarisi, veri akışı, diyagramlar
├── DATABASE.md                   # PostgreSQL/SQLAlchemy/Alembic şema + tenancy + migration kuralları
├── DATABASE_MIGRATION.md         # MSSQL/SQLite → PostgreSQL geçiş özeti + manuel kontrol listesi
├── PROJECT_STANDARDS.md          # Kod/naming/security/test standartları
├── DEVELOPMENT_GUIDE.md          # Kurulum, çalıştırma, yeni özellik ekleme adımları
└── README.md / *_README.md       # Mevcut (İngilizce/Türkçe) dokümanlar
```

> **Not:** Bu doküman seti dosya yapısı ve kaynak kod okunarak çıkarılmıştır. Runtime davranışını kesinleştirmek gerekirse ilgili modülü/`.env`'i kontrol et.

---

## 4. Kurulum ve Çalıştırma Komutları

Her uygulamanın (monolith, ai-service, web-service) **kendi `requirements.txt` ve `.env` dosyası** vardır. Veri katmanı ortak `packages/database/` paketindedir.

### En kolay: Docker Compose (postgres + web, otomatik migrate/seed/provision)
```bash
cp .env.example .env          # değerleri düzenle (gerçek secret'ları .env'e)
docker compose up --build     # postgres + web; entrypoint: DB bekle→migrate→seed→provision→başlat
# Web: http://localhost:5055 (host portu WEB_HOST_PORT ile değişir)
```

### Veritabanı (elle, Docker'sız)
```bash
pip install -e packages/database             # paylaşılan trading_db paketi
export DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/app_db
python -m trading_db.migrate shared          # shared şema migration
python -m trading_db.seed_shared             # coin kataloğu + platform admin (idempotent)
python -m trading_db.provisioning default --name Default   # default tenant + şema
# Yeni tenant: python -m trading_db.provisioning acme --admin-user admin_acme --admin-password '***'
```

### Monolith
```bash
# Kök klasörde
python -m venv trading_env && source trading_env/bin/activate
pip install -r requirements.txt && pip install -e packages/database
cp .env.example .env          # SONRA .env'i düzenle (DATABASE_URL + secret)

python main.py                # İnteraktif CLI analizi
python main.py --demo         # BTC otomatik demo
python run_dashboard.py       # Web dashboard (Flask + SocketIO)
```

### Mikroservisler
```bash
# En kolay: kökten
bash start_microservices.sh   # ai-service (8000) → sonra web-service'i başlatır

# Veya ayrı ayrı:
cd ai-service && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && python run_ai_service.py     # → http://localhost:8000 (Swagger: /docs)

cd web-service && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && python run_web_service.py    # → WEB_PORT (config default 25629)
```

### Test / Lint / Typecheck
```bash
python test_ai_service.py             # AI service entegrasyon testi (servis ayakta olmalı)
python test_web_service.py            # Web service entegrasyon testi
python quick_test.py                  # Hızlı LSTM/pipeline dumanı testi
python test_comprehensive_system.py   # Uçtan uca sistem testi
```
> **Lint / format / typecheck kurulu DEĞİL.** Repoda `flake8`, `pylint`, `pyproject.toml`, `mypy`, `black`, `ruff`, `pytest.ini`, `.pre-commit-config.yaml` **yoktur**. Testler `pytest` ile değil, `python <dosya>.py` ile çalışan **gevşek script'lerdir** (çoğu servis ayakta olmayı gerektirir). README'deki `pytest`/`pre-commit` komutları **aspirasyoneldir, kurulu değildir** — dokümanda böyle sun.

---

## 5. Ortam Değişkenleri (Environment Variables)

- Değerleri **asla** dokümana, koda veya log'a yazma. Yalnızca `.env` içinde tut.
- Üç ayrı `.env` vardır: kök (`.env.example` şablonlu), `ai-service/.env`, `web-service/.env`.
- Anahtar grupları: **DB (`DATABASE_URL`, `POSTGRES_HOST/PORT`)**, seed/provision admin (`PLATFORM_ADMIN_*`, `ADMIN_*`, `DEFAULT_TENANT_SLUG`), Binance (`BINANCE_API_KEY/SECRET/TESTNET`), Haber (`NEWSAPI_KEY`), Whale (`WHALE_ALERT_API_KEY`), Flask (`FLASK_SECRET_KEY`/`SECRET_KEY`, `WEB_PORT`), AI service (`AI_SERVICE_URL`, `AI_SERVICE_API_KEY`), model (`LSTM_EPOCHS`, `LSTM_TRAINING_DAYS`, `DQN_EPISODES`). **`MSSQL_*` KALDIRILDI.**
- `.gitignore` `.env`, `*.db`, `*.h5`, `*.pkl`, `*secret*`, `*key*` dosyalarını dışlar — **iyi**. `.env` dosyalarını asla commit etme.

> 🔴 **GÜVENLİK UYARISI:** Kökteki `.env.example` daha önce **gerçek görünen canlı MSSQL kimlik bilgileri** içeriyordu; PostgreSQL geçişinde placeholder ile **temizlendi**. Ancak o kimlik bilgileri git geçmişinde kalmış olabilir — **sızmış kabul edilip rotate edilmeli**. Ayrıca varsa yerel `.env` dosyanızdaki eski MSSQL secret'larını da geçersiz kılın. Bu değerleri hiçbir çıktıya kopyalama.

---

## 6. AI Agent İçin Kesin Kurallar (MUTLAKA uyulmalı)

### 6.1 En kritik: Kod tekrarı (duplication) — düzenleme yapmadan önce OKU
Aynı isimli dosyalar birden fazla yerde **fiziksel kopya** olarak durur. Bir kopyayı değiştirmek diğerlerini **güncellemez**.

| Dosya | Kopyalar | Durum |
|---|---|---|
| `database.py` / `mssql_database.py` / `auth.py` / `system_persistence.py` | — | ✅ **DEDUP EDİLDİ** → `packages/database/` (tek kaynak `trading_db`). Artık kopya YOK. |
| `binance_trader.py` | kök == `web-service/` | **BİREBİR AYNI** (henüz dedup edilmedi) |
| `predictor.py` | kök == `ai-service/` | **BİREBİR AYNI** |
| `data_preprocessor.py` | kök == `ai-service/` | **BİREBİR AYNI** |
| `model_cache.py` | kök == `ai-service/` | **BİREBİR AYNI** |
| `news_analyzer.py` | kök == `ai-service/` | **BİREBİR AYNI** |
| `dqn_trading_model.py` | kök vs `ai-service/` | **FARKLI** (mikroservis versiyonu değiştirilmiş) |
| `lstm_model.py` | kök vs `ai-service/` | **FARKLI** |

**Kural:** DB/auth katmanı artık **tek yerde** (`packages/database/trading_db/`) — buradan düzenle, `pip install -e packages/database` ile kurulur. Kalan kopyalı dosyalar (binance_trader, model dosyaları) için: doğru kopyayı düzenle ve senkron gerekliliğini açıkça belirt; kendi kararınla büyük birleştirme yapma.

Aynı şey template'ler için de geçerli: monolith `templates/*.html` (Flask'ın render ettiği) vs `web-service/*.html` ve `web-service/templates/*.html`. Yanlış kopyayı düzenlersen UI'a yansımaz.

### 6.2 Değiştirilmemesi gereken sözleşmeler (contracts)
- **AI Service API contract**: endpoint path'leri (`/analyze`, `/train`, `/models/{coin}`, `/health`, `/training/status`) ve Pydantic modelleri (`CoinAnalysisRequest`, `TrainingRequest`, `PredictionResponse`, `TrainingResponse`). `web-service/web_service.py:AIServiceClient` bunlara birebir bağlı.
- **Jinja2 template contract**: `analyze_coin.html` tamamen `analysis.*` dict ağacına bağımlı (`analysis.technical_analysis.rsi.value`, `.confidence`, `predicted_price`, `price_change_percent` ...). Backend bu dict şemasını korumazsa template render hatası verir.
- **SocketIO event adları** (monolith): `analysis_complete`, `analysis_error`, `analysis_update`, `connected`, `dashboard_update`, `request_update` — client/server birebir eşleşmeli.
- **Ensemble/DQN semantiği**: `recommendation` enum'u `STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL`; DQN `action` kodları `0=HOLD, 1-4=BUY_%, 5-8=SELL_%`. Bu semantiği tüketen kodu kırma.
- **Response formatını değiştirme**: mevcut (tutarsız da olsa) `{success, error, ...}` / `jsonify(result)` desenini koru; downstream kod ve frontend buna bağlı.

### 6.3 ML / model katmanı — sessiz bozulma riski yüksek
- **`tf_config` import sırası kutsaldır**: her model dosyasının başında `from tf_config import get_tensorflow` gelir ve TensorFlow'dan **önce** çalışır. Sırayı bozma → M1/M2'de Metal çökmesi.
- **Scaler fit vs transform**: ilk eğitim `fit_scaler=True`; cache'den yükleme / incremental / tahmin `fit_scaler=False`. Karıştırırsan tahminler **sessizce yanlış** ölçeklenir. Scaler daima modelle birlikte `_scaler.pkl` olarak saklanır.
- **Feature sırası**: OHLCV ilk 5 sütun, hedef = **close (index 3)**. Toplam **25 feature** (5 OHLCV + 20 teknik gösterge). Feature sırasını/sayısını değiştirmek tüm cache'i geçersiz kılar ve `inverse_transform`'u bozar.
- **`directional_accuracy` custom metriği** `.h5` yüklemeden önce Keras'a register edilmeli — kaldırma.
- **Model cache adlandırması**: iki farklı şema bir arada (`ModelCache`: `models/{SYMBOL}_{md5hash8}.h5`; diskte fiili: `{tip}_{coin}_model.h5`). Herhangi birini değiştirmeden önce ikisini de anla.

### 6.4 Güvenlik ve veri
- **Auth ve permission kontrollerini atlama.** Monolitte `@login_required` korumalı route'ları korumasız yapma. (Not: bazı `/api/*` route'ları zaten korumasız — bunları "örnek pattern" sanıp yaymaya çalışma; PROJECT_STANDARDS'a göre bunlar **düzeltilmesi gereken** açıklardır.)
- **Production verisini bozabilecek işlem önerme.** (`add_coin` veri kaybı bug'ı PostgreSQL geçişinde düzeltildi — artık upsert.)
- **Migration**: Artık **Alembic** (iki track: shared + tenant). `python -m trading_db.migrate shared|all-tenants`. Prod'da yalnız `upgrade`, autogenerate/db-push **yok**. Şema değiştirmeden önce mutlaka **DATABASE.md** oku ve modelleri (`models_shared/tenant.py`) + migration'ı birlikte güncelle.
- **Multi-tenancy (schema-per-tenant)**: Tenant tablolarına erişim aktif tenant bağlamı (`search_path`) gerektirir; bağlam yoksa fail-safe (yazma atlanır). **Tenant izolasyonunu bozma**; `before_request` tenant bağlama mantığını kaldırma. Ortak (shared) vs tenant tablo bölümüne uy (bkz. DATABASE.md §2).
- **Tenant izolasyonu**: Bu projede multi-tenant yapı **yoktur**; tek kurulum/tek kullanıcı-tabanlı. Uydurmalar ekleme.

### 6.5 Genel çalışma disiplini
- Mevcut mimariyi bozmadan geliştir; büyük refactor'dan önce bağımlılıkları analiz et ve kullanıcıya danış.
- **Kullanılmıyor gibi görünen kodu hemen silme** — bu projede çok sayıda yarım/placeholder entegrasyon var (bkz. §7); bunlar bilinçli iskele olabilir.
- **Yeni dependency eklemeden önce** mevcut paketlerle çözüm ara (proje zaten ağır bağımlılık taşıyor).
- Kod üretmeden önce **ilgili klasördeki benzer örneği** oku ve aynı pattern'i izle (naming, error handling, log stili).
- Belirsiz alanda **varsayım yapma**; dokümana/koda açık `# NOT:` bırak veya kullanıcıya sor.
- Emoji'li Türkçe log/print stili projenin mevcut konvansiyonudur — yeni kodda buna uy (PROJECT_STANDARDS.md).

---

## 7. Bilinen Yarım / Placeholder / Buglı Yapılar (körlemesine "düzeltme")

Bunlar mevcut durumdur; birini değiştireceksen etkisini anla ve kullanıcıya bildir:

- `ai-service` `/analyze` gerçek `CryptoPricePredictor`'ı **kullanmıyor** — iç helper'lar placeholder değer döndürüyor (`predicted_price = close*1.02`, sabit `confidence`).
- `ai-service` background comprehensive eğitim, var olmayan `train_coin_async` metodunu çağırıyor → `AttributeError` ile sessizce düşer.
- `ai-service` `/training/status` her zaman **boş** döner (gerçek durum takibi yok). AI service'te **auth yok**, CORS `*`, `reload=True`, ve `config.py` import edilmediği için oradaki env ayarları etkisiz.
- `web-service` SocketIO tanımlı ama **hiçbir handler/emit yok**; portfolio/dashboard değerleri **hardcoded 0** (BinanceTrader bağlı değil); `SystemPersistence` metod adı uyuşmazlığı nedeniyle `tracked_coins.json`'a düşüyor.
- `settings.html` formları backend'e **yazmıyor** — sadece `alert(...)` gösteriyor (mock).
- MSSQL katmanı **kısmi**: `trades`/`positions`/`analysis_results` gerçek tablolara değil `system_state`'e JSON olarak yazıyor; birkaç metod stub; iki tutarsız şema kaynağı var (bkz. DATABASE.md).
- Haber/whale analizörleri API key yoksa **mock/demo veri** üretir → production'da sessizce sahte sinyal riski.

---

## 8. Alt Rehberler

| Nerede çalışıyorsan | Oku |
|---|---|
| Genel / mimari | `ARCHITECTURE.md` |
| AI service (FastAPI, modeller) | `ai-service/CLAUDE.md` |
| Web service + frontend (Flask, Jinja2) | `web-service/CLAUDE.md` |
| Veritabanı / şema / migration | `DATABASE.md` |
| Kod/naming/security/test standartları | `PROJECT_STANDARDS.md` |
| Kurulum / yeni özellik ekleme adımları | `DEVELOPMENT_GUIDE.md` |

---

## 9. Deployment Notları

- **Docker deployment artık var**: `docker/Dockerfile` + `docker-compose.yml` (postgres + web) + `docker/entrypoint.sh` (DB bekle → shared migrate → tenant migrate → seed → default tenant provision → Flask). `docker compose up --build` ile ayağa kalkar. CI/CD ve nginx conf henüz yok. ai-service (ML-ağır, DB'siz) compose'da opsiyonel/yorumlu. Prod'da Flask dev sunucusu yerine gunicorn+eventlet önerilir.
- Mevcut çalıştırma: `bash start_microservices.sh` (nohup + arka plan + log dosyaları) veya doğrudan launcher'lar.
- Üretime almadan önce mutlaka kapatılması gerekenler: `DEBUG`/`reload=True`, CORS `*`, açık `SECRET_KEY` fallback'i, korumasız `/api/*` route'ları, log'a şifre yazımı (bkz. `PROJECT_STANDARDS.md` Security Checklist).
- **Risk uyarısı**: Bu bir gerçek-para trading sistemidir. `BINANCE_TESTNET` varsayılanı `True`'dur; canlıya almadan önce risk yönetimi ayarlarını (`default_risk_percent=2.0`, `max_leverage=10`) gözden geçir.
