# ARCHITECTURE.md — Sistem Mimarisi

> Bu doküman, kaynak kodun okunmasıyla çıkarılmıştır. Runtime davranışını kesinleştirmek için ilgili modül/`.env` kontrol edilmelidir.

---

## 1. Genel Bakış

Sistem, **kripto para analizi ve otomatik ticareti** için LSTM (fiyat tahmini) + DQN (aksiyon seçimi) + teknik analiz + haber sentiment + whale tracking'i birleştiren hibrit bir AI platformudur.

Repoda **iki mimari nesil bir arada** bulunur:

- **Monolith** (repo kökü): Tek Flask uygulaması (`web_app.py`) + CLI (`main.py`). Tüm katmanlar tek process.
- **Mikroservisler**: `ai-service` (FastAPI) + `web-service` (Flask), HTTP/REST ile haberleşir.

İkisi aynı iş mantığını (aynı model/DB/trading dosyalarının **kopyalarını**) paylaşır. Yeni geliştirmelerde mikroservis mimarisi tercih edilir; monolith referans/orijinal olarak durur.

---

## 2. Sistem Bileşenleri

### Monolith (kök)
```
Kullanıcı (tarayıcı / CLI)
        │
        ▼
web_app.py (Flask + SocketIO, 23 route)  ── main.py (interaktif CLI, ayrı giriş noktası)
        │
        ├── predictor.py (CryptoPricePredictor) ── lstm_model / dqn_trading_model / hybrid_trading_model
        │        └── data_preprocessor.py (20 teknik gösterge, MinMaxScaler)
        │        └── model_cache.py (eğit/cache/incremental)
        ├── news_analyzer.py (FinBERT/VADER)   ── whale_tracker.py (Whale Alert / web3)
        ├── binance_trader.py (ccxt spot+futures) ── auto_trader_integration.py
        ├── training_scheduler.py (haftalık fine-tune)
        └── trading_db paketi (PostgreSQL/SQLAlchemy: repository + auth + system_persistence + provisioning)
```

### Mikroservis mimarisi
```
        Kullanıcı (tarayıcı)
              │  HTTP + WebSocket
              ▼
┌───────────────────────────────┐        HTTP/REST (httpx)        ┌───────────────────────────────┐
│   WEB SERVICE (Flask)         │ ──────────────────────────────▶ │   AI SERVICE (FastAPI)        │
│   web-service/  port≈25629*   │  POST /analyze, /train          │   ai-service/   port 8000     │
│                               │  GET  /models/{coin}, /health   │                               │
│ • Jinja2 dashboard + auth     │ ◀────────────────────────────── │ • LSTM / DQN / Hybrid modelleri│
│ • Flask-Login (session)       │        JSON yanıt               │ • comprehensive_trainer (async)│
│ • AIServiceClient (httpx)     │                                 │ • news / whale analizi         │
│ • binance_trader (bağlı değil)│                                 │ • model_cache                  │
│ • PostgreSQL (trading_db)     │                                 │ • Pydantic request/response    │
└───────────────────────────────┘                                 └───────────────────────────────┘
        │                                                                   │
        ▼                                                                   ▼
   PostgreSQL (shared + tenant_<slug> şemaları)                      model_cache/ (.h5 / .pkl / .json)
```
\* `web-service/config.py` varsayılan `WEB_PORT=25629`'dur; README/MICROSERVICES_README 5000 der. `.env`'deki `WEB_PORT` gerçek değeri belirler — **doğrulanmalı**.

### Katmanlı görünüm (mantıksal)
```
User
  ↓
Frontend (Jinja2 + Bootstrap 5 + vanilla JS, SocketIO/fetch)
  ↓
Web Backend (Flask: routing, auth, session)   ──HTTP──▶   AI Backend (FastAPI: analyze/train)
  ↓                                                              ↓
Service Layer (predictor, trainer, news, whale, binance_trader)
  ↓
Data Layer (trading_db: PostgreSQL/SQLAlchemy, schema-per-tenant, model_cache, .env)
  ↓
External: Binance (ccxt) · NewsAPI/CoinDesk/Reddit · Whale Alert · PostgreSQL
```

---

## 3. Auth Akışı

- **Mekanizma**: Flask-Login (server-side **session cookie**). `trading_db.auth` içinde `AuthManager` + `User(UserMixin)`.
- **Şifre**: `hashlib.pbkdf2_hmac('sha256', ..., 100000)` + `secrets.token_hex(16)` salt. Format = `salt(32 hex) + hash` (geçişte korundu; bcrypt/JWT DEĞİL).
- **Kullanıcı deposu**: **shared `users`** tablosu (Alembic yönetir) + `tenant_id` FK → tenants. Platform admin `tenant_id=NULL`.
- **Akış**: `login.html` form POST → `authenticate_user` → `verify_password` → `login_user` → session'a `tenant_schema`. Her istekte `before_request` aktif tenant'ı (`search_path`) set eder → kullanıcı yalnız kendi tenant verisini görür. Çıkış: `logout()`.
- **Not**: AI service'te auth yoktur (DB'siz). Platform admin `POST /api/tenants` ile yeni tenant provision eder.

---

## 4. Veri Akışı (Analiz İsteği)

```
1. Kullanıcı dashboard'da coin ekler/analiz ister
2. web-service: POST /api/analyze_coin (login gerekli)
3. AIServiceClient.analyze_coin() → httpx POST http://ai-service:8000/analyze
4. ai-service: data_fetcher (Binance OHLCV 4h) → data_preprocessor (25 feature) →
   predictor/model_cache (LSTM→DQN→Hybrid) → ensemble önerisi + (opsiyonel) news/whale
5. ai-service → PredictionResponse (JSON: success, predictions, technical/news/whale_analysis)
6. web-service: _format_for_web() ile UI şemasına dönüştürür → jsonify
7. Frontend: analiz sonucunu render eder (analyze_coin.html: 9 sekme, Jinja2 `analysis.*`)
8. (Monolith yolunda) sonuç DB'ye (analysis_results / system_state) ve SocketIO ile canlı UI'a
```

> **Önemli**: Mevcut `ai-service` `/analyze` gerçek predictor'ı tam bağlamamıştır (placeholder değerler dönebilir). Monolith `web_app.py` yolunda predictor tam çalışır.

## 5. Eğitim Akışı

```
Senkron (tek model):   POST /train (training_type != comprehensive) → _single_model_training → yanıt
Async (comprehensive): POST /train (comprehensive) → BackgroundTasks → comprehensive_trainer
                       (LSTM+DQN+Hybrid, ThreadPoolExecutor ile bloklamayan TF çağrıları)
Zamanlanmış:           training_scheduler.py → her Pazar 02:00 → tracked_coins için fine-tune
                       (mevcut model varsa is_fine_tune=True, yoksa sıfırdan)
```
Model cache: `model_cache.py` config-hash ile `cached` / `new` / `incremental` kararı verir (yaş > 7 gün veya accuracy < 0.85 → yeniden eğit).

---

## 6. API Akışı (Servisler arası sözleşme)

`web-service` → `ai-service` çağrıları (`AIServiceClient`):

| Web metodu | AI endpoint | Timeout |
|---|---|---|
| `analyze_coin()` | `POST /analyze` | 60s |
| `train_model()` | `POST /train` | 300s |
| `get_model_status()` | `GET /models/{coin}` | 30s |
| `health_check()` | `GET /health` | 10s |

Flask senkron dünyası ile httpx async dünyası, her istekte `asyncio.new_event_loop()` + `run_until_complete` ile köprülenir. Bu köprü **kritik**tir; kaldırılırsa API çöker.

---

## 7. Deployment Akışı

- **Mevcut**: `start_microservices.sh` → ai-service'i `nohup` ile arka planda başlat, `/health` bekle, sonra web-service'i başlat. Loglar `ai-service/ai_service.log`, `web-service/web_service.log`.
- **Hazır CI/CD, çalışan Docker, nginx conf YOK.** `MICROSERVICES_README.md` içindeki Docker/nginx blokları örnek/öneri metnidir, repoda dosya yoktur.
- Ayrı makinelerde çalıştırma desteklenir: `web-service` `AI_SERVICE_URL` env'i ile uzak ai-service'e bağlanabilir.

---

## 8. Entegrasyonlar (dış servisler)

| Entegrasyon | Modül | Not |
|---|---|---|
| Binance (fiyat + trade) | `data_fetcher.py`, `binance_trader.py` (ccxt) | Spot + futures; `BINANCE_TESTNET` default True |
| NewsAPI / CoinDesk / Reddit | `news_analyzer.py` | FinBERT + VADER + TextBlob; key yoksa mock veri |
| Whale Alert / on-chain | `whale_tracker.py` (web3) | Bilinen borsa cüzdan adresleri hardcoded; key yoksa demo veri |
| PostgreSQL | `trading_db` (SQLAlchemy/psycopg) | `.env` `DATABASE_URL`; schema-per-tenant |

---

## 9. Kritik Modüller ve Modüller Arası Bağımlılıklar

- **`tf_config.py`**: Tüm model modüllerinin **en tepe bağımlılığı**. TensorFlow'dan önce import edilir (M1/M2 Metal ayarları). Bu sıra bozulamaz.
- **`data_preprocessor.py`**: LSTM/DQN/Hybrid'in ortak feature üreticisi. Feature sırası/sayısı (25) tüm modelleri ve cache'i bağlar.
- **`model_cache.py`**: Eğitim maliyetini yöneten merkez. `predictor.py`, `main.py`, `comprehensive_trainer.py`, `training_scheduler.py` buna bağlı.
- **`trading_db` (paylaşılan paket)**: Tek PostgreSQL veri katmanı (repository/auth/persistence/provisioning). Monolit + web-service ortak kullanır. Schema-per-tenant izolasyon (bkz. DATABASE.md).
- **`predictor.py` (CryptoPricePredictor)**: LSTM→DQN→Hybrid orkestratörü + ensemble. LSTM ilk eğitimde başarısızsa DQN/Hybrid atlanır (bilinçli iş kuralı).
- **`AIServiceClient` (web_service.py)**: web ↔ ai köprüsü; AI endpoint contract'ına bağlı.

## 10. Monorepo / Workspace Yapısı

Üç bağımsız Python uygulaması (monolith, ai-service, web-service) + **bir paylaşılan kurulabilir paket** `packages/database/` (`trading_db`). Veri katmanı (DB/auth/persistence/provisioning) artık tek kaynakta (dedup edildi); `pip install -e packages/database` ile kurulur. Kalan bazı dosyalar (binance_trader, model dosyaları) hâlâ kopyalıdır (bkz. `CLAUDE.md §6.1`). Import stili yine `sys.path` tabanlı flat.
