# CLAUDE.md — Web Service (Flask + Frontend)

> Bu klasör, web dashboard mikroservisidir (Flask + Jinja2 frontend). AI işleme yoktur; `ai-service`'e HTTP ile bağlanır. Önce kök `../CLAUDE.md` ve `../ARCHITECTURE.md`'yi oku. Bu dosya hem **backend** (§1-8) hem **frontend** (§9-14) kurallarını içerir.

---

## 1. Sorumluluk

Kullanıcı arayüzü, kimlik doğrulama, session, portfolio görünümü ve `ai-service` çağrıları. Ağır iş `ai-service`'e delege edilir.

## 2. Framework ve Server Yapısı

- **Flask** app: `web_service.py` → `app = Flask(__name__)`, `secret_key` env (`SECRET_KEY`, hardcoded fallback'li).
- **Flask-SocketIO**: `socketio = SocketIO(app, cors_allowed_origins="*")`. ⚠️ Tanımlı ama **hiçbir handler/emit yok** (real-time iskele boş). Uygulama `socketio.run(...)` ile başlar.
- **Blueprint YOK** — tüm route'lar tek dosyada `@app.route`.
- **Global nesneler**: `database` (`trading_db.TradingDatabase` — PostgreSQL, tenant-aware), `binance_trader` (veya `DummyBinanceTrader`), `system_persistence`, `auth_manager` (`trading_db.AuthManager`), `ai_client`, `coin_monitor`.
- **Multi-tenancy**: `before_request` → `set_current_tenant(current_user.tenant_schema)`, `teardown_request` → `clear_current_tenant`. Provisioning: `POST /api/tenants` (platform_admin).
- **Launcher**: `run_web_service.py` → `socketio.run(app, host, port, debug=WEB_DEBUG)`.
- **Config**: `config.py:WebConfig`. ⚠️ **Varsayılan `WEB_PORT=25629`** (README'deki 5000 değil); `.env`'deki `WEB_PORT` gerçek değeri belirler. `WEB_DEBUG` default `True`.

## 3. Route'lar

**HTML sayfaları**: `/` (→dashboard), `/dashboard` 🔒, `/analyze_coin` 🔒, `/portfolio` 🔒, `/settings` 🔒, `/login` (GET/POST), `/logout` 🔒. (🔒 = `@login_required`)

**JSON API** (`/api/*`, hepsi 🔒): `POST /api/analyze_coin`, `POST /api/train_coin`, `GET /api/ai_service_status`, `GET /api/model_status/<coin_symbol>`.

**Error handler**: `404.html`, `500.html`.

> Not: Bu servisteki `/portfolio` ve `/dashboard` `binance_trader`'ı kullanmıyor — portfolio değerleri **hardcoded 0** (BinanceTrader entegrasyonu yarım). (Monolitteki `web_app.py` route seti ayrıdır ve daha zengindir; karıştırma.)

## 4. AI Service HTTP Client (contract)

- **Sınıf**: `AIServiceClient` (global `ai_client`), `httpx.AsyncClient`.
- **Base URL**: `AI_SERVICE_URL` (default `http://localhost:8000`). Auth: `AI_SERVICE_API_KEY` varsa `Authorization: Bearer` header.
- **Metod → endpoint**: `analyze_coin()`→`POST /analyze` (60s), `train_model()`→`POST /train` (300s), `get_model_status()`→`GET /models/{coin}` (30s), `health_check()`→`GET /health` (10s).
- **Sync↔async köprüsü**: Flask route'ları her istekte `asyncio.new_event_loop()` + `run_until_complete` + `loop.close()`. ⚠️ **Bu köprüyü kaldırma** — tüm API çöker.
- **Retry YOK**; tek deneme. Non-200 → `{"success": False, "error": ...}`.

## 5. Authentication (`trading_db.auth`)

- **Flask-Login** (session cookie). **JWT/bcrypt DEĞİL** — `hashlib.pbkdf2_hmac('sha256', ..., 100000)` + `secrets` salt; format `salt(32 hex)+hash` (geçişte korundu).
- **Kullanıcı deposu**: shared `users` tablosu (Alembic yönetir) + `tenant_id` FK. Login → `User.tenant_schema` doldurulur → `before_request` aktif tenant'ı set eder. Platform admin `tenant_id=NULL`.
- **Akış**: `login.html` POST → `authenticate_user` → `verify_password` → `login_user`. `@login_required` korumalı sayfalar.
- ⚠️ **Riskler (koruma, çoğaltma)**: env yoksa varsayılan admin auto-create + **şifre log'a yazılıyor**; DB yoksa `SimpleAuthManager` **hash'siz düz karşılaştırma**; `SECRET_KEY` hardcoded fallback.
- **auth artık `packages/database/trading_db/auth.py`'dedir** (tek kaynak; monolit ile ortak). Kopya yok.

## 6. Response Formatı

- Gevşek zarf: başarı → `jsonify(result)`; hata → `{'success': False, 'error': msg}` + HTTP kodu (400 eksik, 415 JSON değil, 500 exception). `health`/`model_status` farklı zarf (`status`/`error`) kullanıyor (tutarsız).
- `_format_for_web()`: AI çıktısını UI şemasına (`multi_model_results` + geriye dönük düz alanlar) çevirir. **Template'ler bu alanlara bağlı olabilir — koru.**

## 7. Background / Persistence

- **Gerçek monitoring thread / scheduler YOK.** Her şey HTTP isteğiyle tetiklenir. `WebCoinMonitor` yalnız `tracked_coins` listesini yükler/günceller.
- ⚠️ `WebCoinMonitor`, `system_persistence`'ta **var olmayan** `load_state`/`save_state` metodlarını çağırıyor → `hasattr` False → `tracked_coins.json` dosyasına düşüyor (persistence katmanı fiilen bypass). Bunu "düzeltmeden" önce doğru metod adını netleştir.

## 8. Backend'te Kesinlikle Yapılmaması Gerekenler

- ❌ `AIServiceClient` endpoint eşleşmelerini (path/timeout) veya sync↔async köprü desenini bozma.
- ❌ `before_request`/`teardown_request` tenant bağlama mantığını kaldırma (tenant izolasyonu buna bağlı).
- ❌ `password_hash` salt+hash formatını değiştirme (mevcut kullanıcılar login olamaz).
- ❌ `_format_for_web` çıktı alanlarını sessizce değiştirme (template bağımlı).
- ❌ `@login_required`'ı kaldırma; korumasızlığı çoğaltma.
- ❌ DB/auth katmanını `web-service/` içinde arama — artık `packages/database/trading_db/`'dedir. (`binance_trader.py` hâlâ monolitle kopyadır — dikkat.)

---

## 9. Frontend — Genel

- **Yaklaşım**: Jinja2 **server-side render** + minimal inline vanilla JS. SPA/build step **yok**.
- **Kütüphaneler (hepsi CDN, head'de)**: Bootstrap 5 (dashboard 5.1.3, login 5.3.0 — sürüm tutarsız), Font Awesome 6, **Chart.js** (yalnız `portfolio.html`), **Socket.IO client** (yalnız `dashboard.html`). jQuery / axios / TradingView / Plotly **yok**.
- ⚠️ **İki template kopyası**: monolit `../templates/*.html` (Flask'ın render ettiği) vs bu klasördeki `*.html` ve `templates/*.html`. **Doğru kopyayı düzenle**, yoksa UI'a yansımaz. `css/` ve `js/` klasörleri **boş** — tüm CSS/JS HTML içinde inline.

## 10. Sayfalar

| Sayfa | İşlev | Özellik |
|---|---|---|
| `dashboard.html` | Ana panel, coin ekle/izle | SocketIO canlı durum → Bootstrap **toast** (`socket = io()`) |
| `portfolio.html` | Portfolio + pozisyon + P&L | **2 Chart.js** grafiği; `fetch('/api/portfolio')`, `fetch('/api/close_position')` |
| `analyze_coin.html` | Analiz raporu (en büyük) | **9 Bootstrap nav-tab**; tamamen Jinja2 `analysis.*` render (fetch yok) |
| `settings.html` | API/model/monitoring ayarları | ⚠️ Formlar backend'e **yazmıyor** — `preventDefault()` + `alert()` (mock) |
| `login.html` | Giriş | Düz form POST → Flask-Login |

## 11. Stil ve Tasarım Dili

- Bootstrap utility class + her sayfada **tek inline `<style>`**. Harici .css **yok**.
- **Tema tek (açık) — dark mode YOK**, tema switcher yok.
- Görsel dil: mor→pembe gradient (`#667eea → #764ba2`), yuvarlak köşe (border-radius 15-20px), glassmorphism (`backdrop-filter: blur`), yüzen animasyonlu şekiller, gradient text-clip başlıklar.
- İkonlar: Font Awesome (`fas fa-*`). Renk sınıfları: Bootstrap + birkaç custom (`price-up`/`price-down`).

## 12. API Client Kullanımı (frontend → backend)

İki desen:
1. **Form POST / link GET** (Jinja `url_for`): `add_coin`, `remove_coin`, `login`, monitoring başlat/durdur.
2. **`fetch(url).then(r=>r.json())`** promise zinciri (portfolio, close_position). `async/await` yaygın değil, `XMLHttpRequest`/axios yok.

Canlı güncelleme: yalnız dashboard'da `socket.on('analysis_complete'|'analysis_error'|'analysis_update'|'dashboard_update')`. **Event adlarını backend ile senkron tut.**

## 13. Ortak Bileşenler ve Yeni Sayfa Ekleme

- Bileşenler: navbar, card (`card-header`/`card-title`), nav-tabs + tab-pane, toast, badge, table, form.
- **Yeni sayfa**: mevcut bir `.html`'i şablon al (CDN head + inline style/script kopyala) → `render_template` route ekle (gerekiyorsa `@login_required`) → Türkçe UI metni → `url_for` linkleri. Grafik gerekiyorsa Chart.js CDN'i portfolio örneğinden al. Detay: `../DEVELOPMENT_GUIDE.md §12`.

## 14. Frontend'te Kesinlikle Yapılmaması Gerekenler

- ❌ **Jinja2 `analysis.*` dict contract'ını bozma** (`analysis.technical_analysis.rsi.value`, `.bollinger_bands.upper/lower`, `.macd.trend`, `analysis.confidence`, `predicted_price`, `price_change_percent`). Eksik alan → render hatası.
- ❌ `url_for(...)` endpoint adlarına bağlı form/link'leri, fonksiyon adını değiştirerek kırma.
- ❌ Bootstrap tab `data-bs-toggle`/`data-bs-target` id eşleşmelerini (analyze_coin 9 sekme) bozma.
- ❌ SocketIO event adlarını veya `data.coin`/`data.message` alanlarını tek taraflı değiştirme.
- ❌ `login.html` form alan adlarını (`username`/`password`) değiştirme (`request.form.get` ile eşleşiyor).
- ❌ Yanlış template kopyasını düzenleyip "yansımadı" durumuna düşme.
- ❌ Dark mode / yeni CSS framework'ü mevcut sistemi bozacak şekilde dayatma; mevcut Bootstrap + inline stile uy.
- ⚠️ `settings.html` formlarının şu an backend'e yazmadığını bil (mock `alert`); "çalışıyor" varsayma.

## 15. Bu Serviste Çalışırken Önce Oku

`web_service.py` (route + AIServiceClient), `auth.py` (auth akışı), ilgili `*.html`. Kök: `../CLAUDE.md §6`, `../PROJECT_STANDARDS.md §10` (UI), `../DEVELOPMENT_GUIDE.md §11-12`.
