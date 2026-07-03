# PROJECT_STANDARDS.md — Proje Standartları

> Bu standartlar, mevcut koddan **çıkarılmıştır** (dayatılan yeni bir stil değildir). Yeni kod, mevcut komşu dosyalarla tutarlı olmalıdır.

---

## 1. Genel Kod Standartları (Python)

- **Dil**: Python. Monolith `requirements.txt` 3.9 sürümlerine sabitli (`==`); mikroservisler 3.13 uyumlu esnek sürümler (`>=`) kullanır.
- **Stil**: Fiili olarak PEP 8'e yakın ama **otomatik enforce edilmiyor** (lint/format aracı kurulu değil). 4 boşluk girinti, snake_case.
- **Type hints**: Kısmen kullanılıyor (özellikle imzalarda `Optional`, `Dict`, `List`). Yeni kodda mevcut modülün yoğunluğuna uy; tip eklemek serbest ama zorunlu değil.
- **Docstring/yorum**: Türkçe + İngilizce karışık, emoji yoğun. Mevcut stile uy.
- **Import deseni**: `sys.path.append(...)` ile flat import (paket yapısı yok). Model dosyalarında `from tf_config import get_tensorflow` **en başta** (TF'den önce).

---

## 2. Naming Convention

| Öğe | Kural | Örnek |
|---|---|---|
| Dosya | snake_case | `binance_trader.py`, `data_preprocessor.py` |
| Sınıf | PascalCase, alan-öneki | `CryptoLSTMModel`, `DQNTradingModel`, `HybridTradingModel`, `TradingDatabase`, `AuthManager`, `AIServiceClient` |
| Fonksiyon/metod | snake_case | `predict_next_price`, `get_active_coins` |
| Private metod | `_` öneki | `_get_state`, `_train_new_model`, `_format_for_web` |
| Async metod | `_async` son eki (varyantı) | `_analyze_lstm_async`; sync eşi `..._sync` |
| Flask API handler | `api_` öneki | `api_analyze_coin`, `api_portfolio_summary` |
| Route path | snake_case | `/add_coin`, `/api/close_position` |
| SocketIO event | snake_case | `analysis_complete`, `dashboard_update` |
| DB tablo/kolon | snake_case | `analysis_results`, `coin_symbol` |
| Feature adı | snake_case; Yigit göstergeleri `yigit_` | `sma_7`, `rsi`, `yigit_atr_stop` |
| JS fonksiyon | camelCase | `closePosition`, `togglePassword` |
| HTML id | camelCase | `apiSettingsForm`, `portfolioChart` |
| Model ID (cache) | `{SYMBOL}_{md5hash8}` | `BTC_USDT_a1b2c3d4` |
| Ensemble öneri | UPPER_SNAKE enum | `STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL` |
| DQN aksiyon | `HOLD` / `BUY_{25..100}%` / `SELL_{...}%` | `BUY_50%` |

**Dil kuralı**: Kod (isimler, log) İngilizce; **kullanıcıya görünen UI/flash mesajları Türkçe**. Bu ayrımı koru.

---

## 3. Dosya / Klasör İsimlendirme

- Yeni modül: snake_case, tek sorumluluk (örn. `whale_tracker.py`).
- Test dosyaları: `test_*.py` (kökte, ayrı `tests/` klasörü yok).
- Model artefaktları: `{id}.h5`, `{id}_weights.h5`, `{id}.json` (metadata), `{id}.pkl` (preprocessor), `{id}_scaler.pkl` (scaler); DQN: `{base}_params.json`, `{base}_training_history.pkl`; Hybrid: `{base}_lstm.h5`, `{base}_dqn.h5`, `{base}_hybrid_metadata.json`.
- Doküman: UPPER_SNAKE `.md` (`ARCHITECTURE.md`, `*_README.md`).

---

## 4. Commit / Branch Standardı (öneri)

Mevcut git geçmişi tek `main` dalı ve serbest formatlı (Türkçe+İngilizce karışık, bazen tek kelime) commit'ler içerir. İyileştirme önerisi (zorunlu değil, mevcut geçmişe saygı göster):
- **Branch**: `feature/<konu>`, `fix/<konu>`, `chore/<konu>`. `main`'e doğrudan büyük değişiklik pushlama.
- **Commit**: Kısa, açıklayıcı, tercihen tek dilde. Öneri: `type: özet` (ör. `fix: MSSQL analysis_history LIKE deseni düzeltildi`).
- Sırlar (`.env`, `*.h5`, `*.db`) `.gitignore` ile dışlanmıştır — **asla** commit etme.

---

## 5. Error Handling Standardı

- **Servis/DB katmanı**: try/except ile hatayı yut, güvenli varsayılan döndür (`None`/`[]`/`False`/`{success:False, error:...}`) ve logla. Mevcut baskın desen budur.
- **FastAPI (ai-service)**: İki desen karışık — bazı durumlar `HTTPException` (400/404/500), bazı iş hataları `success=False` + **HTTP 200**. Yeni endpoint'te **tutarlılık için** gerçek hataları `HTTPException` ile ver; iş sonucu "başarısız" ise `success:False` kullan ve bunu açıkça belirt.
- **Flask (web)**: `{'success': False, 'error': msg}` + uygun HTTP kodu (400 eksik veri, 415 JSON değil, 500 exception). Yeni JSON endpoint'lerde bu zarfı kullan.
- Kullanıcıya gösterilen hata mesajları **Türkçe**.

---

## 6. Logging Standardı

- `logging` modülü + emoji'li Türkçe `print`/log karışık kullanılıyor (`🚀 başlatılıyor`, `✅ başarılı`, `⚠️ uyarı`, `❌ hata`).
- `training_scheduler.py` format: `%(asctime)s - %(levelname)s - %(message)s`, `FileHandler` + `StreamHandler`.
- **Sırları loglama**: Mevcut kodda birkaç ihlal var (admin şifresi log'a yazılıyor `auth.py`; API key ilk karakterleri loglanıyor `news_analyzer.py`). **Yeni kodda bunu yapma**; token/şifre/key loglama.

---

## 7. Python / JS Kuralları

**Python**
- `tf_config` import sırasını koru; TF'yi doğrudan en tepede import etme.
- Yeni dependency eklemeden önce mevcut paketlerle çöz; eklersen doğru `requirements.txt`'e ekle (monolith / ai-service / web-service ayrı).
- Async: ai-service'te `async def` + `httpx.AsyncClient`; Flask tarafında sync↔async köprüsü (`asyncio.new_event_loop`) mevcut desendir, bozma.

**JavaScript (frontend)**
- Vanilla JS, inline `<script>`. jQuery/axios **yok**. `document.getElementById` + `fetch().then(r=>r.json())` (promise zinciri; `async/await` yaygın değil).
- Yeni JS'i ilgili HTML'in inline script'ine ekle (harici .js dosyası konvansiyonu yok; `static/js` ve `web-service/js` boş).
- CDN bağımlılıkları (Bootstrap 5, Font Awesome 6, Chart.js, Socket.IO) head'de; self-host edilmemiş.

---

## 8. API Standardı

- **Endpoint adları**: `/api/` öneki (Flask), snake_case; FastAPI'de kısa isimler (`/analyze`, `/train`).
- **Response**: JSON. Mevcut zarf gevşek ve tutarsız (`{success, error, ...}` vs `jsonify(result)` vs `{status: ...}`). **Yeni endpoint'te** `{success: bool, data/..., error?: str}` desenini tercih et ve mevcut tüketiciyi bozma.
- **AI service contract'ı sabittir** (bkz. `ai-service/CLAUDE.md`); path/model alan adlarını değiştirme.
- Auth: korumalı olması gereken her endpoint `@login_required` almalı. (Mevcut korumasız `/api/*` route'ları güvenlik açığıdır — çoğaltma, düzelt.)

---

## 9. Database Standardı

**PostgreSQL + SQLAlchemy + Alembic + schema-per-tenant.** Bkz. **DATABASE.md**. Özet:
- snake_case tablo/kolon; PK Integer autoincrement (`users.id` String); `.upper()` normalizasyonu.
- Para/miktar/fiyat = `Numeric(18,8)` (Decimal, Float DEĞİL); yüzde/confidence `Numeric(10,4)`; JSON = `JSONB`.
- Soft delete (`is_active`/`is_open`/`is_valid`); hard delete önerme (schema drop hariç, onaylı).
- Migration = **Alembic** (iki track: shared/tenant); prod'da yalnız `upgrade`, autogenerate/db-push yok.
- Modeller (`models_shared/tenant.py`) + migration + repository'yi birlikte güncelle.
- **Tenant izolasyonu:** tenant tablosuna erişim aktif tenant bağlamı (`search_path`) gerektirir; bozma.
- Repository dış katmana `float`/ISO-string döndürür (API/jsonify uyumu) — bu sınır kuralını koru.

---

## 10. UI Standardı

Bkz. **web-service/CLAUDE.md** (tam kurallar). Özet:
- Bootstrap 5 + Font Awesome 6 (CDN), Jinja2 server-render, vanilla JS inline.
- Tema tek (açık), gradient (mor #667eea → #764ba2), yuvarlak köşe/glassmorphism; **dark mode yok**.
- Ortak bileşenler: navbar, card, nav-tabs, toast (dashboard), badge, table.
- Grafik: Chart.js (yalnız portfolio). Canlı güncelleme: SocketIO (yalnız dashboard).
- Jinja2 `analysis.*` dict contract'ını ve `url_for(...)` endpoint adlarını bozma.
- Türkçe UI metni; iki template kopyası (monolith `templates/` vs `web-service/`) — doğru kopyayı düzenle.

---

## 11. Test Standardı

- **Mevcut**: `test_*.py` + `*_test.py` gevşek script'leri (çoğu servis ayakta olmayı gerektiren entegrasyon testleri). `pytest`/`unittest` yapısı ve `tests/` klasörü **yok**.
- Çalıştırma: `python test_ai_service.py`, `python test_web_service.py`, `python quick_test.py`, `python test_comprehensive_system.py`.
- **Öneri**: Yeni testleri mevcut `test_*.py` konvansiyonunda yaz; büyük test altyapısı (pytest'e geçiş) ancak kullanıcı isterse yapılır.
- README'deki `python -m pytest tests/` **kurulu değildir** — dokümana böyle sunma.

---

## 12. Security Checklist (üretim öncesi)

- [ ] `.env.example` içindeki **gerçek kimlik bilgileri** temizlendi ve rotate edildi (şu an sızık).
- [ ] Tüm `.env` dosyaları commit dışı (`.gitignore` kapsıyor, doğrula).
- [ ] `SECRET_KEY`/`FLASK_SECRET_KEY` prod'da set edildi (hardcoded fallback kullanılmıyor).
- [ ] `DEBUG` / `reload=True` prod'da kapalı.
- [ ] CORS `*` prod'da spesifik domain'lerle değiştirildi (ai-service + web SocketIO).
- [ ] AI service'e auth/API key eklendi (`AI_SERVICE_API_KEY` tanımlı ama enforce edilmiyor).
- [ ] Korumasız `/api/*` route'ları (`portfolio_summary`, `coin_list`, `recent_trades`, `portfolio`, `close_position`) `@login_required` ile korundu.
- [ ] Log'a şifre/token/API key yazımı kaldırıldı (`auth.py`, `news_analyzer.py`).
- [ ] Varsayılan admin auto-create + `SimpleAuthManager` plaintext fallback prod'da devre dışı.
- [ ] `BINANCE_TESTNET` bilinçli ayarlandı (canlı para riski).
- [ ] Haber/whale mock-veri fallback'i prod'da sahte sinyal üretmiyor (key zorunlu veya açık uyarı).

---

## 13. Performance Checklist

- [ ] Model cache aktif (`model_cache.py`) — gereksiz yeniden eğitim yok (yaş 7 gün / accuracy 0.85 eşiği).
- [ ] Incremental training kullanılıyor (düşük LR, kısa epoch) — sıfırdan eğitim yalnız gerektiğinde.
- [ ] TF cihaz seçimi doğru (`tf_config`): M1/M2'de Metal/CPU fallback çalışıyor.
- [ ] Ağır TF çağrıları async/thread'e alınmış (`comprehensive_trainer` ThreadPoolExecutor).
- [ ] DB: PostgreSQL connection pool aktif (`session.py` pool_size/max_overflow, pool_pre_ping). Çok tenant'ta deploy migration döngüsü süresi göz önünde.
- [ ] `httpx` timeout'ları makul (analyze 60s, train 300s); retry yok (gerekirse ekle, contract'ı bozmadan).
- [ ] Frontend CDN'e bağımlı — offline/yavaş ağda stil çöker (gerekirse self-host).

---

## 14. Refactor Kuralları

- **Küçük ve izole tut.** Bu projede aynı dosyanın 2-3 fiziksel kopyası olduğu için (bkz. `CLAUDE.md §6.1`) bir refactor beklenenden geniş etki yaratabilir.
- Refactor'dan önce: hangi kopyalar etkilenir, hangi contract'lar (API/Jinja2/SocketIO/DB) bağlı, hangi model artefaktı geçersiz olur — **listele**.
- **Kullanılmıyor gibi görüneni silme**: proje çok sayıda yarım/placeholder entegrasyon içerir (bkz. `CLAUDE.md §7`).
- Feature/DB şeması/model I/O değişikliği model cache'i geçersiz kılabilir — bunu belirt.
- Büyük refactor'ları kullanıcı onayı olmadan yapma.

---

## 15. AI Agent Çalışma Kuralları (özet)

1. Değişiklikten önce ilgili `CLAUDE.md` + benzer örnek dosyayı oku.
2. Duplication tablosunu (`CLAUDE.md §6.1`) kontrol et; doğru kopyayı düzenle.
3. API/Jinja2/SocketIO/DB contract'larını bozma; response formatını değiştirme.
4. Auth/permission atlamama; tenant yapısı yok (uydurma ekleme).
5. Migration = Alembic (prod'da yalnız `upgrade`); `DROP SCHEMA`/veri kaybı riskli işlem önerme; tenant izolasyonunu bozma.
6. `tf_config` import sırası, scaler fit/transform, feature sırası (close=index 3, 25 feature) kutsal.
7. Sır/kimlik değerlerini dokümana/log'a/çıktıya yazma.
8. Yeni dependency'den önce mevcut paketle çöz.
9. Belirsizde varsayım yapma; `# NOT:` bırak veya sor.
10. Test/çalıştırma komutlarını (`DEVELOPMENT_GUIDE.md`) kullanarak değişikliği doğrula.
