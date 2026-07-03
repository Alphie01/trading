# CLAUDE.md — AI Service (FastAPI)

> Bu klasör, AI/ML işleme mikroservisidir (FastAPI, port 8000). Önce kök `../CLAUDE.md` ve `../ARCHITECTURE.md`'yi oku. Model I/O kuralları için ayrıca bu dosyanın §5'ine bak.

---

## 1. Sorumluluk

LSTM / DQN / Hybrid model eğitimi ve tahmini, haber & whale analizi, model cache yönetimi. `web-service` bu servise **HTTP/REST** ile bağlanır. Servis **stateless HTTP** sunar; kalıcı durum yalnızca `model_cache/` disk artefaktlarındadır.

## 2. Framework ve Server Yapısı

- **FastAPI** app: `ai_service.py` → `app = FastAPI(title="Crypto AI Trading Service", version="1.0.0")`.
- **Middleware**: yalnızca CORS, `allow_origins=["*"]` (kod yorumu "prod'da daralt" der).
- **Startup/shutdown/lifespan handler YOK**. Global nesneler (`data_fetcher`, `news_analyzer`, `whale_tracker`, `comprehensive_trainer`) modül import anında oluşur.
- **Çalıştırma**: `run_ai_service.py` → `uvicorn.run("ai_service:app", host="0.0.0.0", port=8000, reload=True)`.
- **Flat import**: `sys.path.append(...)`; paket yapısı yok.

> ⚠️ **`config.py` import EDİLMİYOR.** `ai_service.py` host/port/ayarları kendi main bloğunda hardcoded tutar; `config.py`'deki env tabanlı ayarlar (port, API key, epoch, cache dir) **fiilen etkisizdir**. Bir ayarı gerçekten devreye almak istiyorsan `ai_service.py`'nin `config`'i import etmesini sağlamalısın (ve bunu kullanıcıya bildir).

---

## 3. Endpoint'ler (API contract — DEĞİŞTİRME)

| Method | Path | Fonksiyon | Not |
|---|---|---|---|
| GET | `/` | `root` | servis meta bilgisi |
| GET | `/health` | `health_check` | sabit "healthy" JSON (gerçek sağlık kontrolü değil) |
| POST | `/analyze` | `analyze_coin` | coin analizi (aşağıdaki uyarıya bak) |
| POST | `/train` | `train_model` | comprehensive → BackgroundTasks; diğer → senkron |
| GET | `/models/{coin_symbol}` | `get_model_status` | `model_cache/{tip}_{coin}_model.h5` var mı + mtime |
| GET | `/training/status` | `get_training_status` | **placeholder** — her zaman boş |

### Pydantic modelleri (contract)
- **`CoinAnalysisRequest`**: `coin_symbol: str`, `analysis_type: str="comprehensive"` (lstm/dqn/hybrid/comprehensive), `use_news: bool=True`, `use_whale: bool=True`.
- **`TrainingRequest`**: `coin_symbol: str`, `training_type: str="comprehensive"`, `is_fine_tune: bool=False`, `epochs: Optional[int]`, `data_days: Optional[int]`.
- **`PredictionResponse`**: `success, coin_symbol, model_type, timestamp, current_price, predictions(Dict), technical_analysis?, news_analysis?, whale_analysis?, error?`.
- **`TrainingResponse`**: `success, coin_symbol, training_type, timestamp, results(Dict), performance?, error?`.

**`web-service/web_service.py:AIServiceClient` bu path'lere ve alanlara birebir bağlıdır. Path/alan adı değiştirirsen web servisi kırılır.**

---

## 4. Auth, Validation, Error, Logging, Response

- **Auth**: **YOK.** Hiçbir endpoint API key/token kontrol etmiyor. `config.py:API_KEY` (`AI_SERVICE_API_KEY`) tanımlı ama **enforce edilmiyor** (ve config zaten import edilmiyor). Güvenlik gerekiyorsa dependency-based auth ekle; mevcut korumasızlığı "pattern" sanma.
- **Validation**: Pydantic modelleriyle request doğrulama. `coin_symbol` normalizasyonu (upper) downstream'de yapılır.
- **Error handling (iki desen karışık)**: (a) `HTTPException` — veri yok=404, geçersiz tip=400, iç hata=500; (b) genel `except` iş hatalarını `success=False, error=str(e)` ile **HTTP 200** döndürür. Yeni endpoint'te: gerçek HTTP hatalarında `HTTPException`, iş "başarısız"ında `success:False` — ve seçimini açıkça belirt.
- **Response**: `/analyze` ve `/train` tipli model döner; `/`, `/health`, `/models`, `/training/status` ham dict döner (tutarsız zarf). Mevcut tüketiciyi bozmadan tutarlılığı artır.
- **Logging**: emoji'li Türkçe log/print. **API key / sır loglama** (mevcut `news_analyzer.py` key ön-eki logluyor — çoğaltma).

---

## 5. Model / ML Katmanı Kuralları (sessiz bozulma riski yüksek)

> Model dosyaları (`predictor.py`, `data_preprocessor.py`, `model_cache.py`, `news_analyzer.py`) monolitten **birebir kopyadır**; `dqn_trading_model.py` ve `lstm_model.py` ise **monolitten farklıdır** (bu servise özel değişiklikler içerir). Bir modeli düzenlerken hangi kopyada olduğunu bil.

- **`tf_config` import sırası KUTSAL**: her model dosyasının başında `from tf_config import get_tensorflow` gelir, TensorFlow'dan **önce** çalışır (M1/M2 Metal ayarları: memory growth, Metal placement kapatma). Bu sırayı bozma → çökme.
- **`directional_accuracy` custom metriği** `.h5` yüklemeden önce Keras'a register edilir (`predictor.py`). Kaldırma → model yüklenemez.
- **Scaler fit vs transform**: ilk eğitim `fit_scaler=True`; cache/incremental/tahmin `fit_scaler=False` (yalnız `transform`). Karıştırma → sessizce yanlış ölçekli tahmin. Scaler daima `_scaler.pkl` olarak modelle saklanır.
- **Feature sözleşmesi**: OHLCV ilk 5 sütun; hedef = **close (index 3)**; toplam **25 feature** (5 OHLCV + 20 teknik gösterge, `data_preprocessor.py`). Sıra/sayı değişince `create_sequences`/`inverse_transform` ve tüm cache bozulur.
- **DQN**: `action_space=9` (`0=HOLD, 1-4=BUY_%25..100, 5-8=SELL_%25..100`), state ~31 boyut (technical 15 + portfolio 5 + market 8 + lstm 3 placeholder). `confidence` clip **[0.25, 0.80]** (kod yorumu hâlâ "85%" der ama gerçek cap 0.80'dir; 0.85 yalnız cache-retrain eşiğidir — karıştırma). State boyutu değişirse kayıtlı model yüklenmez.
- **Hybrid**: ağırlıklı ensemble (`lstm:0.35, dqn:0.45, technical:0.2`, çalışma anında optimize edilebilir). `recommendation` enum: `STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL`. Hybrid'de açık %85 confidence cap yoktur.
- **`predictor.py` iş kuralı**: LSTM ilk eğitimde başarısız + cache yoksa DQN/Hybrid **atlanır** — bilinçli, koru.
- **Model cache adlandırma (iki şema bir arada)**:
  - `ModelCache` sınıfı: `model_cache/models/{id}.h5`, `.../metadata/{id}.json`, `.../preprocessors/{id}.pkl` + `{id}_scaler.pkl`, `id = {SYMBOL}_{md5(config)[:8]}`.
  - Diskte fiilen kullanılan (bazı yollar): `model_cache/{tip}_{coin}_model.h5` (ör. `lstm_btc_model.h5`) — `ai_service.py` `/models` bunu kontrol eder.
  - Bu iki şema **tutarsızdır ve kırılgandır**; birini değiştirmeden önce ikisini de anla.
- **`feature_columns` mutasyon bug'ı**: `prepare_data` her çağrıda listeyi `extend` eder; aynı preprocessor örneğiyle tekrar çağrılırsa feature sayısı büyür → shape uyumsuzluğu. Refactor'da dikkat.

---

## 6. Async Training

- **Tetikleme**: `/train` + `comprehensive` → `BackgroundTasks.add_task(...)`, anında `{"status":"training_started"}` döner. Diğer tipler senkron.
- **Trainer**: `comprehensive_trainer.py:ComprehensiveTrainer` → `ThreadPoolExecutor`/`run_in_executor` ile bloklamayan TF çağrıları.
- **Durum takibi YOK**: `/training/status` sabit boş döner; background görev yalnız `logger`'a yazar.

> 🐞 **Bilinen bug**: `_background_comprehensive_training`, `comprehensive_trainer.train_coin_async(...)` çağırıyor ama bu metod **tanımlı değil** (sadece `train_all_models_for_coin` ve `train_coin_sync` var) → background comprehensive eğitim `AttributeError` ile sessizce düşer. "Düzeltmeden" önce hangi metodun kullanılması gerektiğini kullanıcıyla netleştir.

---

## 7. Kesinlikle Yapılmaması Gerekenler

- ❌ Endpoint path'lerini veya Pydantic alan adlarını değiştirme (web-service contract'ı).
- ❌ `tf_config` import sırasını bozma; TensorFlow'u en tepede doğrudan import etme.
- ❌ Feature sırası/sayısını (25, close=index 3) veya scaler fit/transform mantığını değiştirme.
- ❌ Model cache adlandırma şemasını tek taraflı değiştirme.
- ❌ CORS/auth durumunu "örnek" sanıp çoğaltma; gerçek gizli veri döndüren endpoint eklerken auth ekle.
- ❌ Placeholder değerleri (`*1.02`, sabit confidence) gerçek çıktı sanıp üstüne mantık kurma — `/analyze` gerçek `CryptoPricePredictor`'ı henüz tam kullanmıyor.
- ❌ Mock/demo veri fallback'lerini (news/whale) sessizce "gerçek veri" gibi sunma.
- ❌ Sır/API key'i log'a veya yanıta yazma.

## 8. Bu Serviste Çalışırken Önce Oku

- `ai_service.py` (endpoint + contract), `comprehensive_trainer.py` (async eğitim).
- Model I/O değişikliği: `data_preprocessor.py` (feature), `model_cache.py` (cache/scaler), ilgili `*_model.py`.
- Kök: `../CLAUDE.md §6` (kesin kurallar), `../PROJECT_STANDARDS.md`, `../DEVELOPMENT_GUIDE.md §14` (yeni feature ekleme).
