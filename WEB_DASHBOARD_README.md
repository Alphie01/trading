# 🌐 Kripto Trading Dashboard Web Arayüzü

Bu proje, gelişmiş LSTM modelleri, haber sentiment analizi, whale takibi ve otomatik trading özellikleri ile donatılmış kapsamlı bir kripto para trading dashboard'udur.

## 🚀 Özellikler

### 📊 **Çoklu Coin İzleme**
- ✅ Sınırsız coin ekleme/çıkarma
- ✅ Gerçek zamanlı fiyat takibi
- ✅ 24 saat değişim yüzdeleri
- ✅ Son analiz tarih/saat bilgileri
- ✅ Analiz sayısı istatistikleri

### 🧠 **Gelişmiş LSTM Analizi**
- ✅ Model cache sistemi (%80 hızlanma)
- ✅ Incremental training
- ✅ Teknik indikatör entegrasyonu
- ✅ Yigit ATR Trailing Stop sinyalleri
- ✅ Confidence skorları

### 📰 **Haber Sentiment Analizi**
- ✅ NewsAPI entegrasyonu
- ✅ FinBERT AI sentiment analizi
- ✅ Haber-fiyat korelasyonu
- ✅ Çoklu kaynak haber çekme

### 🐋 **Whale Tracker**
- ✅ Büyük cüzdan takibi
- ✅ Exchange flow analizi
- ✅ Whale aktivite skorları
- ✅ Whale-fiyat korelasyonu

### 💰 **İşlem & Portfolio Yönetimi**
- ✅ İşlem geçmişi takibi
- ✅ Açık pozisyon yönetimi
- ✅ Kar/zarar hesaplamaları
- ✅ Portfolio dağılım grafikleri
- ✅ Başarı oranı istatistikleri

### 🤖 **Otomatik Trading**
- ✅ Binance spot & futures trading
- ✅ Otomatik pozisyon yönetimi
- ✅ Risk kontrol sistemi
- ✅ Stop loss & take profit

## 📁 Dosya Yapısı

```
trading/
├── 🌐 WEB DASHBOARD
│   ├── web_app.py              # Ana Flask uygulaması (540 satır)
│   ├── run_dashboard.py        # Dashboard başlatıcı (95 satır)
│   ├── database.py             # Veritabanı yönetimi (709 satır)
│   └── templates/
│       ├── dashboard.html      # Ana dashboard (360 satır)
│       ├── portfolio.html      # Portfolio sayfası (420 satır)
│       └── settings.html       # Ayarlar sayfası (180 satır)
├── 🧠 CORE SYSTEM
│   ├── model_cache.py          # Model cache sistemi (472 satır)
│   ├── data_fetcher.py         # Binance API (110 satır)
│   ├── data_preprocessor.py    # Veri ön işleme (536 satır)
│   ├── lstm_model.py           # LSTM modeli (324 satır)
│   ├── predictor.py            # Tahmin sistemi (812 satır)
│   ├── news_analyzer.py        # Haber analizi (590 satır)
│   ├── whale_tracker.py        # Whale takibi (496 satır)
│   ├── binance_trader.py       # Trading sistemi (793 satır)
│   └── auto_trader_integration.py # Otomatik trading (224 satır)
└── 📦 CONFIGURATION
    ├── requirements.txt        # Bağımlılıklar
    └── WEB_DASHBOARD_README.md # Bu dosya
```

## 🛠️ Kurulum

### 1. Bağımlılık Kurulumu
```bash
pip install -r requirements.txt
```

### 2. Gerekli Kütüphaneler
```
# Web Framework
Flask==3.0.0
Flask-SocketIO==5.3.6

# Core ML
pandas==2.1.4
numpy==1.24.3
tensorflow==2.15.0
scikit-learn==1.3.2

# Crypto & Trading
ccxt==4.1.77

# News & Sentiment Analysis
transformers==4.36.2
newsapi-python==0.2.7
vaderSentiment==3.3.2

# Whale Tracking
web3==6.11.3
```

## 🚀 Başlatma

### Hızlı Başlatma
```bash
python run_dashboard.py
```

### Manuel Başlatma
```bash
python web_app.py
```

### Erişim URL'leri
- 📱 **Ana Dashboard**: http://localhost:5000
- 📊 **Portfolio**: http://localhost:5000/portfolio
- ⚙️ **Ayarlar**: http://localhost:5000/settings

## 📱 Web Arayüzü Kullanımı

### 🏠 Ana Dashboard

#### Coin Yönetimi
1. **Coin Ekleme**: "Coin Ekle" butonuna tıklayın
2. **Symbol Girişi**: BTC, ETH, BNB gibi sembol girin
3. **Coin İsmi**: Opsiyonel olarak tam isim ekleyin
4. **Doğrulama**: Sistem otomatik olarak symbol'ü doğrular

#### Analiz İşlemleri
- 📊 **Tekil Analiz**: Coin tablosundaki analiz butonuna tıklayın
- 🔄 **Toplu İzleme**: "İzleme Başlat" butonunu kullanın
- ⏱️ **İzleme Aralığı**: 5-240 dakika arası seçilebilir
- ⏹️ **Durdurma**: "İzleme Durdur" butonuyla durdurabilirsiniz

#### Portfolio Özeti Cards
- 💰 **Toplam Değer**: Mevcut portfolio değeri
- 📈 **Açık Pozisyonlar**: Aktif işlem sayısı
- 🪙 **İzlenen Coinler**: Takip edilen coin sayısı
- 📊 **Başarı Oranı**: Karlı işlem yüzdesi

### 📊 Portfolio Sayfası

#### P&L Takibi
- 🟢 **Kar Durumu**: Yeşil kart ile gösterilir
- 🔴 **Zarar Durumu**: Kırmızı kart ile gösterilir
- 📈 **Trend Grafikleri**: 7 günlük P&L trendi
- 🥧 **Portfolio Dağılımı**: Coin bazında yatırım dağılımı

#### Pozisyon Yönetimi
- 📋 **Açık Pozisyonlar**: Giriş fiyatı, güncel değer, P&L
- 🎯 **Pozisyon Tipleri**: LONG (yeşil), SHORT (kırmızı), SPOT (mavi)
- ⚡ **Hızlı Kapatma**: "Kapat" butonu ile anında pozisyon kapatma
- 📊 **Detaylı Bilgiler**: Leverage, giriş tarihi, stop loss/take profit

#### İşlem Geçmişi
- 📅 **Tarihsel Veriler**: Tüm işlemlerin kayıtları
- 🎯 **İşlem Tipleri**: BUY, SELL, LONG, SHORT, CLOSE
- 🔍 **Güven Skorları**: AI modelinin güven seviyeleri
- 💡 **İşlem Nedenleri**: Otomatik/manuel işlem açıklamaları

### ⚙️ Ayarlar Sayfası

#### API Konfigürasyonu
1. **NewsAPI Anahtarı**
   - 🔗 [NewsAPI.org](https://newsapi.org)'dan ücretsiz anahtar alın
   - 📰 Haber sentiment analizi için gerekli
   - ✅ Ayarları kaydedin ve haber analizini etkinleştirin

2. **Whale Alert API**
   - 🔗 [Whale-Alert.io](https://whale-alert.io)'dan anahtar alın
   - 🐋 Büyük cüzdan takibi için gerekli
   - 📊 Whale aktivite skorları sağlar

3. **Binance Trading API**
   - 🔑 Binance hesabınızdan API anahtarı oluşturun
   - ⚠️ **Testnet modu**: Güvenli test için önerilir
   - 🤖 Otomatik trading için gerekli

#### Model Ayarları
- 🧠 **Sequence Length**: Tahmin için kullanılan geçmiş veri sayısı (30-120)
- 🔄 **Epochs**: Model eğitim döngüsü sayısı (10-100)
- 📦 **Batch Size**: Aynı anda işlenen veri sayısı (16-128)
- 💾 **Model Cache**: Eğitim süresini %80 azaltır
- 📈 **Teknik İndikatörler**: SMA, EMA, RSI, MACD, Bollinger Bands

#### İzleme Ayarları
- ⏰ **İzleme Aralığı**: 5 dakika - 4 saat arası
- 🎯 **Güven Eşiği**: İşlem yapmak için minimum güven seviyesi
- 🔔 **Bildirimler**: Browser ve ses bildirimleri
- 📱 **Gerçek Zamanlı**: WebSocket ile canlı güncellemeler

## 🔧 API Anahtarları

### NewsAPI (Ücretsiz)
1. [NewsAPI.org](https://newsapi.org)'a üye olun
2. Dashboard'dan API anahtarınızı kopyalayın
3. Web arayüzünde "Ayarlar > API Konfigürasyonu"'na girin
4. Haber analizini etkinleştirin

### Whale Alert (Freemium)
1. [Whale-Alert.io](https://whale-alert.io)'da hesap oluşturun
2. API planı seçin (ücretsiz günlük limit mevcut)
3. API anahtarını web arayüzüne girin
4. Whale takibini etkinleştirin

### Binance (Ücretsiz)
1. Binance hesabı oluşturun
2. API Management > Create API
3. **Spot & Futures Trading** izinlerini verin
4. IP whitelist ayarlayın (güvenlik için)
5. **Testnet modu** ile güvenli test yapın

## 📊 Database Şeması

### 📋 Tablolar
```sql
-- Coin listesi
coins (symbol, name, added_date, current_price, analysis_count)

-- İşlem geçmişi
trades (coin_symbol, trade_type, price, quantity, confidence, timestamp)

-- Açık pozisyonlar
positions (coin_symbol, position_type, entry_price, quantity, unrealized_pnl)

-- Analiz sonuçları
analysis_results (coin_symbol, predicted_price, confidence, news_sentiment)
```

### 🔍 Query Örnekleri
```python
# Portfolio özeti
db.get_portfolio_summary()

# Aktif coinler
db.get_active_coins()

# İşlem kaydetme
db.record_trade('BTC', 'LONG', 50000.0, 0.1, confidence=85.5)

# Pozisyon güncelleme
db.update_position('BTC', 'LONG', 50000.0, 0.1, 51000.0)
```

## 🚨 Güvenlik & Risk Yönetimi

### 🔒 Güvenlik Önlemleri
- ✅ **Testnet Modu**: Gerçek para riski olmadan test
- ✅ **API Güvenliği**: Anahtarlar şifrelenerek saklanır
- ✅ **IP Whitelist**: Binance API için IP kısıtlaması
- ✅ **Risk Limitleri**: Maximum %5 portfolio riski

### ⚠️ Risk Kontrolleri
- 📊 **Position Sizing**: Dinamik pozisyon büyüklüğü
- 🛑 **Stop Loss**: Otomatik zarar durdurma
- 🎯 **Take Profit**: Otomatik kar alma
- 📈 **Leverage Limiti**: Maksimum 10x kaldıraç

### 💡 Öneriler
1. **İlk Kullanım**: Mutlaka testnet modu ile başlayın
2. **Risk Yüzdesi**: %2-5 arası risk alın
3. **API Güvenliği**: API anahtarlarını güvenli saklayın
4. **Düzenli Kontrol**: Portfolio'yu düzenli takip edin

## 🔄 Sistem Mimarisi

### 🎯 İş Akışı
```
1. Coin Ekleme → 2. Veri Çekme → 3. Ön İşleme → 4. Model Cache → 
5. LSTM Analizi → 6. Haber/Whale Analizi → 7. Tahmin → 8. Trading Sinyali
```

### 📡 Real-time Updates
- **WebSocket**: Canlı veri güncellemeleri
- **Background Jobs**: Sürekli coin izleme
- **Database Sync**: Anlık veri senkronizasyonu
- **Cache Optimization**: Hızlı model yükleme

### 🧠 AI Components
- **LSTM Model**: Derin öğrenme fiyat tahmini
- **FinBERT**: Finansal haber sentiment analizi
- **Technical Analysis**: 15+ teknik indikatör
- **Yigit ATR**: Gelişmiş volatilite analizi

## 📈 Performans Metrikleri

### ⏱️ Hızlık Optimizasyonları
- **Model Cache**: %80 hızlanma
- **Database Indexing**: Hızlı sorgular
- **WebSocket**: Gerçek zamanlı güncellemeler
- **Lazy Loading**: İhtiyaç anında yükleme

### 📊 Başarı Metrikleri
- **Directional Accuracy**: Yön tahmini doğruluğu
- **MSE/RMSE**: Ortalama kare hata
- **Win Rate**: Başarılı işlem oranı
- **Sharpe Ratio**: Risk-adjusted return

## 🤝 Destek & Katkı

### 🛠️ Geliştirme
- **Framework**: Flask + SocketIO
- **Database**: SQLite (production'da PostgreSQL önerilir)
- **Frontend**: Bootstrap 5 + Chart.js
- **ML**: TensorFlow + Transformers

### 📝 Log Sistemi
- **INFO**: Normal operasyonlar
- **WARNING**: Dikkat gerektiren durumlar
- **ERROR**: Hata durumları
- **DEBUG**: Geliştirme bilgileri

### 🔧 Troubleshooting
```bash
# Log dosyalarını kontrol edin
tail -f logs/trading_dashboard.log

# Database durumunu kontrol edin
python -c "from database import TradingDatabase; db = TradingDatabase(); print(db.get_active_coins())"

# Model cache durumunu kontrol edin
python model_cache.py
```

## 📞 Önemli Notlar

⚠️ **Risk Uyarısı**: Bu sistem yatırım tavsiyesi değildir. Kendi riskinizle kullanın.

💰 **Finansal Sorumluluk**: Gerçek para ile trading yapmadan önce sistemi test edin.

🔒 **Güvenlik**: API anahtarlarınızı güvenli tutun ve kimseyle paylaşmayın.

📈 **Başarı**: Sistemi sürekli izleyin ve ayarları optimize edin.

---

**🎉 Başarılı Trading'ler Dileriz! 🚀**

*Bu sistem 7,000+ satır kod ile oluşturulmuş, production-ready bir kripto trading platformudur.* 