# 🚀 Kripto Para LSTM Fiyat Tahmini Sistemi

Bu proje, kullanıcıdan bir kripto para (coin) ismi alarak Binance borsasından bu coine ait son 100 günlük verileri 4 saatlik mumlar (OHLCV - Open, High, Low, Close, Volume) şeklinde çeken ve bu verilerle bir LSTM (Long Short-Term Memory) modeli eğiten bir Python uygulamasıdır.

## 📋 Özellikler

- 🔗 **Binance API Entegrasyonu**: Gerçek zamanlı kripto para verisi çekme
- 📊 **Teknik Analiz Göstergeleri**: RSI, MACD, Bollinger Bands, SMA, EMA
- 🧠 **LSTM Derin Öğrenme Modeli**: Gelişmiş zaman serisi tahmini
- 📰 **Haber Sentiment Analizi**: NewsAPI, CoinDesk, Reddit entegrasyonu
- 🤖 **FinBERT AI Modeli**: Finansal sentiment analizi için özel AI modeli
- 🐋 **Whale Tracker**: Büyük kripto cüzdanlarının hareketlerini takip ve analiz
- 📈 **Hibrit Tahmin Sistemi**: Fiyat + Haber + Whale analizinin birleşimi
- 📊 **Haber-Fiyat Korelasyonu**: Haberlerin fiyat etkisini ölçme
- 💰 **Whale-Fiyat Korelasyonu**: Büyük transferlerin piyasa etkisini analiz
- 📝 **Kapsamlı Raporlar**: LSTM + Haber + Whale analizi birleşik raporları
- 🎯 **Aksiyon Önerileri**: Gelecek haberler ve whale hareketleri için strateji
- 📈 **Görselleştirme**: Detaylı grafik ve analiz çıktıları
- ⚡ **GPU Desteği**: TensorFlow GPU akselerasyonu
- 🎲 **Güvenilirlik Skoru**: Çok boyutlu güvenilirlik değerlendirmesi

## 🛠️ Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- GPU kullanımı için CUDA (opsiyonel)

### 1. Proje Dosyalarını İndirin

```bash
git clone https://github.com/kullanici/crypto-lstm-prediction.git
cd crypto-lstm-prediction
```

### 2. Sanal Ortam Oluşturun (Önerilen)

```bash
python -m venv crypto_env
source crypto_env/bin/activate  # Linux/Mac
# veya
crypto_env\Scripts\activate     # Windows
```

### 3. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

## 🚀 Kullanım

### Temel Kullanım

```bash
python main.py
```

Uygulama başladığında:

1. 📝 Analiz etmek istediğiniz coin ismini girin (örn: BTC, ETH, ADA)
2. ⚙️ Model parametrelerini ayarlayın veya varsayılan değerleri kullanın
3. 📰 Haber analizi kullanıp kullanmayacağınızı seçin
4. 🔑 NewsAPI anahtarınızı girin (opsiyonel - daha fazla haber için)
5. ⏳ Sistem otomatik olarak:
   - Binance'den fiyat verisi çekme
   - Haber kaynaklarından sentiment analizi
   - Teknik göstergeler hesaplama
   - Haber-fiyat korelasyon analizi
   - LSTM model eğitimi (hibrit özelliklerle)
   - Gelecek haber etkilerini tahmin etme
   - Kapsamlı rapor hazırlama

### Hızlı Demo

BTC ile hızlı bir demo için:

```bash
python main.py --demo
```

### 📰 Haber Analizi Örneği

Sadece haber analizi özelliklerini test etmek için:

```bash
python example_usage.py
```

### 🔑 API Anahtarları (Opsiyonel)

#### NewsAPI Anahtarı

Daha fazla haber kaynağı için ücretsiz NewsAPI anahtarı alabilirsiniz:

1. [NewsAPI.org](https://newsapi.org/) adresine gidin
2. Ücretsiz hesap oluşturun
3. API anahtarınızı kopyalayın
4. Uygulamada istendiğinde girin

**Not**: NewsAPI anahtarı olmadan da sistem CoinDesk ve Reddit kaynaklarından haber çeker.

#### Whale Alert API Anahtarı

Büyük cüzdan transferleri için Whale Alert API anahtarı:

1. [Whale Alert API](https://whale-alert.io/api) adresine gidin
2. Ücretsiz hesap oluşturun (günde 100 sorgu)
3. API anahtarınızı alın
4. Uygulamada whale analizi kullanırken girin

**Not**: Whale Alert API anahtarı olmadan da sistem demo whale verileri oluşturur.

## 📊 Model Parametreleri

### Varsayılan Ayarlar

- **Sekans Uzunluğu**: 60 (240 saat = 10 gün)
- **Epoch Sayısı**: 50
- **Batch Boyutu**: 32
- **LSTM Katmanları**: [50, 50, 50]
- **Dropout Oranı**: 0.2
- **Öğrenme Oranı**: 0.001

### Özelleştirme

Model parametrelerini çalışma zamanında değiştirebilirsiniz:

```python
from lstm_model import CryptoLSTMModel

model = CryptoLSTMModel(sequence_length=60, n_features=16)
model.build_model(
    lstm_units=[64, 64, 32],  # Daha büyük model
    dropout_rate=0.3,         # Daha yüksek regularization
    learning_rate=0.0005      # Daha düşük öğrenme oranı
)
```

## 📈 Teknik Göstergeler

Sistem aşağıdaki teknik analiz göstergelerini otomatik hesaplar:

- **SMA (Simple Moving Average)**: 7 ve 25 günlük
- **EMA (Exponential Moving Average)**: 12 günlük
- **RSI (Relative Strength Index)**: 14 periyotluk
- **MACD**: 12-26-9 parametreleri
- **Bollinger Bands**: 20 günlük, 2 standart sapma
- **Yigit ATR Trailing Stop**: Pine Script ported, trend takibi ve sinyaller
- **Volume-Price Analizi**: Hacim/Fiyat oranı hesaplaması
- **Fiyat Değişim Yüzdesi**
- **Volume Değişim Yüzdesi**

## 📁 Proje Yapısı

```
crypto-lstm-prediction/
│
├── main.py                 # Ana uygulama dosyası (hibrit sistem)
├── data_fetcher.py         # Binance API veri çekme modülü
├── data_preprocessor.py    # Veri ön işleme ve teknik göstergeler
├── news_analyzer.py        # Haber sentiment analizi modülü
├── whale_tracker.py        # Whale (büyük cüzdan) takip modülü (YENİ!)
├── lstm_model.py          # LSTM model tanımı ve eğitimi
├── predictor.py           # Hibrit tahmin ve rapor oluşturma
├── example_usage.py       # Örnek kullanım senaryoları
├── requirements.txt       # Python kütüphane gereksinimleri
└── README.md             # Bu dosya
```

## 💡 Kullanım Örnekleri

### 1. Bitcoin Analizi

```python
from data_fetcher import CryptoDataFetcher
from data_preprocessor import CryptoDataPreprocessor
from lstm_model import CryptoLSTMModel
from predictor import CryptoPricePredictor

# Veri çekme
fetcher = CryptoDataFetcher()
btc_data = fetcher.fetch_ohlcv_data('BTC')

# Veri hazırlama
preprocessor = CryptoDataPreprocessor()
processed_data = preprocessor.prepare_data(btc_data)
scaled_data = preprocessor.scale_data(processed_data)
X, y = preprocessor.create_sequences(scaled_data)

# Model eğitimi
model = CryptoLSTMModel(60, X.shape[2])
model.build_model()
model.train_model(X_train, y_train, X_val, y_val)

# Tahmin
predictor = CryptoPricePredictor(model, preprocessor)
prediction = predictor.predict_next_price(processed_data)
```

### 2. Hibrit Tahmin (Fiyat + Haber + Whale Analizi)

```python
from news_analyzer import CryptoNewsAnalyzer
from whale_tracker import CryptoWhaleTracker

# Haber analizi ekle
news_analyzer = CryptoNewsAnalyzer(newsapi_key="your_api_key")
all_news = news_analyzer.fetch_all_news('BTC', days=100)

# Sentiment analizi
news_sentiment_df = news_analyzer.analyze_news_sentiment_batch(all_news)
sentiment_df = news_analyzer.create_daily_sentiment_features(news_sentiment_df, btc_data)

# Whale analizi ekle
whale_tracker = CryptoWhaleTracker(whale_alert_api_key="your_whale_key")
whale_transactions = whale_tracker.fetch_whale_alert_transactions('BTC', hours=48)
whale_analysis = whale_tracker.analyze_whale_transactions(whale_transactions)
whale_features = whale_tracker.create_whale_features(whale_analysis, 48)

# Hibrit veri hazırlama (tüm özellikler)
processed_data = preprocessor.prepare_data(btc_data, 
                                         use_technical_indicators=True,
                                         sentiment_df=sentiment_df,
                                         whale_features=whale_features)

# Hibrit tahmin
predictor = CryptoPricePredictor(model, preprocessor, news_analyzer, whale_tracker)
prediction = predictor.predict_next_price(processed_data)

# Haber tabanlı strateji analizi
news_analysis = predictor.analyze_recent_news_impact('BTC', days=7)
whale_impact = predictor.analyze_whale_impact('BTC', hours=24)

print(f"Haber Stratejisi: {news_analysis['recommended_action']}")
print(f"Whale Stratejisi: {whale_impact.get('strategy', 'Veri yok')}")
```

### 3. Çoklu Dönem Tahmini

```python
# 24 saatlik tahmin (6 dönem x 4 saat)
multiple_predictions = predictor.predict_multiple_periods(processed_data, periods=6)

for i, pred in enumerate(multiple_predictions):
    print(f"Dönem {i+1}: ${pred['predicted_price']:.2f} ({pred['price_change_percent']:+.2f}%)")
```

### 4. Kapsamlı Haber Analizi

```python
# Son günlerin haberleri
recent_news = news_analyzer.fetch_all_news('BTC', days=7)
sentiment_results = news_analyzer.analyze_news_sentiment_batch(recent_news)

# Sentiment istatistikleri
avg_sentiment = sentiment_results['overall_sentiment'].mean()
positive_news = len(sentiment_results[sentiment_results['overall_sentiment'] > 0.1])

print(f"Ortalama Sentiment: {avg_sentiment:.3f}")
print(f"Pozitif Haberler: {positive_news}")

# Haber-fiyat korelasyonu
correlation = news_analyzer.calculate_news_price_correlation(sentiment_df, price_data)
print(f"Korelasyon: {correlation['correlation']:.3f}")
```

### 5. Whale (Büyük Cüzdan) Analizi

```python
from whale_tracker import CryptoWhaleTracker

# Whale tracker oluştur
whale_tracker = CryptoWhaleTracker(whale_alert_api_key="your_api_key")

# Whale transferlerini çek
whale_transactions = whale_tracker.fetch_whale_alert_transactions('BTC', hours=48)

if whale_transactions:
    # Whale analizi
    whale_analysis = whale_tracker.analyze_whale_transactions(whale_transactions)
    
    print(f"Whale İşlem Sayısı: {whale_analysis['transaction_count']}")
    print(f"Toplam Hacim: ${whale_analysis['total_volume']:,.0f}")
    print(f"Net Flow: ${whale_analysis['net_flow']:,.0f}")
    print(f"Aktivite Skoru: {whale_analysis['whale_activity_score']:.1f}/100")
    
    # Whale-fiyat korelasyonu
    correlation = whale_tracker.analyze_whale_price_correlation(whale_analysis, price_data, 'BTC')
    print(f"Fiyat Korelasyonu: {correlation['correlation']:.3f}")
    
    # Strateji önerisi
    strategy = whale_tracker.get_whale_strategy_recommendation(whale_analysis, correlation)
    print(f"Whale Stratejisi: {strategy['strategy']}")
    print(f"Güven Seviyesi: {strategy['confidence']}")
    
    # Detaylı analiz
    whale_features = whale_tracker.create_whale_features(whale_analysis, 48)
    print(f"Whale Sentiment: {whale_features['whale_sentiment']:.3f}")
    print(f"Exchange Giriş Oranı: {whale_features['whale_inflow_ratio']:.2f}")
    print(f"Exchange Çıkış Oranı: {whale_features['whale_outflow_ratio']:.2f}")
```

### 6. Yigit ATR Trailing Stop Analizi

```python
# Yigit indikatör sinyallerini analiz et
yigit_analysis = predictor.analyze_yigit_signals(processed_data)

if yigit_analysis['has_yigit']:
    print(f"Trend Durumu: {yigit_analysis['direction']}")
    print(f"Son Sinyal: {yigit_analysis['current_signal']}")
    print(f"Trend Gücü: {yigit_analysis['trend_strength']:.3f}")
    print(f"Strateji: {yigit_analysis['strategy_recommendation']}")
    
    # Volume-Price analizi
    print(f"V/P Oranı: {yigit_analysis['volume_price_ratio']:.6f}")
    
    # Son sinyaller
    print(f"Son 10 dönem Al sinyali: {yigit_analysis['recent_buy_signals']}")
    print(f"Son 10 dönem Sat sinyali: {yigit_analysis['recent_sell_signals']}")
```

## ⚠️ Önemli Uyarılar

- 📊 **Yatırım Tavsiyesi Değildir**: Bu sistem sadece eğitim ve araştırma amaçlıdır
- 🎲 **Yüksek Risk**: Kripto para yatırımları son derece risklidir
- 📈 **Geçmiş Performans**: Geçmiş veriler gelecekteki performansı garanti etmez
- 💰 **Sorumlu Yatırım**: Sadece kaybetmeyi göze alabileceğiniz parayla yatırım yapın

## 🔧 Sorun Giderme

### Yaygın Hatalar

**1. ModuleNotFoundError**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**2. GPU Hatası**
```python
# CPU kullanımını zorlamak için
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

**3. API Hatası**
- İnternet bağlantınızı kontrol edin
- Binance API'sinin çalışır durumda olduğunu doğrulayın

### Performans Optimizasyonu

**GPU Kullanımı**
```bash
# NVIDIA GPU için
pip install tensorflow-gpu

# Veya yeni TensorFlow sürümlerinde
pip install tensorflow[and-cuda]
```

**Bellek Kullanımı**
```python
# Sequence length'i azaltın
sequence_length = 30  # Varsayılan: 60

# Batch size'ı küçültün
batch_size = 16  # Varsayılan: 32
```

## 📚 Teknik Detaylar

### Model Mimarisi

```
Input Layer (60, n_features)
    ↓
LSTM Layer (50 units) + Dropout + BatchNorm
    ↓
LSTM Layer (50 units) + Dropout + BatchNorm
    ↓
LSTM Layer (50 units) + Dropout + BatchNorm
    ↓
Dense Layer (25 units, ReLU)
    ↓
Output Layer (1 unit, Linear)
```

### Veri Akış Şeması

```
Binance API → OHLCV Data → Technical Indicators
     ↓
NewsAPI/CoinDesk/Reddit → Sentiment Analysis → Daily Features
     ↓
Whale Alert API → Whale Transactions → Whale Features
     ↓
Feature Integration → Normalization → Sequences → Train/Val/Test Split
     ↓
LSTM Training → Hibrit Prediction → Comprehensive Report
```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 📞 İletişim

- 📧 Email: your-email@example.com
- 🐱 GitHub: [@yourusername](https://github.com/yourusername)
- 💼 LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Teşekkürler

- [Binance API](https://binance-docs.github.io/apidocs/) - Kripto para verileri
- [TensorFlow](https://tensorflow.org/) - Derin öğrenme framework'ü
- [CCXT](https://github.com/ccxt/ccxt) - Kripto exchange kütüphanesi
- [Scikit-learn](https://scikit-learn.org/) - Makine öğrenimi araçları

---

⭐ Bu projeyi beğendiyseniz, lütfen star verin!

**Risk Uyarısı**: Bu yazılım yalnızca eğitim amaçlıdır. Finansal yatırım kararları vermek için kullanmayın. Kripto para yatırımları yüksek risk içerir ve tüm yatırımınızı kaybedebilirsiniz. 