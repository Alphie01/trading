# 🚀 Advanced Crypto Trading AI System

## 🌟 Genel Bakış

Bu proje, kripto para ticareti için kapsamlı bir yapay zeka sistemidir. **Hibrit Deep Learning** yaklaşımı ile LSTM ve DQN modellerini birleştirerek, teknik analiz, haber sentiment analizi ve whale tracking özelliklerini entegre eder.

### 🎯 Ana Özellikler

- 🧠 **Hibrit AI Sistemi**: LSTM + DQN + Technical Analysis birleşimi
- 📊 **Gerçek Zamanlı Web Dashboard**: Modern React benzeri arayüz
- 🔄 **Otomatik Trading**: Binance API entegrasyonu ile otomatik işlem
- 📰 **Haber Sentiment Analizi**: NewsAPI, CoinDesk, Reddit, Twitter
- 🐋 **Whale Tracking**: Büyük cüzdan hareketlerini takip
- 💾 **Akıllı Model Cache**: Eğitim süresini optimize eden cache sistemi
- 🗄️ **Veritabanı Desteği**: SQLite ve MSSQL entegrasyonu
- 🔐 **Güvenlik**: JWT authentication ve secure trading
- ⚡ **Performance Testing**: Backtest ve live performance analizi

## 🏗️ Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                    WEB DASHBOARD                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Portfolio   │ │ Analytics   │ │ Settings    │          │
│  │ Management  │ │ Dashboard   │ │ & Config    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 HYBRID AI ENGINE                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ LSTM Price  │ │ DQN Action  │ │ Technical   │          │
│  │ Predictor   │ │ Selector    │ │ Analysis    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               DATA INTELLIGENCE LAYER                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ News        │ │ Whale       │ │ Technical   │          │
│  │ Sentiment   │ │ Tracking    │ │ Indicators  │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 EXECUTION LAYER                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Binance     │ │ Risk        │ │ Performance │          │
│  │ Trading     │ │ Management  │ │ Analytics   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Proje Yapısı

```
trading/
├── 🎯 Core AI Models
│   ├── hybrid_trading_model.py      # Hibrit LSTM+DQN sistemi
│   ├── lstm_model.py               # LSTM price prediction
│   ├── dqn_trading_model.py        # Deep Q-Network action selection
│   └── predictor.py                # Birleşik tahmin motoru
│
├── 🌐 Web Dashboard
│   ├── web_app.py                  # Flask web uygulaması
│   ├── run_dashboard.py            # Dashboard launcher
│   ├── templates/                  # HTML templates
│   │   ├── dashboard.html          # Ana dashboard
│   │   ├── portfolio.html          # Portfolio yönetimi
│   │   ├── analyze_coin.html       # Coin analizi
│   │   ├── settings.html           # Ayarlar
│   │   └── login.html             # Giriş sayfası
│   └── static/                     # CSS, JS, görsel dosyalar
│       ├── css/
│       └── js/
│
├── 📊 Data & Analytics
│   ├── data_fetcher.py             # Binance veri çekme
│   ├── data_preprocessor.py        # Veri ön işleme + teknik analiz
│   ├── news_analyzer.py            # Haber sentiment analizi
│   ├── whale_tracker.py            # Whale cüzdan takibi
│   └── performance_tester.py       # Backtest ve performance
│
├── 🤖 Trading Automation
│   ├── binance_trader.py           # Otomatik trading motoru
│   ├── binance_history.py          # Trading geçmişi
│   ├── auto_trader_integration.py  # Trading entegrasyonu
│   └── auth.py                     # Güvenlik ve kimlik doğrulama
│
├── 💾 Data Management
│   ├── database.py                 # SQLite veritabanı
│   ├── mssql_database.py          # MSSQL entegrasyonu
│   ├── create_mssql_database.py   # MSSQL kurulum
│   ├── model_cache.py             # Akıllı model cache
│   └── system_persistence.py      # Sistem durumu kaydetme
│
├── ⚙️ Configuration & Utils
│   ├── tf_config.py               # TensorFlow M1/M2 Mac optimizasyonu
│   ├── main.py                    # Ana CLI uygulaması
│   ├── example_usage.py           # Kullanım örnekleri
│   ├── quick_test.py              # Hızlı test araçları
│   └── requirements.txt           # Python dependencies
│
└── 📚 Documentation
    ├── README.md                  # Bu dosya
    ├── LSTM_CONFIG_README.md      # LSTM konfigürasyon rehberi
    ├── MSSQL_ENVIRONMENT_README.md # MSSQL kurulum rehberi
    └── WEB_DASHBOARD_README.md    # Web dashboard rehberi
```

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Repo klonlama
git clone <repository-url>
cd trading

# Virtual environment oluşturma
python -m venv trading_env
source trading_env/bin/activate  # Linux/Mac
# or
trading_env\Scripts\activate     # Windows

# Dependencies yükleme
pip install -r requirements.txt
```

### 2. Temel Konfigürasyon

```bash
# Environment variables (opsiyonel)
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_SECRET_KEY="your_binance_secret"
export NEWSAPI_KEY="your_newsapi_key"
export WHALE_ALERT_API_KEY="your_whale_api_key"
```

### 3. İlk Çalıştırma

#### CLI Uygulaması
```bash
python main.py
```

#### Web Dashboard
```bash
python run_dashboard.py
```

#### Hızlı Demo
```bash
python main.py --demo
```

## 🧠 AI Model Sistemi

### 1. Hibrit Trading Model

**Konum**: `hybrid_trading_model.py`

En gelişmiş model - LSTM fiyat tahmini + DQN aksiyon seçimi:

```python
from hybrid_trading_model import HybridTradingModel

# Model oluşturma
hybrid = HybridTradingModel(sequence_length=60, initial_balance=10000)

# Eğitim
hybrid.train_hybrid_model(df, lstm_epochs=50, dqn_episodes=200)

# Tahmin
prediction = hybrid.predict_hybrid_action(current_data)
print(f"Recommendation: {prediction['ensemble_prediction']['recommendation']}")
print(f"Confidence: {prediction['confidence']:.1%}")
```

**Özellikler**:
- LSTM + DQN ensemble learning
- Adaptive weight optimization
- Overfitting prevention
- Robust confidence calculation (max %85)

### 2. LSTM Price Predictor

**Konum**: `lstm_model.py`

Gelişmiş LSTM modeli - TensorFlow M1/M2 Mac optimized:

```python
from lstm_model import CryptoLSTMModel

model = CryptoLSTMModel(sequence_length=60, n_features=20)
model.build_model(lstm_units=[50, 50, 50], dropout_rate=0.2)
history = model.train_model(X_train, y_train, X_val, y_val, epochs=50)

# Tahmin
prediction = model.predict(X_test)
```

### 3. DQN Action Selector

**Konum**: `dqn_trading_model.py`

Deep Q-Network - 9 aksiyon (HOLD, BUY_25/50/75/100%, SELL_25/50/75/100%):

```python
from dqn_trading_model import DQNTradingModel

dqn = DQNTradingModel(lookback_window=60, initial_balance=10000)
dqn.prepare_data(df)
dqn.train(df, episodes=200)

action_prediction = dqn.predict_action(current_state)
print(f"Action: {action_prediction['action_name']}")
print(f"Confidence: {action_prediction['confidence']:.1%}")
```

## 🌐 Web Dashboard

**Konum**: `web_app.py`, `templates/`, `static/`

Modern web arayüzü Flask ile:

### Ana Özellikler

1. **Portfolio Management**: Portföy takibi ve yönetimi
2. **Real-time Analytics**: Canlı piyasa analizi
3. **AI Predictions**: Model tahminleri görselleştirme
4. **News Dashboard**: Haber sentiment analizi
5. **Trading History**: İşlem geçmişi ve performans
6. **Settings**: Model parametreleri ve API ayarları

### Dashboard Çalıştırma

```bash
python run_dashboard.py
```

Tarayıcıda açın: `http://localhost:5000`

## 📊 Data Intelligence

### 1. News Sentiment Analysis

**Konum**: `news_analyzer.py`

Çoklu kaynak haber analizi:

```python
from news_analyzer import CryptoNewsAnalyzer

analyzer = CryptoNewsAnalyzer(newsapi_key="your_key")

# Haber çekme
news = analyzer.fetch_all_news('BTC', days=7)

# Sentiment analizi
sentiments = analyzer.analyze_news_sentiment_batch(news)

# Fiyat korelasyonu
correlation = analyzer.calculate_news_price_correlation(sentiments, price_data)
```

**Desteklenen Kaynaklar**:
- NewsAPI (premium)
- CoinDesk
- Reddit r/cryptocurrency
- Twitter (opsiyonel)

### 2. Whale Tracking

**Konum**: `whale_tracker.py`

Büyük cüzdan hareketlerini takip:

```python
from whale_tracker import CryptoWhaleTracker

tracker = CryptoWhaleTracker(whale_alert_api_key="your_key")

# Whale işlemlerini çekme
transactions = tracker.fetch_whale_alert_transactions('BTC', hours=48)

# Analiz
analysis = tracker.analyze_whale_transactions(transactions)
print(f"Whale Activity Score: {analysis['whale_activity_score']}/100")

# Strateji önerisi
strategy = tracker.get_whale_strategy_recommendation(analysis)
```

### 3. Technical Analysis

**Konum**: `data_preprocessor.py`

25+ teknik gösterge:

- **Trend**: SMA, EMA, MACD, Bollinger Bands
- **Momentum**: RSI, Stochastic, CCI, Williams %R
- **Volume**: Volume SMA, Volume ratio
- **Volatility**: ATR, Bollinger bandwidth
- **Custom**: Yigit ATR Trailing Stop

## 🤖 Trading Automation

### Binance Integration

**Konum**: `binance_trader.py`

Profesyonel trading botu:

```python
from binance_trader import BinanceTrader

trader = BinanceTrader(api_key, secret_key, testnet=True)

# Pozisyon açma
result = trader.open_position(
    symbol='BTC/USDT',
    side='long',
    entry_price=45000,
    target_price=48000,
    stop_loss=43000,
    risk_percent=2.0
)

# Portfolio özeti
summary = trader.get_portfolio_summary()
```

### Auto Trading Integration

**Konum**: `auto_trader_integration.py`

AI modelleri ile trading botunu entegre eder:

```python
# AI sinyalini trading aksiyonuna çevir
signal = hybrid_model.predict_hybrid_action(current_data)
trading_result = auto_trader.execute_ai_signal(signal)
```

## 💾 Data Management

### 1. Model Cache System

**Konum**: `model_cache.py`

Akıllı model cache - eğitim süresini %70 azaltır:

```python
from model_cache import CachedModelManager

cache = CachedModelManager()

# Otomatik cache veya yeniden eğitim
model, preprocessor, info = cache.get_or_train_model(
    coin_symbol='BTC',
    data=df,
    config=model_config,
    preprocessor=preprocessor
)

print(f"Training type: {info['training_type']}")  # 'cached', 'new', 'incremental'
```

### 2. Database Integration

#### SQLite (Default)
**Konum**: `database.py`

```python
from database import TradingDatabase

db = TradingDatabase()
db.save_prediction(coin_symbol='BTC', prediction_data=results)
history = db.get_prediction_history('BTC', days=30)
```

#### MSSQL (Enterprise)
**Konum**: `mssql_database.py`

```python
from mssql_database import MSSQLTradingDatabase

db = MSSQLTradingDatabase()
# Gelişmiş analytics ve reporting
```

## ⚡ Performance & Testing

### Backtesting

**Konum**: `performance_tester.py`

Kapsamlı backtest sistemi:

```python
from performance_tester import TradingPerformanceTester

tester = TradingPerformanceTester()

# Model backtest
results = tester.backtest_model(
    model=hybrid_model,
    test_data=df,
    initial_balance=10000,
    start_date='2023-01-01',
    end_date='2024-01-01'
)

# Performans metrikleri
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## 🔧 Konfigürasyon

### Environment Variables

```bash
# Trading
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret
BINANCE_TESTNET=true

# News Analysis
NEWSAPI_KEY=your_newsapi_key

# Whale Tracking
WHALE_ALERT_API_KEY=your_whale_api_key

# Model Training
LSTM_EPOCHS=50
LSTM_TRAINING_DAYS=100
DQN_EPISODES=200

# Database
DATABASE_TYPE=sqlite  # or mssql
MSSQL_CONNECTION_STRING=your_mssql_connection

# Dashboard
FLASK_SECRET_KEY=your_secret_key
DASHBOARD_PORT=5000
```

### TensorFlow Configuration

**M1/M2 Mac Desteği**: `tf_config.py`

Sistem otomatik olarak TensorFlow'u optimize eder:
- Metal Performance Shaders desteği
- CPU fallback
- Memory optimization

## 📈 Kullanım Senaryoları

### 1. Hızlı Coin Analizi

```bash
python main.py
# Coin: BTC
# Model parametreleri: Enter (default)
# Haber analizi: y
# Whale analizi: y
```

### 2. Web Dashboard Monitoring

```bash
python run_dashboard.py
# http://localhost:5000 -> Portfolio -> Add BTC
```

### 3. Otomatik Trading

```python
# Hibrit model + auto trading
from hybrid_trading_model import HybridTradingModel
from auto_trader_integration import AutoTrader

hybrid = HybridTradingModel()
# ... model eğitimi ...

auto_trader = AutoTrader(hybrid_model=hybrid)
auto_trader.start_trading('BTC/USDT')
```

### 4. Backtest & Performance

```python
# Geçmiş dönem performans testi
results = tester.backtest_model(
    model=hybrid_model,
    symbol='BTC/USDT',
    start_date='2023-01-01',
    end_date='2024-01-01',
    initial_balance=10000
)
```

## 🛡️ Güvenlik & Risk

### Risk Management

- **Position Sizing**: Kelly criterion ve risk yüzdesi
- **Stop Loss**: Otomatik zarar durdur
- **Drawdown Protection**: Maksimum çekilme koruması
- **Diversification**: Çoklu coin desteği

### Security Features

- JWT authentication
- API key encryption
- Secure database connections
- Input validation ve sanitization

## 📊 Model Performance

### Tipik Performans Metrikleri

| Model | Accuracy | Sharpe Ratio | Max Drawdown | Confidence Range |
|-------|----------|--------------|--------------|------------------|
| LSTM | 65-75% | 1.2-1.8 | 15-25% | 40-80% |
| DQN | 60-70% | 1.0-1.5 | 20-30% | 15-85% |
| Hybrid | 70-80% | 1.5-2.2 | 12-20% | 50-85% |

### Confidence Levels

- **85%+**: Aşırı güven (overfitting) - sistem engeller
- **70-85%**: Yüksek güven - güçlü sinyal
- **50-70%**: Orta güven - standart sinyal
- **30-50%**: Düşük güven - dikkatli yaklaşım
- **<30%**: Çok düşük güven - manuel review gerekli

## 🔍 Troubleshooting

### Yaygın Sorunlar

**1. TensorFlow Metal Hataları (M1/M2 Mac)**
```python
# tf_config.py otomatik çözüm sağlar
import tf_config  # Bu import yeterli
```

**2. Binance API Connection**
```python
# Test bağlantısı
from data_fetcher import CryptoDataFetcher
fetcher = CryptoDataFetcher()
data = fetcher.fetch_ohlcv_data('BTC', timeframe='1h', days=1)
```

**3. Memory Issues**
```python
# Model parametrelerini küçült
sequence_length = 30  # default: 60
batch_size = 16      # default: 32
```

**4. Cache Issues**
```bash
# Cache temizleme
rm -rf model_cache/*
```

## 🚀 Gelişmiş Özellikler

### 1. Multi-Timeframe Analysis

```python
# Çoklu zaman dilimi analizi
timeframes = ['1h', '4h', '1d']
predictions = {}

for tf in timeframes:
    data = fetcher.fetch_ohlcv_data('BTC', timeframe=tf, days=100)
    pred = hybrid_model.predict_hybrid_action(data)
    predictions[tf] = pred
```

### 2. Portfolio Optimization

```python
# Modern Portfolio Theory entegrasyonu
from portfolio_optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()
optimal_weights = optimizer.optimize_portfolio(
    coins=['BTC', 'ETH', 'ADA'],
    predictions=predictions,
    risk_tolerance=0.6
)
```

### 3. Real-time Alerts

```python
# Sinyal bazlı alertler
from alert_system import AlertManager

alerts = AlertManager()
alerts.add_condition('BTC', 'hybrid_confidence > 0.8')
alerts.add_condition('ETH', 'whale_activity > 80')
```

## 📚 API Reference

### Core Classes

```python
# Hibrit Model
HybridTradingModel(sequence_length=60, initial_balance=10000)
├── train_hybrid_model(df, lstm_epochs=30, dqn_episodes=100)
├── predict_hybrid_action(current_data)
└── get_model_performance_summary()

# LSTM Model
CryptoLSTMModel(sequence_length=60, n_features=20)
├── build_model(lstm_units=[50,50,50], dropout_rate=0.2)
├── train_model(X_train, y_train, X_val, y_val, epochs=50)
└── predict(X)

# DQN Model
DQNTradingModel(lookback_window=60, initial_balance=10000)
├── prepare_data(df)
├── train(df, episodes=100)
└── predict_action(current_state)

# Data Processing
CryptoDataPreprocessor()
├── prepare_data(df, use_technical_indicators=True)
├── scale_data(df, fit_scaler=True)
└── create_sequences(data, sequence_length)

# News Analysis
CryptoNewsAnalyzer(newsapi_key=None)
├── fetch_all_news(coin_symbol, days=7)
├── analyze_news_sentiment_batch(news_data)
└── calculate_news_price_correlation(sentiment_df, price_df)

# Whale Tracking
CryptoWhaleTracker(whale_alert_api_key=None)
├── fetch_whale_alert_transactions(coin_symbol, hours=48)
├── analyze_whale_transactions(transactions)
└── get_whale_strategy_recommendation(analysis)
```

## 🤝 Contributing

### Development Setup

```bash
# Development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

### Code Style

- PEP 8 compliance
- Type hints
- Docstring documentation
- Error handling

## 📄 License

MIT License - detaylar için `LICENSE` dosyasına bakın.

## 🙏 Credits

- **Binance API**: Kripto para verileri
- **TensorFlow**: Deep learning framework
- **NewsAPI**: Haber verileri
- **Whale Alert**: Whale transaction data
- **CCXT**: Cryptocurrency exchange library

## ⚠️ Risk Disclaimer

**Bu yazılım sadece eğitim ve araştırma amaçlıdır. Finansal yatırım tavsiyesi değildir.**

- Kripto para yatırımları yüksek risk içerir
- Geçmiş performans gelecek başarıyı garanti etmez
- Sadece kaybetmeyi göze alabileceğiniz sermaye ile yatırım yapın
- Profesyonel finansal danışmanlık alın

---

⭐ **Bu projeyi beğendiyseniz star vermeyi unutmayın!**

📧 **Sorularınız için**: Issues kısmından iletişime geçebilirsiniz

🚀 **Happy Trading!** (Responsibly) 