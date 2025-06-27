# 🚀 Comprehensive Training System Guide

Bu guide, yeni eklenen **Comprehensive Training System** ve **Weekly Fine-Tune Scheduler** özelliklerini açıklar.

## 🎯 Özet Değişiklikler

### 1. **Yeni Coin Ekleme Davranışı**
- Coin eklendiğinde **tüm algoritmalar** (LSTM, DQN, Hybrid) **aynı anda** eğitilir
- **4 saatlik** ve **1 günlük** tahminler oluşturulur
- Coin otomatik olarak **haftalık fine-tune schedule**'a eklenir

### 2. **Haftalık Fine-Tune Scheduler**
- Her hafta belirtilen gün/saatte otomatik fine-tune
- Mevcut cache edilmiş modellerin üzerine fine-tune
- Manual training de mümkün

### 3. **Multi-Timeframe Predictions**
- **4h tahmin**: Bir sonraki 4 saatlik kapanış
- **1d tahmin**: 24 saat sonraki (6 x 4h) kapanış
- Her model için ayrı tahminler

## 📁 Yeni Dosyalar

### `comprehensive_trainer.py`
```python
from comprehensive_trainer import ComprehensiveTrainer

trainer = ComprehensiveTrainer()
result = trainer.train_coin_sync('BTC', is_fine_tune=False)
```

**Özellikler:**
- LSTM, DQN, Hybrid modelleri paralel eğitir
- 4h ve 1d tahminler oluşturur
- Model cache'leme ve performans tracking
- Fine-tune mode desteği

### `training_scheduler.py`
```python
from training_scheduler import TrainingScheduler

scheduler = TrainingScheduler(
    schedule_day="sunday",
    schedule_time="02:00"
)
scheduler.start_scheduler()
```

**Özellikler:**
- Haftalık otomatik fine-tune
- Manual training trigger
- Coin ekleme/çıkarma
- JSON ile state persistence

### `test_comprehensive_system.py`
Sistemin çalışıp çalışmadığını test eden script.

```bash
python test_comprehensive_system.py
```

## 🔧 Environment Variables

### Training Konfigürasyonu
```bash
# Model Training
LSTM_EPOCHS=50                    # LSTM epoch sayısı
LSTM_TRAINING_DAYS=100           # Eğitim için kaç günlük data
DQN_EPISODES=200                 # DQN episode sayısı

# Scheduler
TRAINING_SCHEDULE_DAY=sunday     # Haftalık training günü
TRAINING_SCHEDULE_TIME=02:00     # Haftalık training saati

# News & Whale Analysis
NEWSAPI_KEY=your_newsapi_key
WHALE_ALERT_API_KEY=your_whale_key
```

## 🌐 Web Interface Değişiklikleri

### Coin Ekleme Formu
Artık coin eklerken:
- ✅ **Comprehensive Training**: Tüm modelleri eğitir (önerilen)
- ⚠️ **Normal Analysis**: Sadece LSTM

### API Endpoints

#### Scheduler Status
```javascript
GET /api/scheduler_status
{
  "success": true,
  "scheduler_status": {
    "is_running": true,
    "schedule_day": "sunday",
    "schedule_time": "02:00",
    "tracked_coins_count": 5,
    "tracked_coins": ["BTC", "ETH", "BNB", "ADA", "SOL"],
    "next_run_time": "2024-01-14 02:00:00"
  }
}
```

#### Scheduler Control
```javascript
POST /api/scheduler_control
{
  "action": "force_run",      // start, stop, force_run, add_coin, remove_coin
  "coin_symbol": "BTC"        // opsiyonel
}
```

### WebSocket Events

#### Comprehensive Training Complete
```javascript
socket.on('comprehensive_training_complete', function(data) {
  console.log('Training complete:', data.coin);
  console.log('Successful models:', data.successful_models);
  console.log('4h predictions:', data.predictions_4h);
  console.log('1d predictions:', data.predictions_1d);
});
```

## 🔄 Workflow

### 1. Yeni Coin Ekleme
```
Kullanıcı coin ekler
    ↓
Comprehensive Training başlar (background)
    ↓
LSTM + DQN + Hybrid paralel eğitim
    ↓
4h ve 1d tahminler oluşturulur
    ↓
Sonuçlar cache'lenir ve DB'ye kaydedilir
    ↓
Coin haftalık schedule'a eklenir
    ↓
WebSocket ile sonuç bildirilir
```

### 2. Haftalık Fine-Tune
```
Pazar 02:00 (varsayılan)
    ↓
Schedule'daki her coin için:
  - Mevcut model var mı kontrol
  - Varsa: Fine-tune mode
  - Yoksa: İlk eğitim mode
    ↓
Tüm modeller güncellenir
    ↓
Yeni tahminler oluşturulur
    ↓
Sonuçlar raporlanır
```

## 📊 Model Training Stratejisi

### İlk Eğitim
```python
config = {
    'data_days': 1000,        # 1000 gün veri
    'lstm_epochs': 100,       # 100 epoch
    'dqn_episodes': 150,      # 150 episode
    'hybrid_lstm_epochs': 80,
    'hybrid_dqn_episodes': 120
}
```

### Fine-Tune
```python
config = {
    'data_days': 200,         # 200 gün veri
    'lstm_epochs': 30,        # 30 epoch
    'dqn_episodes': 50,       # 50 episode
    'hybrid_lstm_epochs': 25,
    'hybrid_dqn_episodes': 40
}
```

## 💾 Dosya Yapısı

```
trading/
├── comprehensive_trainer.py        # Ana training sistemi
├── training_scheduler.py          # Haftalık scheduler
├── test_comprehensive_system.py   # Test script

├── model_cache/                   # Model cache'leri
│   ├── lstm_btc_comprehensive.h5
│   ├── dqn_btc_comprehensive.h5
│   └── hybrid_btc_comprehensive/

├── training_results/              # Training sonuçları
│   └── BTC_20240101_120000.json

├── scheduler_data/               # Scheduler verileri
│   ├── tracked_coins.json
│   └── weekly_results/
│       └── weekly_training_20240107_020000.json

└── requirements.txt              # schedule==1.2.0 eklendi
```

## 🧪 Test Etme

### 1. System Test
```bash
python test_comprehensive_system.py
```

### 2. Manual Training Test
```bash
python -c "
from comprehensive_trainer import ComprehensiveTrainer
trainer = ComprehensiveTrainer()
result = trainer.train_coin_sync('BTC', is_fine_tune=False)
print('Success:', result['success'])
"
```

### 3. Scheduler Test
```bash
python -c "
from training_scheduler import TrainingScheduler
scheduler = TrainingScheduler()
scheduler.add_coin_to_schedule('BTC')
status = scheduler.get_scheduler_status()
print('Tracked coins:', status['tracked_coins'])
"
```

## 🎛️ Web Dashboard Kullanımı

### Coin Ekleme
1. Dashboard'da **"Add Coin"** butonuna tık
2. Coin sembolünü gir (örn: BTC)
3. **✅ Comprehensive Training** seçili bırak
4. **"Add"** butonuna tık
5. Background'da eğitim başlar (birkaç dakika sürer)
6. WebSocket bildirimleri ile ilerleme takibi

### Scheduler Kontrol
1. Settings sayfasına git
2. **"Training Scheduler"** bölümünde:
   - Scheduler durumunu gör
   - Manual training başlat
   - Takip edilen coinleri yönet

### Sonuçları Görme
1. Coin analiz sayfasında:
   - **4h Prediction**: Sonraki 4 saatlik tahmin
   - **1d Prediction**: 24 saatlik tahmin
   - **Model Comparison**: LSTM vs DQN vs Hybrid
   - **Ensemble Recommendation**: Birleşik öneri

## ⚠️ Önemli Notlar

### Performance
- **İlk eğitim**: 5-15 dakika (model ve veri boyutuna göre)
- **Fine-tune**: 2-5 dakika
- **Memory**: ~2-4GB RAM gerekli (model boyutuna göre)

### Disk Kullanımı
- Model cache: ~100-500MB per coin
- Training results: ~1-10MB per training
- Scheduler data: ~1MB

### Best Practices
1. **İlk eğitim** için comprehensive training kullan
2. **Haftalık schedule** Pazar sabahına ayarla
3. **Environment variables** ile konfigürasyon yap
4. **Test script** ile sistem health check yap
5. **Model cache** dizinini düzenli temizle

## 🔧 Troubleshooting

### Training Başarısız
```bash
# Log kontrol
tail -f training_scheduler.log

# Manual test
python test_comprehensive_system.py

# Model cache temizle
rm -rf model_cache/*
```

### Scheduler Çalışmıyor
```bash
# Environment kontrol
echo $TRAINING_SCHEDULE_DAY
echo $TRAINING_SCHEDULE_TIME

# Manual başlat
python -c "
from training_scheduler import get_scheduler
scheduler = get_scheduler()
scheduler.start_scheduler()
"
```

### Web Interface Sorunları
```bash
# WebSocket bağlantı kontrol
# Browser console'da:
# socket.connected

# API test
curl http://localhost:5000/api/scheduler_status
```

## 📈 Gelecek Geliştirmeler

1. **Multi-GPU support** - Paralel eğitim için
2. **Advanced scheduling** - Farklı coinler için farklı schedule
3. **Model ensemble voting** - Daha akıllı ensemble
4. **Performance monitoring** - Model drift detection
5. **Auto-scaling** - Cloud deployment için

---

## 🎉 Özet

Bu sistem ile:
- ✅ Coin eklediğinizde **TÜM algoritmalar eğitilir**
- ✅ **4h ve 1d tahminler** otomatik oluşturulur
- ✅ **Haftalık fine-tune** otomatik çalışır
- ✅ **Cache sistemi** ile hızlı re-training
- ✅ **Web interface** ile kolay yönetim
- ✅ **API endpoints** ile programmatic kontrol

**Artık coin eklemek = Comprehensive AI sistemi eğitmek! 🚀** 