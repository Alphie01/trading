# 🔧 LSTM Model Konfigürasyon Kılavuzu

Bu kılavuz, LSTM modelinin gün sayısı, epoch sayısı ve diğer parametrelerini `.env` dosyası ile nasıl yapılandıracağınızı açıklar.

## 📋 Environment Variables (.env dosyası)

Proje klasörünüzde `.env` dosyası oluşturun ve aşağıdaki parametreleri ekleyin:

```env
# ===== LSTM EĞİTİM YAPILANDIRMASI =====
# LSTM modeli için kullanılacak gün sayısı
LSTM_TRAINING_DAYS=100

# Haber analizi için kullanılacak gün sayısı  
NEWS_ANALYSIS_DAYS=7

# LSTM eğitimi için epoch sayısı
LSTM_EPOCHS=30

# ===== DİĞER API ANAHTARLARI =====
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
NEWSAPI_KEY=your_newsapi_key_here
WHALE_ALERT_API_KEY=your_whale_alert_key_here
```

## 🎯 Farklı Senaryolar İçin Önerilen Ayarlar

### 🚀 **Hızlı Test (5-10 dakika)**
```env
LSTM_TRAINING_DAYS=30
NEWS_ANALYSIS_DAYS=3
LSTM_EPOCHS=15
```
- ✅ Çok hızlı
- ❌ Düşük accuracy

### ⚡ **Dengeli Ayar (15-30 dakika)**
```env
LSTM_TRAINING_DAYS=100
NEWS_ANALYSIS_DAYS=7
LSTM_EPOCHS=30
```
- ✅ İyi denge
- ✅ Makul süre
- ✅ Kabul edilebilir accuracy

### 🎯 **Yüksek Accuracy (1-2 saat)**
```env
LSTM_TRAINING_DAYS=200
NEWS_ANALYSIS_DAYS=14
LSTM_EPOCHS=100
```
- ✅ En yüksek accuracy
- ❌ Uzun süre

### 💎 **Profesyonel Trading (2-4 saat)**
```env
LSTM_TRAINING_DAYS=300
NEWS_ANALYSIS_DAYS=21
LSTM_EPOCHS=150
```
- ✅ En iyi sonuçlar
- ✅ Güvenilir tahminler
- ❌ En uzun süre

## 🪙 Coin Türüne Göre Öneriler

### **Bitcoin/Ethereum (Ana Coinler)**
```env
LSTM_TRAINING_DAYS=200  # Daha stabil, uzun veri faydalı
LSTM_EPOCHS=50          # Daha fazla epoch gerekir
```

### **Altcoin'ler (ADA, DOT, LINK vb.)**
```env
LSTM_TRAINING_DAYS=100  # Orta vadeli veri yeterli
LSTM_EPOCHS=30          # Standart epoch
```

### **Meme Coin'ler (DOGE, SHIB, WIF vb.)**
```env
LSTM_TRAINING_DAYS=60   # Kısa vadeli, volatil
LSTM_EPOCHS=20          # Az epoch yeterli
```

### **Yeni Listenen Coinler**
```env
LSTM_TRAINING_DAYS=30   # Az tarihsel veri mevcut
LSTM_EPOCHS=15          # Hızlı eğitim
```

## 📊 Model Cache Sistemi

Model cache sistemi sayesinde:
- **İlk eğitim**: Belirtilen epoch sayısı kadar eğitilir
- **Sonraki çalıştırmalar**: Model cache'den yüklenir (çok hızlı)
- **Incremental Training**: Eğer model 7 günden eskiyse, az epoch ile güncellenip

## 🔧 Ayarları Test Etme

1. **Hızlı test için:**
   ```bash
   python3 quick_test.py
   ```

2. **Ana sistem ile test:**
   ```bash
   python3 main.py
   ```

3. **Web dashboard ile:**
   ```bash
   python3 run_dashboard.py
   ```

## 📈 Performance İzleme

WIF modelindeki gibi training sırasında görebileceğiniz metrikler:

```
Epoch 30/30
loss: 0.0914 - mae: 0.2277 - mape: 838242.3125 - directional_accuracy: 0.8194
val_loss: 0.0395 - val_mae: 0.1790 - val_mape: 25.7350 - val_directional_accuracy: 0.4713

Model Değerlendirme Sonuçları:
  DIRECTIONAL_ACCURACY: 0.547821  # Fiyat yönü tahmin doğruluğu (%54.7)
  MAPE: 51.353361                  # Ortalama mutlak yüzde hatası
```

## ⚠️ Önemli Notlar

1. **Gün sayısı arttıkça:**
   - ✅ Daha fazla veri = Daha iyi tahmin
   - ❌ Daha uzun veri çekme süresi

2. **Epoch sayısı arttıkça:**
   - ✅ Daha iyi öğrenme
   - ❌ Daha uzun eğitim süresi
   - ⚠️ Çok fazla epoch = Overfitting riski

3. **Haber analizi günü arttıkça:**
   - ✅ Daha kapsamlı sentiment analizi
   - ❌ Daha fazla API çağrısı

## 🛠️ Sorun Giderme

**Model çok yavaş eğitiliyor:**
```env
LSTM_EPOCHS=15
LSTM_TRAINING_DAYS=50
```

**Accuracy çok düşük:**
```env
LSTM_EPOCHS=100
LSTM_TRAINING_DAYS=200
```

**Memory hatası alıyorsanız:**
```env
LSTM_TRAINING_DAYS=60
LSTM_EPOCHS=20
```

**NewsAPI limiti aşıldı:**
```env
NEWS_ANALYSIS_DAYS=3
``` 