# 🗄️ MSSQL Database & Environment Variables Kurulum Rehberi

Bu rehber kripto trading dashboard sisteminin **MSSQL Server** ve **Environment Variables** ile nasıl çalıştırılacağını açıklar.

## 📋 Genel Bakış

Sisteminiz artık şu özellikleri destekliyor:

- ✅ **MSSQL Server Integration** (45.141.151.4)
- ✅ **Environment Variables** (.env dosyası)
- ✅ **System State Persistence** (kaldığı yerden devam etme)
- ✅ **Automatic Session Resume**
- ✅ **Secure API Key Management**

## 🚀 Hızlı Başlangıç

### 1. Environment Variables Kurulumu

```bash
# .env dosyası oluştur
cp .env.example .env

# .env dosyasını düzenle
nano .env
```

**Önemli ayarlar:**
```bash
# Database (MSSQL Server)
MSSQL_SERVER=45.141.151.4
MSSQL_DATABASE=crypto_trading_db
MSSQL_USERNAME=sa
MSSQL_PASSWORD=YourStrongPassword123!

# Flask Security
FLASK_SECRET_KEY=your_super_secret_flask_key_here_change_this

# API Keys (İsteğe bağlı)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
NEWSAPI_KEY=your_newsapi_key
WHALE_ALERT_API_KEY=your_whale_alert_key
```

### 2. MSSQL Paketlerini Kur

```bash
# MSSQL için gerekli paketler
pip install pyodbc pymssql python-dotenv

# Tüm paketleri güncelle
pip install -r requirements.txt
```

### 3. Sistemi Test Et

```bash
# Environment ve MSSQL test
python test_env_mssql.py
```

### 4. Dashboard'u Başlat

```bash
# Otomatik kurulum kontrolü ile
python run_dashboard.py

# Veya doğrudan
python web_app.py
```

## 🗄️ MSSQL Server Detayları

### Bağlantı Bilgileri
- **Server:** 45.141.151.4
- **Port:** 1433 (default)
- **Database:** crypto_trading_db
- **Authentication:** SQL Server Authentication

### Tablolar
Sistem otomatik olarak şu tabloları oluşturur:

1. **coins** - Coin listesi ve durumları
2. **trades** - İşlem geçmişi
3. **positions** - Açık pozisyonlar
4. **analysis_results** - LSTM analiz sonuçları
5. **portfolio_summary** - Portfolio özetleri
6. **system_state** - Sistem durumu (persistence)
7. **monitoring_sessions** - İzleme session'ları

## 💾 System State Persistence

### Özellikler
- ✅ **Automatic Resume** - Sistem kapanıp açıldığında kaldığı yerden devam eder
- ✅ **Session Management** - Monitoring durumları kaydedilir
- ✅ **Coin List Persistence** - Aktif coin listesi korunur
- ✅ **Trading State** - Trading ayarları ve durumları
- ✅ **API Configuration** - API ayarları güvenli şekilde saklanır

### Dashboard'da Resume
Web interface'de sistem açıldığında:

```
🔄 Önceki session restore edilebilir!
   - Session ID: abc12345
   - Coin sayısı: 5
   - Monitoring interval: 15 dakika
   - Son aktivite: 2024-01-15 14:30:25

[Resume Previous Session] butonu
```

## 🔐 Environment Variables Listesi

### Gerekli (Required)
```bash
MSSQL_SERVER=45.141.151.4          # MSSQL server IP
MSSQL_DATABASE=crypto_trading_db    # Database adı
MSSQL_USERNAME=sa                   # Kullanıcı adı
MSSQL_PASSWORD=YourPassword         # Şifre
FLASK_SECRET_KEY=your_secret_key    # Flask güvenlik anahtarı
```

### İsteğe Bağlı (Optional)
```bash
# Binance Trading
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
BINANCE_TESTNET=true                # Test modunda çalışır

# News Analysis
NEWSAPI_KEY=your_newsapi_key
NEWSAPI_ENABLED=true

# Whale Tracking
WHALE_ALERT_API_KEY=your_api_key
WHALE_TRACKER_ENABLED=true

# Flask Settings
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=false

# Trading Settings
AUTO_TRADING_ENABLED=false
RISK_PERCENTAGE=2
MAX_LEVERAGE=10
MIN_TRADE_AMOUNT=10

# Monitoring
DEFAULT_MONITORING_INTERVAL=15
CONFIDENCE_THRESHOLD=75
```

## 🧪 Test ve Sorun Giderme

### Test Script Çalıştır
```bash
python test_env_mssql.py
```

**Test edilen özellikler:**
- Environment variables kurulumu
- MSSQL Server bağlantısı
- System persistence
- API key'lerin geçerliliği

### Yaygın Sorunlar

#### 1. MSSQL Bağlantı Hatası
```bash
❌ MSSQL bağlantısı başarısız!
```
**Çözüm:**
- MSSQL Server'ın çalıştığından emin olun
- Firewall ayarlarını kontrol edin
- Kullanıcı adı/şifre doğruluğunu kontrol edin

#### 2. pyodbc Import Hatası
```bash
❌ mssql_database modülü import edilemedi!
```
**Çözüm:**
```bash
pip install pyodbc
# Veya
pip install pymssql
```

#### 3. Environment Variables Eksik
```bash
❌ Eksik gerekli variables: MSSQL_PASSWORD
```
**Çözüm:**
- .env dosyasını kontrol edin
- Gerekli değerleri girin

## 🌐 Web Interface Özellikleri

### Dashboard Ana Sayfa
- 📊 **System Status** - Database type ve session bilgileri
- 🔄 **Resume Button** - Önceki session'ı restore etme
- 💾 **State Indicators** - Persistence durumu

### Settings Sayfası
- 🔑 **API Configuration** - API key'leri güvenli ayarlama
- ⚙️ **System Settings** - Monitoring ayarları
- 🗄️ **Database Info** - Database durumu

## 📱 Sistem Başlatma Akışı

1. **Environment Load** - .env dosyası yüklenir
2. **Database Selection** - MSSQL var mı kontrol edilir
3. **Connection Test** - MSSQL bağlantısı test edilir
4. **State Load** - Önceki sistem durumu yüklenir
5. **Resume Check** - Resume edilecek session var mı bakılır
6. **Web Start** - Flask uygulaması başlatılır

## 🔄 Otomatik Resume Akışı

### Sistem Kapanması
```python
# Monitoring durdur ve state kaydet
persistence.save_monitoring_state(
    is_active=False,
    interval_minutes=15,
    active_coins=['BTC', 'ETH'],
    session_info={
        'shutdown_reason': 'user_interrupt',
        'shutdown_time': '2024-01-15T14:30:25'
    }
)
```

### Sistem Açılması
```python
# State yükle ve resume kontrol et
monitoring_state = persistence.load_monitoring_state()
if monitoring_state['should_resume']:
    # Dashboard'da resume butonu göster
    show_resume_button(monitoring_state)
```

## 📞 Destek

Herhangi bir sorun yaşarsanız:

1. **Test Script Çalıştırın:** `python test_env_mssql.py`
2. **Logları Kontrol Edin:** `logs/trading_dashboard.log`
3. **Database Bağlantısını Test Edin:** MSSQL Management Studio ile
4. **Environment Variables'ı Kontrol Edin:** `.env` dosyası

## 🎯 Sonraki Adımlar

Sistem kurulumu tamamlandıktan sonra:

1. **API Key'leri Ayarlayın** - Settings sayfasından
2. **Test Coinleri Ekleyin** - BTC, ETH gibi
3. **Monitoring Başlatın** - 15 dakika aralıklarla
4. **Performance Test** - 24 saatlik test yapın
5. **Backup Schedule** - Otomatik backup ayarlayın

---

💡 **İpucu:** Sistem artık production-ready! MSSQL ile güvenilir, persistence ile sürekli çalışabilir.
