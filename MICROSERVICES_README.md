# 🚀 Crypto Trading AI System - Mikroservis Mimarisi

Bu proje **AI Service** ve **Web Service** olmak üzere iki ayrı servise bölünmüştür.

## 🏗️ **Mimari Genel Bakış**

```
┌─────────────────────┐     HTTP/REST API     ┌─────────────────────┐
│   WEB SERVICE       │ ←──────────────────→ │    AI SERVICE       │
│   (Port: 5000)      │                       │   (Port: 8000)      │
│                     │                       │                     │
│ • Flask Web App     │                       │ • FastAPI           │
│ • Dashboard UI      │                       │ • LSTM Models       │
│ • Authentication    │                       │ • DQN Models        │
│ • Database          │                       │ • Hybrid Models     │
│ • Portfolio         │                       │ • Data Processing   │
│ • User Management   │                       │ • Model Training    │
└─────────────────────┘                       └─────────────────────┘
```

## 🤖 **AI Service (Port: 8000)**

### Özellikler
- **FastAPI** tabanlı yüksek performanslı API
- **LSTM, DQN, Hybrid** model eğitimi ve tahmin
- **Asynchronous** model eğitimi
- **News & Whale** analizi
- **Model cache** yönetimi
- **Multi-timeframe** tahminler (4h, 1d)

### Kurulum
```bash
cd ai-service

# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # MacOS/Linux
# veya
venv\Scripts\activate     # Windows

# Dependencies yükle
pip install -r requirements.txt

# Environment variables konfigüre et
cp .env.example .env
# .env dosyasını düzenle

# AI Service'i başlat
python run_ai_service.py
```

### API Endpoints

#### 1. Health Check
```bash
GET http://localhost:8000/health
```

#### 2. Coin Analizi
```bash
POST http://localhost:8000/analyze
Content-Type: application/json

{
  "coin_symbol": "BTC",
  "analysis_type": "comprehensive",
  "use_news": true,
  "use_whale": true
}
```

#### 3. Model Eğitimi
```bash
POST http://localhost:8000/train
Content-Type: application/json

{
  "coin_symbol": "BTC",
  "training_type": "comprehensive",
  "is_fine_tune": false,
  "epochs": 50,
  "data_days": 100
}
```

#### 4. Model Durumu
```bash
GET http://localhost:8000/models/BTC
```

### Konfigürasyon (.env)
```bash
# AI Service
AI_SERVICE_HOST=0.0.0.0
AI_SERVICE_PORT=8000
AI_SERVICE_DEBUG=True

# Model Training
LSTM_EPOCHS=50
LSTM_TRAINING_DAYS=100
DQN_EPISODES=200

# External APIs
NEWSAPI_KEY=your_newsapi_key
WHALE_ALERT_API_KEY=your_whale_alert_key
```

## 🌐 **Web Service (Port: 5000)**

### Özellikler
- **Flask** tabanlı web dashboard
- **AI Service** ile HTTP iletişimi
- **User authentication**
- **Portfolio management**
- **Real-time** coin monitoring
- **Database** integration (SQLite/MSSQL)

### Kurulum
```bash
cd web-service

# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # MacOS/Linux
# veya
venv\Scripts\activate     # Windows

# Dependencies yükle
pip install -r requirements.txt

# Environment variables konfigüre et
cp .env.example .env
# .env dosyasını düzenle

# Web Service'i başlat
python run_web_service.py
```

### Konfigürasyon (.env)
```bash
# Web Service
WEB_HOST=0.0.0.0
WEB_PORT=5000
WEB_DEBUG=True
SECRET_KEY=your_secret_key

# AI Service Connection
AI_SERVICE_URL=http://localhost:8000
AI_SERVICE_API_KEY=your_api_key

# Database (Optional)
MSSQL_SERVER=your_server
MSSQL_DATABASE=TradingDB
MSSQL_USERNAME=your_username
MSSQL_PASSWORD=your_password

# Binance
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

## 🚀 **Başlatma Sırası**

### 1. AI Service'i Başlat
```bash
cd ai-service
python run_ai_service.py
```
✅ AI Service: http://localhost:8000

### 2. Web Service'i Başlat
```bash
cd web-service
python run_web_service.py
```
✅ Web Dashboard: http://localhost:5000

## 🔧 **Geliştirme Ortamı**

### Ayrı Makinelerde Çalıştırma

#### Makine 1: AI Service
```bash
# AI Service
cd ai-service
export AI_SERVICE_HOST=0.0.0.0
export AI_SERVICE_PORT=8000
python run_ai_service.py
```

#### Makine 2: Web Service
```bash
# Web Service
cd web-service
export AI_SERVICE_URL=http://AI_MACHINE_IP:8000
export WEB_HOST=0.0.0.0
export WEB_PORT=5000
python run_web_service.py
```

## 🐳 **Docker Deployment**

### AI Service Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY ai-service/ .
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "run_ai_service.py"]
```

### Web Service Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY web-service/ .
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "run_web_service.py"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  ai-service:
    build: ./ai-service
    ports:
      - "8000:8000"
    environment:
      - AI_SERVICE_HOST=0.0.0.0
      - AI_SERVICE_PORT=8000
    volumes:
      - ./model_cache:/app/model_cache

  web-service:
    build: ./web-service
    ports:
      - "5000:5000"
    environment:
      - WEB_HOST=0.0.0.0
      - WEB_PORT=5000
      - AI_SERVICE_URL=http://ai-service:8000
    depends_on:
      - ai-service
```

## 📊 **API Test Scripts**

### AI Service Test
```python
import requests

# Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# Analyze coin
analyze_data = {
    "coin_symbol": "BTC",
    "analysis_type": "comprehensive"
}
response = requests.post('http://localhost:8000/analyze', json=analyze_data)
print(response.json())
```

### Web Service Test
```python
import requests

# Login
login_data = {'username': 'admin', 'password': 'admin'}
session = requests.Session()
response = session.post('http://localhost:5000/login', data=login_data)

# Analyze via Web API
analyze_data = {
    "coin_symbol": "BTC", 
    "analysis_type": "comprehensive"
}
response = session.post('http://localhost:5000/api/analyze_coin', json=analyze_data)
print(response.json())
```

## 🔒 **Güvenlik**

### API Key Authentication
AI Service için API key kullanın:
```bash
# AI Service
AI_SERVICE_API_KEY=your_secure_api_key

# Web Service
AI_SERVICE_API_KEY=your_secure_api_key
```

### HTTPS
Production'da HTTPS kullanın:
```bash
# Nginx reverse proxy ile
# SSL sertifikası ekleyin
```

## 📈 **Performans**

### AI Service Optimizasyonu
- **GPU** desteği için CUDA kurulumu
- **Model cache** kullanımı
- **Async** endpoint'ler
- **Batch processing**

### Web Service Optimizasyonu
- **Connection pooling**
- **Caching** (Redis)
- **Load balancing**
- **Database indexing**

## 🔍 **Troubleshooting**

### AI Service Bağlantı Hatası
```bash
# AI Service çalışıyor mu?
curl http://localhost:8000/health

# Port açık mı?
netstat -an | grep 8000
```

### Model Yükleme Hatası
```bash
# Model cache kontrol
ls -la ai-service/model_cache/

# Permissions kontrol
chmod -R 755 ai-service/model_cache/
```

### Web Service Hatası
```bash
# AI Service bağlantısı kontrol
curl http://localhost:8000/health

# Environment variables kontrol
echo $AI_SERVICE_URL
```

## 🎯 **Sonuç**

Bu mikroservis mimarisi ile:
- ✅ **Scalable** sistem
- ✅ **Independent** deployment
- ✅ **High performance** AI processing
- ✅ **Flexible** web interface
- ✅ **Easy maintenance**

Her servis kendi sorumluluğunda çalışır ve birbirinden bağımsız olarak geliştirilebilir!
