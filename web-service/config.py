# Web Service Configuration
import os
from dotenv import load_dotenv

load_dotenv()

class WebConfig:
    # Flask Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'trading-dashboard-secret-key-change-in-production')
    WEB_HOST = os.getenv('WEB_HOST', '0.0.0.0')
    WEB_PORT = int(os.getenv('WEB_PORT', 25629))
    WEB_DEBUG = os.getenv('WEB_DEBUG', 'True').lower() == 'true'
    
    # AI Service Settings
    AI_SERVICE_URL = os.getenv('AI_SERVICE_URL', 'http://localhost:8000')
    AI_SERVICE_API_KEY = os.getenv('AI_SERVICE_API_KEY')
    AI_SERVICE_TIMEOUT = int(os.getenv('AI_SERVICE_TIMEOUT', 300))
    
    # Database Settings (PostgreSQL — SQLAlchemy trading_db paketi DATABASE_URL kullanır)
    DATABASE_URL = os.getenv('DATABASE_URL')

    # Binance Settings
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'True').lower() == 'true'
    
    # Security
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', SECRET_KEY)
    SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', 3600))  # 1 hour
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        errors = []
        
        # Check AI Service URL
        if not cls.AI_SERVICE_URL:
            errors.append("AI_SERVICE_URL is required")

        # Database
        if not cls.DATABASE_URL:
            print("⚠️ DATABASE_URL tanımlı değil — trading_db bağlantısı başarısız olabilir")

        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        print("✅ Web configuration validated")
        return True

# Initialize config
web_config = WebConfig()
web_config.validate_config()
