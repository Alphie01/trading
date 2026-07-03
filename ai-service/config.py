# AI Service Configuration
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # AI Service Settings
    AI_SERVICE_HOST = os.getenv('AI_SERVICE_HOST', '0.0.0.0')
    AI_SERVICE_PORT = int(os.getenv('AI_SERVICE_PORT', 8000))
    AI_SERVICE_DEBUG = os.getenv('AI_SERVICE_DEBUG', 'False').lower() == 'true'
    
    # Model Training Settings
    LSTM_EPOCHS = int(os.getenv('LSTM_EPOCHS', 50))
    LSTM_TRAINING_DAYS = int(os.getenv('LSTM_TRAINING_DAYS', 100))
    DQN_EPISODES = int(os.getenv('DQN_EPISODES', 200))
    
    # API Keys
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
    WHALE_ALERT_API_KEY = os.getenv('WHALE_ALERT_API_KEY')
    
    # Model Cache Settings
    MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', 'model_cache')
    
    # Training Scheduler
    TRAINING_SCHEDULE_DAY = os.getenv('TRAINING_SCHEDULE_DAY', 'sunday')
    TRAINING_SCHEDULE_TIME = os.getenv('TRAINING_SCHEDULE_TIME', '02:00')
    
    # Performance Settings
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    
    # Security
    API_KEY = os.getenv('AI_SERVICE_API_KEY')  # Optional API key for security
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        errors = []
        
        # Check if model cache directory exists
        if not os.path.exists(cls.MODEL_CACHE_DIR):
            os.makedirs(cls.MODEL_CACHE_DIR)
            print(f"✅ Model cache directory created: {cls.MODEL_CACHE_DIR}")
        
        # Validate API keys (optional)
        if not cls.NEWSAPI_KEY:
            print("⚠️ NEWSAPI_KEY not set - News analysis will be limited")
        
        if not cls.WHALE_ALERT_API_KEY:
            print("⚠️ WHALE_ALERT_API_KEY not set - Whale tracking will be limited")
            
        return len(errors) == 0

# Initialize config
config = Config()
config.validate_config()
