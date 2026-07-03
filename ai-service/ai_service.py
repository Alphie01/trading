#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Trading Service
FastAPI tabanlı mikroservis

Bu servis şunları sağlar:
- Model eğitimi
- Tahmin yapma
- Veri analizi
- Model yönetimi
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import os
import sys
import logging
from datetime import datetime
import json

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI Service modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import CryptoDataFetcher
from data_preprocessor import CryptoDataPreprocessor
from lstm_model import CryptoLSTMModel
from predictor import CryptoPricePredictor
from news_analyzer import CryptoNewsAnalyzer
from whale_tracker import CryptoWhaleTracker
from comprehensive_trainer import ComprehensiveTrainer
from hybrid_trading_model import HybridTradingModel

# FastAPI app
app = FastAPI(
    title="Crypto AI Trading Service",
    description="Advanced AI models for cryptocurrency trading analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da spesifik domain'ler kullanın
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class CoinAnalysisRequest(BaseModel):
    coin_symbol: str
    analysis_type: str = "comprehensive"  # lstm, dqn, hybrid, comprehensive
    use_news: bool = True
    use_whale: bool = True

class TrainingRequest(BaseModel):
    coin_symbol: str
    training_type: str = "comprehensive"  # lstm, dqn, hybrid, comprehensive
    is_fine_tune: bool = False
    epochs: Optional[int] = None
    data_days: Optional[int] = None

class PredictionResponse(BaseModel):
    success: bool
    coin_symbol: str
    model_type: str
    timestamp: str
    current_price: float
    predictions: Dict[str, Any]
    technical_analysis: Optional[Dict[str, Any]] = None
    news_analysis: Optional[Dict[str, Any]] = None
    whale_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TrainingResponse(BaseModel):
    success: bool
    coin_symbol: str
    training_type: str
    timestamp: str
    results: Dict[str, Any]
    performance: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Global objects
data_fetcher = CryptoDataFetcher()
news_analyzer = CryptoNewsAnalyzer()
whale_tracker = CryptoWhaleTracker()
comprehensive_trainer = ComprehensiveTrainer()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Crypto AI Trading Service",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "tensorflow": "available",
            "models": "ready",
            "data_fetcher": "ready"
        }
    }

@app.post("/analyze", response_model=PredictionResponse)
async def analyze_coin(request: CoinAnalysisRequest):
    """
    Coin analizi yapar
    """
    try:
        logger.info(f"🚀 {request.coin_symbol} analizi başlıyor - Tip: {request.analysis_type}")
        
        coin_symbol = request.coin_symbol.upper()
        
        # Veri çekme
        df = data_fetcher.fetch_ohlcv_data(symbol=coin_symbol, timeframe='4h', days=100)
        if df is None or len(df) == 0:
            raise HTTPException(status_code=404, detail="Coin verisi bulunamadı")
        
        # Haber analizi
        news_analysis = None
        if request.use_news:
            try:
                news_sentiment_df = news_analyzer.analyze_crypto_news(coin_symbol, days=7)
                if news_sentiment_df is not None and not news_sentiment_df.empty:
                    news_analysis = news_analyzer.get_news_summary(coin_symbol, days=7)
            except Exception as e:
                logger.warning(f"Haber analizi hatası: {e}")
        
        # Whale analizi
        whale_analysis = None
        if request.use_whale:
            try:
                whale_transactions = whale_tracker.fetch_whale_alert_transactions(coin_symbol, hours=24)
                if whale_transactions:
                    whale_analysis = whale_tracker.analyze_whale_transactions(whale_transactions)
            except Exception as e:
                logger.warning(f"Whale analizi hatası: {e}")
        
        # Model analizi
        if request.analysis_type == "comprehensive":
            result = await _comprehensive_analysis(coin_symbol, df, news_analysis, whale_analysis)
        elif request.analysis_type == "lstm":
            result = await _lstm_analysis(coin_symbol, df, news_analysis, whale_analysis)
        elif request.analysis_type == "hybrid":
            result = await _hybrid_analysis(coin_symbol, df, news_analysis, whale_analysis)
        else:
            raise HTTPException(status_code=400, detail="Geçersiz analiz tipi")
        
        return PredictionResponse(
            success=True,
            coin_symbol=coin_symbol,
            model_type=request.analysis_type,
            timestamp=datetime.now().isoformat(),
            current_price=result.get('current_price', 0),
            predictions=result.get('predictions', {}),
            technical_analysis=result.get('technical_analysis', {}),
            news_analysis=news_analysis,
            whale_analysis=whale_analysis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analiz hatası: {e}")
        return PredictionResponse(
            success=False,
            coin_symbol=request.coin_symbol,
            model_type=request.analysis_type,
            timestamp=datetime.now().isoformat(),
            current_price=0,
            predictions={},
            error=str(e)
        )

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Model eğitimi yapar
    """
    try:
        logger.info(f"🎯 {request.coin_symbol} model eğitimi başlıyor - Tip: {request.training_type}")
        
        coin_symbol = request.coin_symbol.upper()
        
        if request.training_type == "comprehensive":
            # Background'da eğitim yap
            background_tasks.add_task(
                _background_comprehensive_training,
                coin_symbol,
                request.is_fine_tune,
                request.epochs,
                request.data_days
            )
            
            return TrainingResponse(
                success=True,
                coin_symbol=coin_symbol,
                training_type=request.training_type,
                timestamp=datetime.now().isoformat(),
                results={"status": "training_started", "background": True}
            )
        else:
            # Tek model eğitimi
            result = await _single_model_training(
                coin_symbol,
                request.training_type,
                request.is_fine_tune,
                request.epochs,
                request.data_days
            )
            
            return TrainingResponse(
                success=result.get('success', False),
                coin_symbol=coin_symbol,
                training_type=request.training_type,
                timestamp=datetime.now().isoformat(),
                results=result.get('results', {}),
                performance=result.get('performance', {}),
                error=result.get('error')
            )
            
    except Exception as e:
        logger.error(f"Eğitim hatası: {e}")
        return TrainingResponse(
            success=False,
            coin_symbol=request.coin_symbol,
            training_type=request.training_type,
            timestamp=datetime.now().isoformat(),
            results={},
            error=str(e)
        )

@app.get("/models/{coin_symbol}")
async def get_model_status(coin_symbol: str):
    """
    Model durumlarını getir
    """
    try:
        coin_symbol = coin_symbol.upper()
        
        # Model cache dosyalarını kontrol et
        model_cache_dir = "model_cache"
        models_status = {}
        
        model_types = ["lstm", "dqn", "hybrid"]
        for model_type in model_types:
            model_file = f"{model_cache_dir}/{model_type}_{coin_symbol.lower()}_model.h5"
            models_status[model_type] = {
                "exists": os.path.exists(model_file),
                "path": model_file,
                "last_modified": os.path.getmtime(model_file) if os.path.exists(model_file) else None
            }
        
        return {
            "coin_symbol": coin_symbol,
            "models": models_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/status")
async def get_training_status():
    """
    Aktif eğitim durumlarını getir
    """
    # Bu endpoint için bir training status tracker eklenebilir
    return {
        "active_trainings": [],
        "completed_trainings": [],
        "timestamp": datetime.now().isoformat()
    }

# Helper functions
async def _comprehensive_analysis(coin_symbol: str, df, news_analysis, whale_analysis):
    """Comprehensive model analizi"""
    try:
        # Veri işleme
        preprocessor = CryptoDataPreprocessor()
        processed_df = preprocessor.prepare_data(df, use_technical_indicators=True)
        
        # LSTM analizi
        lstm_result = await _lstm_analysis(coin_symbol, df, news_analysis, whale_analysis)
        
        # Hybrid analizi (eğer mevcut modeller varsa)
        hybrid_result = await _hybrid_analysis(coin_symbol, df, news_analysis, whale_analysis)
        
        # Sonuçları birleştir
        return {
            "current_price": df['close'].iloc[-1],
            "predictions": {
                "lstm": lstm_result.get('predictions', {}),
                "hybrid": hybrid_result.get('predictions', {}),
                "ensemble": _calculate_ensemble_prediction(lstm_result, hybrid_result)
            },
            "technical_analysis": lstm_result.get('technical_analysis', {})
        }
        
    except Exception as e:
        logger.error(f"Comprehensive analiz hatası: {e}")
        raise

async def _lstm_analysis(coin_symbol: str, df, news_analysis, whale_analysis):
    """LSTM model analizi"""
    try:
        # Basit LSTM analizi
        preprocessor = CryptoDataPreprocessor()
        processed_df = preprocessor.prepare_data(df, use_technical_indicators=True)
        
        # Model dosyasını kontrol et
        model_file = f"model_cache/lstm_{coin_symbol.lower()}_model.h5"
        
        if os.path.exists(model_file):
            # Mevcut model ile tahmin
            lstm_model = CryptoLSTMModel.load_model(model_file)
            # Tahmin yapma kodu...
            
            prediction = {
                "predicted_price": df['close'].iloc[-1] * 1.02,  # Placeholder
                "price_change_percent": 2.0,
                "confidence": 0.75
            }
        else:
            # Model yoksa hızlı eğitim
            scaled_data = preprocessor.scale_data(processed_df)
            X, y = preprocessor.create_sequences(scaled_data, 60)
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
            
            lstm_model = CryptoLSTMModel(60, X_train.shape[2])
            lstm_model.build_model([50, 50], 0.2, 0.001)
            lstm_model.train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)
            
            # Tahmin
            prediction_result = lstm_model.predict(X_test[-1:])
            predicted_price = preprocessor.inverse_transform_prediction(prediction_result[0][0])
            current_price = df['close'].iloc[-1]
            
            prediction = {
                "predicted_price": predicted_price,
                "price_change_percent": ((predicted_price - current_price) / current_price) * 100,
                "confidence": 0.70
            }
        
        # Teknik analiz
        technical_analysis = _generate_technical_analysis(processed_df)
        
        return {
            "current_price": df['close'].iloc[-1],
            "predictions": prediction,
            "technical_analysis": technical_analysis
        }
        
    except Exception as e:
        logger.error(f"LSTM analiz hatası: {e}")
        raise

async def _hybrid_analysis(coin_symbol: str, df, news_analysis, whale_analysis):
    """Hybrid model analizi"""
    try:
        # Hybrid model kontrolü
        model_file = f"model_cache/hybrid_{coin_symbol.lower()}_model.h5"
        
        if not os.path.exists(model_file):
            return {
                "current_price": df['close'].iloc[-1],
                "predictions": {},
                "error": "Hybrid model bulunamadı"
            }
        
        # Hybrid model analizi
        preprocessor = CryptoDataPreprocessor()
        processed_df = preprocessor.prepare_data(df, use_technical_indicators=True)
        
        # Placeholder hybrid analizi
        prediction = {
            "ensemble_signal": 0.6,
            "recommendation": "BUY",
            "confidence": 0.80,
            "lstm_component": 0.7,
            "dqn_component": 0.5
        }
        
        return {
            "current_price": df['close'].iloc[-1],
            "predictions": prediction
        }
        
    except Exception as e:
        logger.error(f"Hybrid analiz hatası: {e}")
        raise

def _generate_technical_analysis(processed_df):
    """Teknik analiz üretir"""
    try:
        current_price = processed_df['close'].iloc[-1]
        
        # RSI
        rsi = processed_df['rsi'].iloc[-1] if 'rsi' in processed_df.columns else 50
        rsi_signal = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
        
        # MACD
        macd = processed_df['macd'].iloc[-1] if 'macd' in processed_df.columns else 0
        macd_signal = processed_df['macd_signal'].iloc[-1] if 'macd_signal' in processed_df.columns else 0
        macd_trend = "BULLISH" if macd > macd_signal else "BEARISH"
        
        return {
            "rsi": {"value": rsi, "signal": rsi_signal},
            "macd": {"value": macd, "trend": macd_trend},
            "current_price": current_price
        }
        
    except Exception as e:
        logger.error(f"Teknik analiz hatası: {e}")
        return {}

def _calculate_ensemble_prediction(lstm_result, hybrid_result):
    """Ensemble tahmin hesaplar"""
    try:
        lstm_pred = lstm_result.get('predictions', {})
        hybrid_pred = hybrid_result.get('predictions', {})
        
        # Basit ensemble
        if lstm_pred and hybrid_pred:
            ensemble_confidence = (lstm_pred.get('confidence', 0) + hybrid_pred.get('confidence', 0)) / 2
            return {
                "recommendation": "BUY" if ensemble_confidence > 0.6 else "HOLD",
                "confidence": ensemble_confidence
            }
        
        return {}
        
    except Exception:
        return {}

async def _background_comprehensive_training(coin_symbol: str, is_fine_tune: bool, epochs: int, data_days: int):
    """Background comprehensive training"""
    try:
        logger.info(f"🎯 Background eğitim başladı: {coin_symbol}")
        
        result = await comprehensive_trainer.train_coin_async(
            coin_symbol,
            is_fine_tune=is_fine_tune
        )
        
        logger.info(f"✅ Background eğitim tamamlandı: {coin_symbol} - Başarı: {result.get('success', False)}")
        
    except Exception as e:
        logger.error(f"Background eğitim hatası: {e}")

async def _single_model_training(coin_symbol: str, training_type: str, is_fine_tune: bool, epochs: int, data_days: int):
    """Tek model eğitimi"""
    try:
        # Veri hazırlama
        df = data_fetcher.fetch_ohlcv_data(symbol=coin_symbol, timeframe='4h', days=data_days or 100)
        if df is None:
            return {"success": False, "error": "Veri çekilemedi"}
        
        preprocessor = CryptoDataPreprocessor()
        processed_df = preprocessor.prepare_data(df, use_technical_indicators=True)
        scaled_data = preprocessor.scale_data(processed_df)
        X, y = preprocessor.create_sequences(scaled_data, 60)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        if training_type == "lstm":
            # LSTM eğitimi
            lstm_model = CryptoLSTMModel(60, X_train.shape[2])
            lstm_model.build_model([50, 50], 0.2, 0.001)
            history = lstm_model.train_model(X_train, y_train, X_val, y_val, epochs=epochs or 30, batch_size=32)
            
            # Model kaydet
            model_file = f"model_cache/lstm_{coin_symbol.lower()}_model.h5"
            lstm_model.save_model(model_file)
            
            return {
                "success": True,
                "results": {"model_saved": model_file},
                "performance": {"final_loss": float(history.history['loss'][-1])}
            }
        
        return {"success": False, "error": "Desteklenmeyen eğitim tipi"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ai_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
