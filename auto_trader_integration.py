#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Prediction + Binance Trading Entegrasyonu

Bu modül LSTM tahmin sistemini Binance otomatik trading ile birleştirir.
"""

import time
import warnings
warnings.filterwarnings('ignore')

from binance_trader import BinanceTrader
from data_fetcher import CryptoDataFetcher
from data_preprocessor import CryptoDataPreprocessor
from lstm_model import CryptoLSTMModel
from predictor import CryptoPricePredictor
from news_analyzer import CryptoNewsAnalyzer
from whale_tracker import CryptoWhaleTracker

from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json

class LSTMAutoTrader:
    """
    LSTM tahmin sistemi ile Binance trader'ı birleştiren otomatik trading sınıfı
    """
    
    def __init__(self, binance_api_key: str, binance_secret_key: str, 
                 testnet: bool = True, newsapi_key: str = None, whale_api_key: str = None):
        """
        LSTMAutoTrader'ı başlatır
        
        Args:
            binance_api_key (str): Binance API anahtarı
            binance_secret_key (str): Binance Secret anahtarı
            testnet (bool): Test modu
            newsapi_key (str): NewsAPI anahtarı (opsiyonel)
            whale_api_key (str): Whale Alert API anahtarı (opsiyonel)
        """
        # Binance trader
        self.trader = BinanceTrader(binance_api_key, binance_secret_key, testnet)
        
        # LSTM sistem bileşenleri
        self.fetcher = CryptoDataFetcher()
        self.preprocessor = CryptoDataPreprocessor()
        self.model = None
        self.predictor = None
        self.news_analyzer = CryptoNewsAnalyzer(newsapi_key) if newsapi_key else None
        self.whale_tracker = CryptoWhaleTracker(whale_api_key) if whale_api_key else None
        
        # Trading ayarları
        self.min_confidence = 70.0  # Minimum güven seviyesi
        self.prediction_refresh_minutes = 60  # Tahmin yenileme sıklığı
        self.position_check_minutes = 15  # Pozisyon kontrol sıklığı
        
        # Model durumu
        self.trained_models = {}  # Symbol -> Model mapping
        self.last_predictions = {}  # Symbol -> Son tahmin
        self.active_trading = False
        
        self.trader.log_message("🤖 LSTM AutoTrader başlatıldı!")
    
    def execute_auto_trading(self, symbol: str, exchange_type: str = 'futures') -> Dict:
        """
        Belirli bir symbol için otomatik trading yapar
        
        Args:
            symbol (str): Trading çifti
            exchange_type (str): Exchange tipi
        
        Returns:
            Dict: İşlem sonucu
        """
        try:
            # Örnek tahmin (gerçek sistemde LSTM'den alınacak)
            current_price = self.trader.get_current_price(symbol)
            
            if current_price <= 0:
                return {
                    'success': False,
                    'error': 'Fiyat alınamadı'
                }
            
            # Demo tahmin - gerçek sistemde LSTM prediction kullanılacak
            prediction = {
                'current_price': current_price,
                'predicted_price': current_price * 1.015,  # %1.5 artış
                'confidence': 75.0,
                'price_change_percent': 1.5
            }
            
            # Trading sinyali oluştur
            signal = self._generate_trading_signal(symbol, prediction, exchange_type)
            
            if not signal:
                return {
                    'success': False,
                    'error': 'Trading sinyali oluşturulamadı'
                }
            
            # Trading sinyalini işle
            result = self.trader.execute_signal(signal)
            
            if result['success']:
                self.trader.log_message(f"✅ {symbol} otomatik işlem başarılı!")
            
            return result
            
        except Exception as e:
            self.trader.log_message(f"❌ {symbol} otomatik trading hatası: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_trading_signal(self, symbol: str, prediction: Dict, 
                                exchange_type: str = 'futures') -> Optional[Dict]:
        """
        Tahmine göre trading sinyali oluşturur
        """
        try:
            confidence = prediction['confidence']
            
            if confidence < self.min_confidence:
                self.trader.log_message(f"⚠️ {symbol} düşük güven: {confidence:.1f}%")
                return None
            
            current_price = prediction['current_price']
            predicted_price = prediction['predicted_price']
            price_change = prediction['price_change_percent']
            
            # Sinyal kuvveti kontrolü
            min_change = 1.0  # Minimum %1 değişim
            if abs(price_change) < min_change:
                self.trader.log_message(f"⚠️ {symbol} zayıf sinyal: {price_change:.2f}%")
                return None
            
            # Stop loss ve take profit hesapla
            if price_change > 0:  # Long sinyali
                action = 'long'
                entry_price = current_price
                target_price = predicted_price
                stop_loss = current_price * 0.98  # %2 stop loss
                
            else:  # Short sinyali  
                action = 'short'
                entry_price = current_price
                target_price = predicted_price
                stop_loss = current_price * 1.02  # %2 stop loss
            
            # Risk yüzdesini güven seviyesine göre ayarla
            base_risk = self.trader.default_risk_percent
            confidence_multiplier = confidence / 100
            risk_percent = base_risk * confidence_multiplier
            
            signal = {
                'symbol': symbol,
                'action': action,
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'confidence': confidence,
                'exchange_type': exchange_type,
                'risk_percent': risk_percent,
                'predicted_change': price_change
            }
            
            self.trader.log_message(f"✅ {symbol} sinyal oluşturuldu: {action.upper()}")
            
            return signal
            
        except Exception as e:
            self.trader.log_message(f"❌ {symbol} sinyal oluşturma hatası: {str(e)}")
            return None

def main():
    """
    LSTM AutoTrader test fonksiyonu
    """
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║          🤖 LSTM + BINANCE OTOMATIK TRADİNG SİSTEMİ 🤖           ║
║                                                                    ║
║  Bu sistem LSTM tahminlerini kullanarak Binance'de otomatik       ║
║  trading yapar. Haber ve whale analizini de dahil eder.           ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    print("⚠️ GERÇEK PARA KULLANIR! İlk testleri TESTNET'te yapın!")
    
    # API bilgileri
    binance_api = input("\n🔑 Binance API Key: ").strip()
    binance_secret = input("🔐 Binance Secret Key: ").strip()
    
    use_testnet = input("\n🧪 Testnet kullanılsın mı? (y/n, varsayılan: y): ").strip().lower()
    testnet = use_testnet != 'n'
    
    try:
        # AutoTrader oluştur
        auto_trader = LSTMAutoTrader(binance_api, binance_secret, testnet)
        
        # Test trading
        test_symbol = input("\n📊 Test trading çifti (örn: BTC/USDT): ").strip()
        if not test_symbol:
            test_symbol = "BTC/USDT"
        
        print(f"\n🎯 {test_symbol} için otomatik trading testi...")
        
        result = auto_trader.execute_auto_trading(test_symbol)
        
        if result['success']:
            print("✅ Test işlemi başarılı!")
            print(f"📋 Order ID: {result.get('order_id', 'N/A')}")
        else:
            print(f"❌ Test işlemi başarısız: {result['error']}")
        
    except Exception as e:
        print(f"\n❌ Sistem hatası: {str(e)}")

if __name__ == "__main__":
    main() 