import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Environment variables yükle
load_dotenv()

class CryptoDataFetcher:
    """
    Binance API'si üzerinden kripto para verilerini çeken sınıf
    """
    
    def __init__(self):
        """
        Binance exchange bağlantısını başlatır
        """
        self.exchange = ccxt.binance({
            'apiKey': '',  # API key gerekmez sadece veri çekmek için
            'secret': '',
            'timeout': 30000,
            'enableRateLimit': True,
        })
    
    def fetch_ohlcv_data(self, symbol, timeframe='4h', days=None):
        """
        Belirtilen coin için OHLCV verilerini çeker
        
        Args:
            symbol (str): Coin çifti (örn: 'BTC/USDT')
            timeframe (str): Zaman aralığı (varsayılan: '4h')
            days (int): Kaç günlük veri çekileceği (None ise environment'tan alır)
        
        Returns:
            pd.DataFrame: OHLCV verileri içeren DataFrame
        """
        # Days parametresi verilmemişse environment'tan al
        if days is None:
            days = int(os.getenv('LSTM_TRAINING_DAYS', 100))  # Varsayılan: 100 gün
        try:
            # Symbol'ü doğru formata çevir
            if '/' not in symbol:
                symbol = f"{symbol.upper()}/USDT"
            
            # Başlangıç tarihini hesapla
            since = self.exchange.milliseconds() - days * 24 * 60 * 60 * 1000
            
            print(f"{symbol} için {days} günlük {timeframe} verileri çekiliyor...")
            
            # Verileri çek
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since)
            
            if not ohlcv:
                raise ValueError(f"{symbol} için veri bulunamadı")
            
            # DataFrame'e çevir
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Timestamp'i datetime'a çevir
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
            
            # Veri tiplerini float'a çevir
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            print(f"Başarıyla {len(df)} adet veri çekildi")
            print(f"Tarih aralığı: {df.index[0]} - {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"Veri çekme hatası: {str(e)}")
            return None
    
    def get_available_symbols(self, base_currency='USDT'):
        """
        Mevcut coin çiftlerini listeler
        
        Args:
            base_currency (str): Base para birimi (varsayılan: 'USDT')
        
        Returns:
            list: Mevcut coin çiftleri listesi
        """
        try:
            markets = self.exchange.fetch_markets()
            symbols = [market['symbol'] for market in markets 
                      if market['quote'] == base_currency and market['active']]
            return sorted(symbols)
        except Exception as e:
            print(f"Symbol listesi alınırken hata: {str(e)}")
            return []
    
    def validate_symbol(self, symbol):
        """
        Symbol'ün geçerli olup olmadığını kontrol eder
        
        Args:
            symbol (str): Kontrol edilecek symbol
        
        Returns:
            bool: Symbol geçerliyse True
        """
        try:
            if '/' not in symbol:
                symbol = f"{symbol.upper()}/USDT"
            
            markets = self.exchange.fetch_markets()
            valid_symbols = [market['symbol'] for market in markets if market['active']]
            
            return symbol in valid_symbols
        except Exception as e:
            print(f"Symbol doğrulama hatası: {str(e)}")
            return False 