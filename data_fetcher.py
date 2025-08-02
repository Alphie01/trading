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
        Belirtilen coin için OHLCV verilerini çeker (çoklu isteklerle).
        
        Args:
            symbol (str): Coin çifti (örn: 'BTC/USDT')
            timeframe (str): Zaman aralığı (örn: '4h')
            days (int): Kaç günlük veri çekileceği (varsayılan: env'den alınır)

        Returns:
            pd.DataFrame: OHLCV verileri
        """
        import pandas as pd
        import time
        import os

        if days is None:
            days = int(os.getenv('LSTM_TRAINING_DAYS', 1000))  # Varsayılan 1000 gün

        if '/' not in symbol:
            symbol = f"{symbol.upper()}/USDT"

        print(f"{symbol} için {days} günlük {timeframe} verileri çekiliyor...")

        # Her zaman aralığı için kaç ms olduğunu al
        timeframe_to_ms = {
            '1m': 60_000,
            '5m': 5 * 60_000,
            '15m': 15 * 60_000,
            '30m': 30 * 60_000,
            '1h': 60 * 60_000,
            '2h': 2 * 60 * 60_000,
            '4h': 4 * 60 * 60_000,
            '1d': 24 * 60 * 60_000,
        }
        limit = 500
        ms_per_candle = timeframe_to_ms.get(timeframe)
        if not ms_per_candle:
            raise ValueError(f"{timeframe} zaman aralığı desteklenmiyor.")

        total_candles_needed = (days * 24 * 60 * 60 * 1000) // ms_per_candle

        all_ohlcv = []
        since = self.exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        last_timestamp = None

        while len(all_ohlcv) < total_candles_needed:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=limit)
                print(ohlcv[1])
                if not ohlcv:
                    print("Veri alınamadı, devam ediliyor...")
                    break  # Hiç veri gelmezse çık

                # Eğer aynı timestamp'ten veri dönmeye başladıysa döngüye gerek yok
                if last_timestamp is not None and ohlcv[-1][0] == last_timestamp:
                    print("Aynı veriler tekrar dönüyor, veri bitmiş olabilir.")
                    break

                all_ohlcv.extend(ohlcv)
                last_timestamp = ohlcv[-1][0]
                since = last_timestamp + 1  # bir sonraki başlangıç

                if len(ohlcv) < limit:
                    print("Veri sayısı 500'den az, büyük ihtimalle veri bitti.")
                    break  # Daha fazla veri yok

                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                print(f"Hata oluştu: {e}")
                break

        # Veri tekrarlarını temizle
        unique_ohlcv = [list(x) for x in {tuple(row) for row in all_ohlcv}]
        unique_ohlcv.sort(key=lambda x: x[0])

        df = pd.DataFrame(unique_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        print(f"Toplam {len(df)} mum verisi çekildi.")
        print(f"Tarih aralığı: {df.index[0]} - {df.index[-1]}")

        return df
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