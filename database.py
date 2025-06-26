#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Dashboard Veritabanı Yönetimi

Bu modül web arayüzü için:
- Coin listesi yönetimi
- İşlem geçmişi takibi
- Kar/zarar hesaplamaları
- Portfolio analizi
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd

class TradingDatabase:
    """Trading dashboard için veritabanı yönetimi"""
    
    def __init__(self, db_path: str = "trading_dashboard.db"):
        """
        Veritabanını başlatır
        
        Args:
            db_path (str): Veritabanı dosya yolu
        """
        self.db_path = db_path
        self.init_database()
        print(f"📊 Trading Database başlatıldı: {db_path}")
    
    def init_database(self):
        """Veritabanı tablolarını oluşturur"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Coin listesi tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS coins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    name TEXT,
                    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    last_analysis TIMESTAMP,
                    current_price REAL,
                    price_change_24h REAL,
                    analysis_count INTEGER DEFAULT 0
                )
            ''')
            
            # İşlem geçmişi tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_symbol TEXT NOT NULL,
                    trade_type TEXT NOT NULL, -- 'BUY', 'SELL', 'LONG', 'SHORT', 'CLOSE_LONG', 'CLOSE_SHORT'
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    total_value REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    news_sentiment REAL,
                    whale_activity REAL,
                    yigit_signal TEXT,
                    trade_reason TEXT,
                    exchange TEXT DEFAULT 'binance',
                    fees REAL DEFAULT 0,
                    is_simulated BOOLEAN DEFAULT 1,
                    FOREIGN KEY (coin_symbol) REFERENCES coins(symbol)
                )
            ''')
            
            # Pozisyon takibi tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_symbol TEXT NOT NULL,
                    position_type TEXT NOT NULL, -- 'SPOT', 'LONG', 'SHORT'
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    quantity REAL NOT NULL,
                    entry_value REAL NOT NULL,
                    current_value REAL,
                    unrealized_pnl REAL,
                    entry_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_open BOOLEAN DEFAULT 1,
                    stop_loss REAL,
                    take_profit REAL,
                    leverage REAL DEFAULT 1,
                    FOREIGN KEY (coin_symbol) REFERENCES coins(symbol)
                )
            ''')
            
            # Analiz sonuçları tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_symbol TEXT NOT NULL,
                    predicted_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    price_change_percent REAL NOT NULL,
                    confidence REAL NOT NULL,
                    news_sentiment REAL,
                    whale_activity_score REAL,
                    yigit_position INTEGER,
                    yigit_signal TEXT,
                    trend_strength REAL,
                    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_type TEXT DEFAULT 'LSTM',
                    features_used TEXT, -- JSON string
                    FOREIGN KEY (coin_symbol) REFERENCES coins(symbol)
                )
            ''')
            
            # Portfolio özeti tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_balance REAL NOT NULL,
                    invested_amount REAL NOT NULL,
                    current_value REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    total_pnl_percent REAL NOT NULL,
                    active_positions INTEGER NOT NULL,
                    successful_trades INTEGER NOT NULL,
                    total_trades INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    summary_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            print("✅ Veritabanı tabloları oluşturuldu")
    
    def add_coin(self, symbol: str, name: str = None) -> bool:
        """
        İzleme listesine coin ekler
        
        Args:
            symbol (str): Coin sembolü
            name (str): Coin ismi
        
        Returns:
            bool: Başarı durumu
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO coins (symbol, name, is_active)
                    VALUES (?, ?, 1)
                ''', (symbol.upper(), name or symbol))
                conn.commit()
                print(f"✅ {symbol} coin listesine eklendi")
                return True
        except Exception as e:
            print(f"❌ Coin ekleme hatası: {str(e)}")
            return False
    
    def remove_coin(self, symbol: str) -> bool:
        """
        Coin'i izleme listesinden çıkarır
        
        Args:
            symbol (str): Coin sembolü
        
        Returns:
            bool: Başarı durumu
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE coins SET is_active = 0 WHERE symbol = ?
                ''', (symbol.upper(),))
                conn.commit()
                print(f"🗑️ {symbol} izleme listesinden çıkarıldı")
                return True
        except Exception as e:
            print(f"❌ Coin çıkarma hatası: {str(e)}")
            return False
    
    def get_active_coins(self) -> List[Dict]:
        """
        Aktif izlenen coinleri döndürür
        
        Returns:
            List[Dict]: Coin listesi
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT symbol, name, added_date, last_analysis, 
                           current_price, price_change_24h, analysis_count
                    FROM coins WHERE is_active = 1
                    ORDER BY added_date DESC
                ''')
                
                coins = []
                for row in cursor.fetchall():
                    coins.append({
                        'symbol': row[0],
                        'name': row[1],
                        'added_date': row[2],
                        'last_analysis': row[3],
                        'current_price': row[4],
                        'price_change_24h': row[5],
                        'analysis_count': row[6]
                    })
                
                return coins
        except Exception as e:
            print(f"❌ Coin listesi alma hatası: {str(e)}")
            return []
    
    def record_trade(self, coin_symbol: str, trade_type: str, price: float,
                    quantity: float, confidence: float = None, 
                    news_sentiment: float = None, whale_activity: float = None,
                    yigit_signal: str = None, trade_reason: str = None,
                    fees: float = 0, is_simulated: bool = True) -> int:
        """
        İşlem kaydeder
        
        Args:
            coin_symbol (str): Coin sembolü
            trade_type (str): İşlem tipi (BUY, SELL, LONG, SHORT, etc.)
            price (float): İşlem fiyatı
            quantity (float): Miktar
            confidence (float): Güven skoru
            news_sentiment (float): Haber sentiment'i
            whale_activity (float): Whale aktivitesi
            yigit_signal (str): Yigit sinyali
            trade_reason (str): İşlem nedeni
            fees (float): İşlem ücreti
            is_simulated (bool): Simülasyon mu?
        
        Returns:
            int: İşlem ID'si
        """
        try:
            total_value = price * quantity
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades (
                        coin_symbol, trade_type, price, quantity, total_value,
                        confidence, news_sentiment, whale_activity, yigit_signal,
                        trade_reason, fees, is_simulated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    coin_symbol.upper(), trade_type.upper(), price, quantity, total_value,
                    confidence, news_sentiment, whale_activity, yigit_signal,
                    trade_reason, fees, is_simulated
                ))
                
                trade_id = cursor.lastrowid
                conn.commit()
                
                print(f"💰 İşlem kaydedildi: {trade_type} {quantity} {coin_symbol} @ ${price:.6f}")
                return trade_id
                
        except Exception as e:
            print(f"❌ İşlem kaydetme hatası: {str(e)}")
            return None
    
    def update_position(self, coin_symbol: str, position_type: str, 
                       entry_price: float, quantity: float, current_price: float = None,
                       leverage: float = 1, stop_loss: float = None, 
                       take_profit: float = None) -> int:
        """
        Pozisyon günceller veya oluşturur
        
        Args:
            coin_symbol (str): Coin sembolü
            position_type (str): Pozisyon tipi (SPOT, LONG, SHORT)
            entry_price (float): Giriş fiyatı
            quantity (float): Miktar
            current_price (float): Güncel fiyat
            leverage (float): Kaldıraç
            stop_loss (float): Stop loss
            take_profit (float): Take profit
        
        Returns:
            int: Pozisyon ID'si
        """
        try:
            entry_value = entry_price * quantity
            current_value = (current_price or entry_price) * quantity
            
            # Kar/zarar hesaplaması
            if position_type.upper() == 'SHORT':
                unrealized_pnl = (entry_price - (current_price or entry_price)) * quantity * leverage
            else:
                unrealized_pnl = ((current_price or entry_price) - entry_price) * quantity * leverage
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Mevcut açık pozisyonu kontrol et
                cursor.execute('''
                    SELECT id FROM positions 
                    WHERE coin_symbol = ? AND position_type = ? AND is_open = 1
                ''', (coin_symbol.upper(), position_type.upper()))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Mevcut pozisyonu güncelle
                    cursor.execute('''
                        UPDATE positions SET
                            current_price = ?, current_value = ?, unrealized_pnl = ?,
                            last_update = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (current_price or entry_price, current_value, unrealized_pnl, existing[0]))
                    position_id = existing[0]
                else:
                    # Yeni pozisyon oluştur
                    cursor.execute('''
                        INSERT INTO positions (
                            coin_symbol, position_type, entry_price, current_price,
                            quantity, entry_value, current_value, unrealized_pnl,
                            leverage, stop_loss, take_profit
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        coin_symbol.upper(), position_type.upper(), entry_price, 
                        current_price or entry_price, quantity, entry_value, 
                        current_value, unrealized_pnl, leverage, stop_loss, take_profit
                    ))
                    position_id = cursor.lastrowid
                
                conn.commit()
                print(f"📈 Pozisyon güncellendi: {position_type} {coin_symbol} PnL: ${unrealized_pnl:.2f}")
                return position_id
                
        except Exception as e:
            print(f"❌ Pozisyon güncelleme hatası: {str(e)}")
            return None
    
    def close_position(self, coin_symbol: str, position_type: str, 
                      exit_price: float, exit_reason: str = None) -> Dict:
        """
        Pozisyonu kapatır ve kar/zarar hesaplar
        
        Args:
            coin_symbol (str): Coin sembolü
            position_type (str): Pozisyon tipi
            exit_price (float): Çıkış fiyatı
            exit_reason (str): Kapanış nedeni
        
        Returns:
            Dict: Kapatma sonuçları
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Açık pozisyonu bul
                cursor.execute('''
                    SELECT id, entry_price, quantity, leverage, entry_value
                    FROM positions 
                    WHERE coin_symbol = ? AND position_type = ? AND is_open = 1
                ''', (coin_symbol.upper(), position_type.upper()))
                
                position = cursor.fetchone()
                if not position:
                    return {'success': False, 'message': 'Açık pozisyon bulunamadı'}
                
                pos_id, entry_price, quantity, leverage, entry_value = position
                
                # Kar/zarar hesapla
                if position_type.upper() == 'SHORT':
                    realized_pnl = (entry_price - exit_price) * quantity * leverage
                else:
                    realized_pnl = (exit_price - entry_price) * quantity * leverage
                
                pnl_percent = (realized_pnl / entry_value) * 100
                
                # Pozisyonu kapat
                cursor.execute('''
                    UPDATE positions SET
                        is_open = 0, current_price = ?, unrealized_pnl = ?,
                        last_update = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (exit_price, realized_pnl, pos_id))
                
                # Kapanış işlemini kaydet
                self.record_trade(
                    coin_symbol, f'CLOSE_{position_type}', exit_price, quantity,
                    trade_reason=f'Position closed: {exit_reason or "Manual"}'
                )
                
                conn.commit()
                
                result = {
                    'success': True,
                    'realized_pnl': realized_pnl,
                    'pnl_percent': pnl_percent,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'leverage': leverage
                }
                
                print(f"🏁 Pozisyon kapatıldı: {position_type} {coin_symbol} PnL: ${realized_pnl:.2f} ({pnl_percent:+.2f}%)")
                return result
                
        except Exception as e:
            print(f"❌ Pozisyon kapatma hatası: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def save_analysis_result(self, coin_symbol: str, prediction_result: Dict,
                           news_analysis: Dict = None, whale_analysis: Dict = None,
                           yigit_analysis: Dict = None) -> int:
        """
        Analiz sonucunu kaydeder
        
        Args:
            coin_symbol (str): Coin sembolü
            prediction_result (Dict): Tahmin sonuçları
            news_analysis (Dict): Haber analizi
            whale_analysis (Dict): Whale analizi
            yigit_analysis (Dict): Yigit analizi
        
        Returns:
            int: Analiz ID'si
        """
        try:
            features_used = {
                'news': news_analysis is not None,
                'whale': whale_analysis is not None,
                'yigit': yigit_analysis is not None,
                'technical': True
            }
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO analysis_results (
                        coin_symbol, predicted_price, current_price, price_change_percent,
                        confidence, news_sentiment, whale_activity_score, yigit_position,
                        yigit_signal, trend_strength, features_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    coin_symbol.upper(),
                    prediction_result.get('predicted_price', 0),
                    prediction_result.get('current_price', 0),
                    prediction_result.get('price_change_percent', 0),
                    prediction_result.get('confidence', 0),
                    news_analysis.get('news_sentiment', 0) if news_analysis else None,
                    whale_analysis.get('whale_activity_score', 0) if whale_analysis else None,
                    yigit_analysis.get('current_position', 0) if yigit_analysis else None,
                    yigit_analysis.get('current_signal', None) if yigit_analysis else None,
                    yigit_analysis.get('trend_strength', 0) if yigit_analysis else None,
                    json.dumps(features_used)
                ))
                
                analysis_id = cursor.lastrowid
                
                # Coin'in son analiz tarihini güncelle
                cursor.execute('''
                    UPDATE coins SET 
                        last_analysis = CURRENT_TIMESTAMP,
                        current_price = ?,
                        analysis_count = analysis_count + 1
                    WHERE symbol = ?
                ''', (prediction_result.get('current_price', 0), coin_symbol.upper()))
                
                conn.commit()
                print(f"📊 Analiz sonucu kaydedildi: {coin_symbol}")
                return analysis_id
                
        except Exception as e:
            print(f"❌ Analiz kaydetme hatası: {str(e)}")
            return None
    
    def get_portfolio_summary(self) -> Dict:
        """
        Portfolio özetini döndürür
        
        Returns:
            Dict: Portfolio özeti
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Açık pozisyonlar
                cursor.execute('''
                    SELECT COUNT(*), SUM(entry_value), SUM(current_value), SUM(unrealized_pnl)
                    FROM positions WHERE is_open = 1
                ''')
                pos_data = cursor.fetchone()
                active_positions = pos_data[0] or 0
                invested_amount = pos_data[1] or 0
                current_value = pos_data[2] or 0
                unrealized_pnl = pos_data[3] or 0
                
                # Toplam işlem sayısı ve başarı oranı
                cursor.execute('''
                    SELECT COUNT(*) FROM trades WHERE trade_type IN ('CLOSE_LONG', 'CLOSE_SHORT', 'SELL')
                ''')
                total_trades = cursor.fetchone()[0] or 0
                
                # Başarılı işlemler (kar eden)
                cursor.execute('''
                    SELECT COUNT(*) FROM positions 
                    WHERE is_open = 0 AND unrealized_pnl > 0
                ''')
                successful_trades = cursor.fetchone()[0] or 0
                
                win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
                total_pnl_percent = (unrealized_pnl / invested_amount * 100) if invested_amount > 0 else 0
                
                summary = {
                    'total_balance': current_value,
                    'invested_amount': invested_amount,
                    'current_value': current_value,
                    'unrealized_pnl': unrealized_pnl,
                    'total_pnl_percent': total_pnl_percent,
                    'active_positions': active_positions,
                    'successful_trades': successful_trades,
                    'total_trades': total_trades,
                    'win_rate': win_rate
                }
                
                return summary
                
        except Exception as e:
            print(f"❌ Portfolio özeti hatası: {str(e)}")
            return {}
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """
        Son işlemleri döndürür
        
        Args:
            limit (int): Döndürülecek işlem sayısı
        
        Returns:
            List[Dict]: İşlem listesi
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT coin_symbol, trade_type, price, quantity, total_value,
                           timestamp, confidence, trade_reason, fees, is_simulated
                    FROM trades 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                trades = []
                for row in cursor.fetchall():
                    trades.append({
                        'coin_symbol': row[0],
                        'trade_type': row[1],
                        'price': row[2],
                        'quantity': row[3],
                        'total_value': row[4],
                        'timestamp': row[5],
                        'confidence': row[6],
                        'trade_reason': row[7],
                        'fees': row[8],
                        'is_simulated': row[9]
                    })
                
                return trades
                
        except Exception as e:
            print(f"❌ İşlem geçmişi alma hatası: {str(e)}")
            return []
    
    def get_open_positions(self) -> List[Dict]:
        """
        Açık pozisyonları döndürür
        
        Returns:
            List[Dict]: Pozisyon listesi
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT coin_symbol, position_type, entry_price, current_price,
                           quantity, entry_value, current_value, unrealized_pnl,
                           entry_timestamp, leverage, stop_loss, take_profit
                    FROM positions 
                    WHERE is_open = 1
                    ORDER BY entry_timestamp DESC
                ''')
                
                positions = []
                for row in cursor.fetchall():
                    pnl_percent = ((row[7] or 0) / (row[5] or 1)) * 100
                    positions.append({
                        'coin_symbol': row[0],
                        'position_type': row[1],
                        'entry_price': row[2],
                        'current_price': row[3],
                        'quantity': row[4],
                        'entry_value': row[5],
                        'current_value': row[6],
                        'unrealized_pnl': row[7],
                        'pnl_percent': pnl_percent,
                        'entry_timestamp': row[8],
                        'leverage': row[9],
                        'stop_loss': row[10],
                        'take_profit': row[11]
                    })
                
                return positions
                
        except Exception as e:
            print(f"❌ Açık pozisyon alma hatası: {str(e)}")
            return []
    
    def get_coin_performance(self, coin_symbol: str, days: int = 30) -> Dict:
        """
        Belirli coin'in performansını döndürür
        
        Args:
            coin_symbol (str): Coin sembolü
            days (int): Kaç günlük performans
        
        Returns:
            Dict: Performans verileri
        """
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Analiz sayısı
                cursor.execute('''
                    SELECT COUNT(*) FROM analysis_results 
                    WHERE coin_symbol = ? AND analysis_timestamp > ?
                ''', (coin_symbol.upper(), since_date))
                analysis_count = cursor.fetchone()[0] or 0
                
                # İşlem sayısı
                cursor.execute('''
                    SELECT COUNT(*) FROM trades 
                    WHERE coin_symbol = ? AND timestamp > ?
                ''', (coin_symbol.upper(), since_date))
                trade_count = cursor.fetchone()[0] or 0
                
                # Toplam kar/zarar
                cursor.execute('''
                    SELECT SUM(unrealized_pnl) FROM positions 
                    WHERE coin_symbol = ? AND entry_timestamp > ?
                ''', (coin_symbol.upper(), since_date))
                total_pnl = cursor.fetchone()[0] or 0
                
                # Ortalama güven skoru
                cursor.execute('''
                    SELECT AVG(confidence) FROM analysis_results 
                    WHERE coin_symbol = ? AND analysis_timestamp > ?
                ''', (coin_symbol.upper(), since_date))
                avg_confidence = cursor.fetchone()[0] or 0
                
                return {
                    'coin_symbol': coin_symbol,
                    'analysis_count': analysis_count,
                    'trade_count': trade_count,
                    'total_pnl': total_pnl,
                    'avg_confidence': avg_confidence,
                    'days': days
                }
                
        except Exception as e:
            print(f"❌ Coin performans hatası: {str(e)}")
            return {}
    
    def save_system_state(self, state_key: str, state_value: Any):
        """
        Sistem durumunu kaydeder (persistence için)
        
        Args:
            state_key (str): Durum anahtarı
            state_value (Any): Durum değeri
        """
        try:
            value_str = json.dumps(state_value, default=str, ensure_ascii=False)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # System state tablosu yoksa oluştur
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        state_key TEXT UNIQUE NOT NULL,
                        state_value TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Upsert işlemi
                cursor.execute('''
                    INSERT OR REPLACE INTO system_state (state_key, state_value, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (state_key, value_str))
                
                conn.commit()
                print(f"💾 System state kaydedildi: {state_key}")
                
        except Exception as e:
            print(f"❌ System state kaydetme hatası: {str(e)}")
    
    def load_system_state(self, state_key: str, default_value: Any = None) -> Any:
        """
        Sistem durumunu yükler
        
        Args:
            state_key (str): Durum anahtarı
            default_value (Any): Varsayılan değer
        
        Returns:
            Any: Durum değeri
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # System state tablosu yoksa oluştur
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        state_key TEXT UNIQUE NOT NULL,
                        state_value TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    SELECT state_value FROM system_state WHERE state_key = ?
                ''', (state_key,))
                
                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])
                else:
                    return default_value
                    
        except Exception as e:
            print(f"❌ System state yükleme hatası: {str(e)}")
            return default_value

def main():
    """Test fonksiyonu"""
    print("🧪 Trading Database Test")
    
    db = TradingDatabase()
    
    # Test coinleri ekle
    db.add_coin('BTC', 'Bitcoin')
    db.add_coin('ETH', 'Ethereum')
    db.add_coin('BNB', 'Binance Coin')
    
    # Test işlemi kaydet
    trade_id = db.record_trade(
        'BTC', 'BUY', 50000.0, 0.1, 
        confidence=85.5, trade_reason='LSTM Signal + News Positive'
    )
    
    # Test pozisyonu güncelle
    db.update_position('BTC', 'LONG', 50000.0, 0.1, 51000.0, leverage=2)
    
    # Sonuçları göster
    coins = db.get_active_coins()
    print(f"Aktif coinler: {len(coins)}")
    
    portfolio = db.get_portfolio_summary()
    print(f"Portfolio PnL: ${portfolio.get('unrealized_pnl', 0):.2f}")

if __name__ == "__main__":
    main() 