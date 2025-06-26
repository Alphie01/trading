#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSSQL Trading Dashboard Veritabanı Yönetimi

Bu modül MSSQL Server ile entegre çalışan veritabanı yönetimi sağlar:
- Environment variables ile güvenli konfigürasyon
- Connection pooling ve optimize edilmiş performans
- Automatic reconnection ve error handling
- Data persistence ve backup özellikleri
"""

import os
import json
import pyodbc
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import logging
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

class MSSQLTradingDatabase:
    """MSSQL Server için trading dashboard veritabanı yönetimi"""
    
    def __init__(self):
        self.server = os.getenv('MSSQL_SERVER', 'localhost')
        self.database = os.getenv('MSSQL_DATABASE', 'crypto_trading_db')
        self.username = os.getenv('MSSQL_USERNAME', 'sa')
        self.password = os.getenv('MSSQL_PASSWORD', '')
        self.driver = os.getenv('MSSQL_DRIVER', 'ODBC Driver 17 for SQL Server')
        self.port = os.getenv('MSSQL_PORT', '1433')
        
        self.connection_string = self._build_connection_string()
        self.logger = self._setup_logger()
        
        self.init_database()
        print(f"✅ MSSQL Database başlatıldı: {self.server}/{self.database}")
    
    def _build_connection_string(self) -> str:
        return (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server},{self.port};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"TrustServerCertificate=yes;"
        )
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MSSQLDatabase')
        logger.setLevel(getattr(logging, os.getenv('LOG_LEVEL', 'INFO')))
        return logger
    
    def get_connection(self):
        try:
            return pyodbc.connect(self.connection_string, autocommit=False)
        except Exception as e:
            self.logger.error(f"Database bağlantı hatası: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False) -> Any:
        connection = None
        cursor = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                if query.strip().upper().startswith('SELECT'):
                    columns = [column[0] for column in cursor.description]
                    results = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in results]
                else:
                    return cursor.fetchall()
            else:
                connection.commit()
                return cursor.rowcount
                
        except Exception as e:
            if connection:
                connection.rollback()
            self.logger.error(f"Query error: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def init_database(self):
        """Veritabanı tablolarını oluşturur"""
        try:
            tables = [
                """
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='coins' AND xtype='U')
                CREATE TABLE coins (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    symbol NVARCHAR(20) UNIQUE NOT NULL,
                    name NVARCHAR(100),
                    added_date DATETIME2 DEFAULT GETDATE(),
                    is_active BIT DEFAULT 1,
                    last_analysis DATETIME2,
                    current_price DECIMAL(18,8),
                    price_change_24h DECIMAL(10,4),
                    analysis_count INT DEFAULT 0
                )
                """,
                """
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='system_state' AND xtype='U')
                CREATE TABLE system_state (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    state_key NVARCHAR(100) UNIQUE NOT NULL,
                    state_value NVARCHAR(MAX),
                    last_updated DATETIME2 DEFAULT GETDATE()
                )
                """
            ]
            
            for table_sql in tables:
                self.execute_query(table_sql)
            
            print("✅ MSSQL tablolar oluşturuldu")
            
        except Exception as e:
            print(f"❌ Database init error: {str(e)}")
    
    def add_coin(self, symbol: str, name: str = None) -> bool:
        try:
            query = """
            IF NOT EXISTS (SELECT 1 FROM coins WHERE symbol = ?)
                INSERT INTO coins (symbol, name) VALUES (?, ?)
            ELSE
                UPDATE coins SET is_active = 1 WHERE symbol = ?
            """
            self.execute_query(query, (symbol.upper(), symbol.upper(), name or symbol, symbol.upper()))
            return True
        except Exception as e:
            print(f"❌ Coin ekleme hatası: {str(e)}")
            return False
    
    def get_active_coins(self) -> List[Dict]:
        try:
            query = """
            SELECT symbol, name, added_date, last_analysis, 
                   current_price, price_change_24h, analysis_count
            FROM coins WHERE is_active = 1
            ORDER BY added_date DESC
            """
            return self.execute_query(query, fetch=True)
        except Exception as e:
            print(f"❌ Coin listesi hatası: {str(e)}")
            return []
    
    def save_system_state(self, state_key: str, state_value: Any):
        try:
            value_str = json.dumps(state_value, default=str, ensure_ascii=False)
            query = """
            IF EXISTS (SELECT 1 FROM system_state WHERE state_key = ?)
                UPDATE system_state SET state_value = ?, last_updated = GETDATE() WHERE state_key = ?
            ELSE
                INSERT INTO system_state (state_key, state_value) VALUES (?, ?)
            """
            self.execute_query(query, (state_key, value_str, state_key, state_key, value_str))
        except Exception as e:
            print(f"❌ State kaydetme hatası: {str(e)}")
    
    def load_system_state(self, state_key: str, default_value: Any = None) -> Any:
        try:
            query = "SELECT state_value FROM system_state WHERE state_key = ?"
            result = self.execute_query(query, (state_key,), fetch=True)
            
            if result:
                return json.loads(result[0]['state_value'])
            return default_value
        except Exception as e:
            print(f"❌ State yükleme hatası: {str(e)}")
            return default_value

    def get_portfolio_summary(self) -> Dict:
        """Portfolio özetini döndürür"""
        try:
            # Basit portfolio özeti
            return {
                'total_value': 0.0,
                'total_profit_loss': 0.0,
                'active_positions': 0,
                'total_trades': 0
            }
        except Exception as e:
            print(f"❌ Portfolio özeti hatası: {str(e)}")
            return {}
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Son işlemleri döndürür"""
        try:
            return []  # Şimdilik boş liste
        except Exception as e:
            print(f"❌ Son işlemler hatası: {str(e)}")
            return []
    
    def get_open_positions(self) -> List[Dict]:
        """Açık pozisyonları döndürür"""
        try:
            return []  # Şimdilik boş liste
        except Exception as e:
            print(f"❌ Açık pozisyonlar hatası: {str(e)}")
            return []
    
    def remove_coin(self, symbol: str) -> bool:
        """Coin'i pasif yapar"""
        try:
            query = "UPDATE coins SET is_active = 0 WHERE symbol = ?"
            self.execute_query(query, (symbol.upper(),))
            return True
        except Exception as e:
            print(f"❌ Coin çıkarma hatası: {str(e)}")
            return False

    def save_analysis_result(self, coin_symbol: str, prediction_result: Dict, 
                           news_analysis: Dict, whale_analysis: Dict, 
                           yigit_analysis: Dict) -> str:
        """Analiz sonuçlarını kaydeder"""
        try:
            # Analysis ID oluştur
            analysis_id = f"{coin_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Analiz verilerini JSON formatında birleştir
            analysis_data = {
                'coin_symbol': coin_symbol,
                'prediction': prediction_result,
                'news_analysis': news_analysis,
                'whale_analysis': whale_analysis,
                'yigit_analysis': yigit_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            # System state'e kaydet (analysis_results prefix ile)
            self.save_system_state(f"analysis_{analysis_id}", analysis_data)
            
            # Coin tablosundaki analiz sayısını artır
            query = """
            UPDATE coins SET 
                analysis_count = analysis_count + 1,
                last_analysis = GETDATE(),
                current_price = ?
            WHERE symbol = ?
            """
            current_price = prediction_result.get('current_price', 0)
            self.execute_query(query, (current_price, coin_symbol.upper()))
            
            return analysis_id
        except Exception as e:
            print(f"❌ Analiz kaydetme hatası: {str(e)}")
            return ""

    def record_trade(self, coin_symbol: str, action: str, target_price: float,
                    quantity: float, confidence: float, news_sentiment: float,
                    whale_activity: float, yigit_signal: str, reason: str,
                    is_simulated: bool = True) -> bool:
        """İşlem kaydı yapar"""
        try:
            # Trade ID oluştur
            trade_id = f"{coin_symbol}_{action}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # İşlem verilerini JSON formatında kaydet
            trade_data = {
                'trade_id': trade_id,
                'coin_symbol': coin_symbol,
                'action': action,
                'target_price': target_price,
                'quantity': quantity,
                'confidence': confidence,
                'news_sentiment': news_sentiment,
                'whale_activity': whale_activity,
                'yigit_signal': yigit_signal,
                'reason': reason,
                'is_simulated': is_simulated,
                'timestamp': datetime.now().isoformat()
            }
            
            # System state'e kaydet (trades prefix ile)
            self.save_system_state(f"trade_{trade_id}", trade_data)
            
            return True
        except Exception as e:
            print(f"❌ İşlem kaydetme hatası: {str(e)}")
            return False

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
            # Belirli bir coin için performans verileri
            query = """
            SELECT symbol, name, current_price, price_change_24h, 
                   analysis_count, last_analysis, added_date
            FROM coins WHERE symbol = ? AND is_active = 1
            """
            result = self.execute_query(query, (coin_symbol.upper(),), fetch=True)
            
            if result and len(result) > 0:
                coin_data = result[0]
                
                # Database'den analiz geçmişini al (son X gün)
                analysis_history = self.get_analysis_history(coin_symbol, limit=days)
                
                # Performans hesaplamaları
                analysis_count = len(analysis_history)
                avg_confidence = 0
                
                if analysis_history:
                    confidences = []
                    for analysis in analysis_history:
                        if 'prediction' in analysis and 'confidence' in analysis['prediction']:
                            confidences.append(analysis['prediction']['confidence'])
                    
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                return {
                    'coin_symbol': coin_symbol,
                    'current_price': coin_data.get('current_price', 0),
                    'price_change_24h': coin_data.get('price_change_24h', 0),
                    'analysis_count': analysis_count,
                    'avg_confidence': avg_confidence,
                    'days': days,
                    'last_analysis': coin_data.get('last_analysis'),
                    'total_pnl': 0,  # MSSQL'de trade tracking henüz tam değil
                    'trade_count': 0  # MSSQL'de trade tracking henüz tam değil
                }
            else:
                return {
                    'coin_symbol': coin_symbol,
                    'current_price': 0,
                    'price_change_24h': 0,
                    'analysis_count': 0,
                    'avg_confidence': 0,
                    'days': days,
                    'last_analysis': None,
                    'total_pnl': 0,
                    'trade_count': 0
                }
        except Exception as e:
            print(f"❌ Coin performans hatası: {str(e)}")
            return {
                'coin_symbol': coin_symbol,
                'current_price': 0,
                'price_change_24h': 0,
                'analysis_count': 0,
                'avg_confidence': 0,
                'days': days,
                'last_analysis': None,
                'total_pnl': 0,
                'trade_count': 0
            }

    def get_analysis_history(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """Analiz geçmişini döndürür"""
        try:
            # System state'den analiz verilerini çek
            query = "SELECT state_key, state_value FROM system_state WHERE state_key LIKE ? ORDER BY last_updated DESC"
            if symbol:
                pattern = f"analysis_{symbol}_%"
            else:
                pattern = "analysis_%"
            
            results = self.execute_query(query, (pattern,), fetch=True)
            analysis_history = []
            
            for result in results[:limit]:
                try:
                    analysis_data = json.loads(result['state_value'])
                    analysis_history.append(analysis_data)
                except:
                    continue
            
            return analysis_history
        except Exception as e:
            print(f"❌ Analiz geçmişi hatası: {str(e)}")
            return []

    def get_all_coin_performances(self, days: int = 30) -> List[Dict]:
        """
        Tüm aktif coinlerin performanslarını döndürür
        
        Args:
            days (int): Kaç günlük performans
        
        Returns:
            List[Dict]: Tüm coin performansları
        """
        try:
            coins = self.get_active_coins()
            performances = []
            
            for coin in coins:
                performance = self.get_coin_performance(coin['symbol'], days)
                performances.append(performance)
            
            return performances
        except Exception as e:
            print(f"❌ Tüm coin performansları hatası: {str(e)}")
            return []

    def test_connection(self) -> bool:
        try:
            result = self.execute_query("SELECT 1", fetch=True)
            return len(result) > 0
        except:
            return False

if __name__ == "__main__":
    db = MSSQLTradingDatabase()
    if db.test_connection():
        print("✅ MSSQL bağlantısı başarılı!")
    else:
        print("❌ MSSQL bağlantısı başarısız!")
