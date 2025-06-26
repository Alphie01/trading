#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSSQL Database Oluşturma Script'i

Bu script MSSQL server'da crypto_trading_db database'ini ve 
tüm gerekli tabloları oluşturur.
"""

import os
import pyodbc
import time
from dotenv import load_dotenv

# Environment variables yükle
load_dotenv()

def print_banner():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║              🗄️ MSSQL DATABASE OLUŞTURMA SCRIPT'İ 🗄️              ║
║                                                                    ║
║  Bu script MSSQL server'da crypto_trading_db database'ini          ║
║  ve tüm gerekli tabloları oluşturur.                              ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")

def create_database():
    """MSSQL server'da database oluşturur"""
    
    # Connection string - master database'e bağlan
    server = os.getenv('MSSQL_SERVER')
    username = os.getenv('MSSQL_USERNAME') 
    password = os.getenv('MSSQL_PASSWORD')
    driver = os.getenv('MSSQL_DRIVER', 'ODBC Driver 17 for SQL Server')
    
    print(f"🔗 MSSQL Server'a bağlanılıyor: {server}")
    print(f"👤 Kullanıcı: {username}")
    
    try:
        # Master database'e bağlan
        connection_string = f"""
        DRIVER={{{driver}}};
        SERVER={server};
        DATABASE=master;
        UID={username};
        PWD={password};
        TrustServerCertificate=yes;
        """
        
        print("📡 Master database'e bağlanıyor...")
        conn = pyodbc.connect(connection_string)
        conn.autocommit = True  # CREATE DATABASE için gerekli
        cursor = conn.cursor()
        
        print("✅ Master database bağlantısı başarılı!")
        
        # Database var mı kontrol et
        print("🔍 crypto_trading_db var mı kontrol ediliyor...")
        cursor.execute("""
            SELECT name FROM sys.databases WHERE name = 'crypto_trading_db'
        """)
        
        if cursor.fetchone():
            print("ℹ️ crypto_trading_db zaten mevcut!")
        else:
            print("📦 crypto_trading_db oluşturuluyor...")
            
            # Database oluştur
            cursor.execute("""
                CREATE DATABASE crypto_trading_db
                COLLATE Turkish_CI_AS
            """)
            
            print("✅ crypto_trading_db başarıyla oluşturuldu!")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Database oluşturma hatası: {str(e)}")
        return False

def create_tables():
    """crypto_trading_db'de tabloları oluşturur"""
    
    server = os.getenv('MSSQL_SERVER')
    username = os.getenv('MSSQL_USERNAME') 
    password = os.getenv('MSSQL_PASSWORD')
    driver = os.getenv('MSSQL_DRIVER', 'ODBC Driver 17 for SQL Server')
    
    try:
        # crypto_trading_db'ye bağlan
        connection_string = f"""
        DRIVER={{{driver}}};
        SERVER={server};
        DATABASE=crypto_trading_db;
        UID={username};
        PWD={password};
        TrustServerCertificate=yes;
        """
        
        print("📡 crypto_trading_db'ye bağlanıyor...")
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        
        print("✅ crypto_trading_db bağlantısı başarılı!")
        
        # Coins tablosu
        print("📋 coins tablosu oluşturuluyor...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='coins' AND xtype='U')
            CREATE TABLE coins (
                id int IDENTITY(1,1) PRIMARY KEY,
                symbol nvarchar(20) UNIQUE NOT NULL,
                name nvarchar(100),
                added_date datetime2 DEFAULT GETDATE(),
                is_active bit DEFAULT 1,
                last_analysis datetime2,
                current_price decimal(18,8),
                price_change_24h decimal(10,4),
                analysis_count int DEFAULT 0
            )
        """)
        
        # Trades tablosu  
        print("💰 trades tablosu oluşturuluyor...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='trades' AND xtype='U')
            CREATE TABLE trades (
                id int IDENTITY(1,1) PRIMARY KEY,
                coin_symbol nvarchar(20) NOT NULL,
                trade_type nvarchar(20) NOT NULL,
                price decimal(18,8) NOT NULL,
                quantity decimal(18,8) NOT NULL,
                total_value decimal(18,8) NOT NULL,
                timestamp datetime2 DEFAULT GETDATE(),
                confidence decimal(5,2),
                news_sentiment decimal(5,2),
                whale_activity decimal(5,2),
                yigit_signal nvarchar(50),
                trade_reason nvarchar(500),
                exchange nvarchar(50) DEFAULT 'binance',
                fees decimal(18,8) DEFAULT 0,
                is_simulated bit DEFAULT 1,
                FOREIGN KEY (coin_symbol) REFERENCES coins(symbol)
            )
        """)
        
        # Positions tablosu
        print("📈 positions tablosu oluşturuluyor...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='positions' AND xtype='U')
            CREATE TABLE positions (
                id int IDENTITY(1,1) PRIMARY KEY,
                coin_symbol nvarchar(20) NOT NULL,
                position_type nvarchar(20) NOT NULL,
                entry_price decimal(18,8) NOT NULL,
                current_price decimal(18,8),
                quantity decimal(18,8) NOT NULL,
                entry_value decimal(18,8) NOT NULL,
                current_value decimal(18,8),
                unrealized_pnl decimal(18,8),
                entry_timestamp datetime2 DEFAULT GETDATE(),
                last_update datetime2 DEFAULT GETDATE(),
                is_open bit DEFAULT 1,
                stop_loss decimal(18,8),
                take_profit decimal(18,8),
                leverage decimal(5,2) DEFAULT 1,
                FOREIGN KEY (coin_symbol) REFERENCES coins(symbol)
            )
        """)
        
        # Analysis results tablosu
        print("📊 analysis_results tablosu oluşturuluyor...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='analysis_results' AND xtype='U')
            CREATE TABLE analysis_results (
                id int IDENTITY(1,1) PRIMARY KEY,
                coin_symbol nvarchar(20) NOT NULL,
                predicted_price decimal(18,8) NOT NULL,
                current_price decimal(18,8) NOT NULL,
                price_change_percent decimal(10,4) NOT NULL,
                confidence decimal(5,2) NOT NULL,
                news_sentiment decimal(5,2),
                whale_activity_score decimal(5,2),
                yigit_position int,
                yigit_signal nvarchar(50),
                trend_strength decimal(5,2),
                analysis_timestamp datetime2 DEFAULT GETDATE(),
                model_type nvarchar(50) DEFAULT 'LSTM',
                features_used nvarchar(1000),
                FOREIGN KEY (coin_symbol) REFERENCES coins(symbol)
            )
        """)
        
        # Portfolio summary tablosu
        print("💼 portfolio_summary tablosu oluşturuluyor...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='portfolio_summary' AND xtype='U')
            CREATE TABLE portfolio_summary (
                id int IDENTITY(1,1) PRIMARY KEY,
                total_balance decimal(18,8) NOT NULL,
                invested_amount decimal(18,8) NOT NULL,
                current_value decimal(18,8) NOT NULL,
                total_pnl decimal(18,8) NOT NULL,
                total_pnl_percent decimal(10,4) NOT NULL,
                active_positions int NOT NULL,
                successful_trades int NOT NULL,
                total_trades int NOT NULL,
                win_rate decimal(5,2) NOT NULL,
                summary_date datetime2 DEFAULT GETDATE()
            )
        """)
        
        # System state tablosu (persistence için)
        print("💾 system_state tablosu oluşturuluyor...")
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='system_state' AND xtype='U')
            CREATE TABLE system_state (
                id int IDENTITY(1,1) PRIMARY KEY,
                state_key nvarchar(100) UNIQUE NOT NULL,
                state_value nvarchar(MAX),
                last_updated datetime2 DEFAULT GETDATE()
            )
        """)
        
        # Değişiklikleri kaydet
        conn.commit()
        
        print("✅ Tüm tablolar başarıyla oluşturuldu!")
        
        # Test verileri ekle
        print("🧪 Test verileri ekleniyor...")
        
        # Test coinleri ekle
        test_coins = [
            ('BTC', 'Bitcoin'),
            ('ETH', 'Ethereum'), 
            ('BNB', 'Binance Coin')
        ]
        
        for symbol, name in test_coins:
            try:
                cursor.execute("""
                    IF NOT EXISTS (SELECT 1 FROM coins WHERE symbol = ?)
                    INSERT INTO coins (symbol, name) VALUES (?, ?)
                """, symbol, symbol, name)
                print(f"✅ {symbol} eklendi")
            except Exception as e:
                print(f"⚠️ {symbol} eklenemedi: {str(e)}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("🎉 Database kurulumu tamamlandı!")
        return True
        
    except Exception as e:
        print(f"❌ Tablo oluşturma hatası: {str(e)}")
        return False

def test_connection():
    """Database bağlantısını test eder"""
    
    server = os.getenv('MSSQL_SERVER')
    username = os.getenv('MSSQL_USERNAME') 
    password = os.getenv('MSSQL_PASSWORD')
    driver = os.getenv('MSSQL_DRIVER', 'ODBC Driver 17 for SQL Server')
    
    try:
        connection_string = f"""
        DRIVER={{{driver}}};
        SERVER={server};
        DATABASE=crypto_trading_db;
        UID={username};
        PWD={password};
        TrustServerCertificate=yes;
        """
        
        print("🧪 Final bağlantı testi...")
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        
        # Tablo sayısını kontrol et
        cursor.execute("""
            SELECT COUNT(*) FROM sys.tables
        """)
        table_count = cursor.fetchone()[0]
        
        # Coin sayısını kontrol et
        cursor.execute("SELECT COUNT(*) FROM coins")
        coin_count = cursor.fetchone()[0]
        
        print(f"✅ Bağlantı başarılı!")
        print(f"📋 Tablo sayısı: {table_count}")
        print(f"🪙 Coin sayısı: {coin_count}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Test bağlantısı hatası: {str(e)}")
        return False

def main():
    """Ana fonksiyon"""
    print_banner()
    
    print("🔐 Environment variables kontrol ediliyor...")
    
    required_vars = ['MSSQL_SERVER', 'MSSQL_USERNAME', 'MSSQL_PASSWORD']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Eksik environment variables: {', '.join(missing_vars)}")
        print("🔧 .env dosyasını kontrol edin!")
        return False
    
    print("✅ Environment variables tamam!")
    
    # Adım 1: Database oluştur
    print("\n" + "="*60)
    print("🔄 Adım 1: Database oluşturma")
    print("="*60)
    
    if not create_database():
        print("❌ Database oluşturma başarısız!")
        return False
    
    # Adım 2: Tabloları oluştur  
    print("\n" + "="*60)
    print("🔄 Adım 2: Tabloları oluşturma")
    print("="*60)
    
    if not create_tables():
        print("❌ Tablo oluşturma başarısız!")
        return False
    
    # Adım 3: Test bağlantısı
    print("\n" + "="*60)
    print("🔄 Adım 3: Test bağlantısı")
    print("="*60)
    
    if not test_connection():
        print("❌ Test bağlantısı başarısız!")
        return False
    
    print("\n" + "="*60)
    print("🎉 MSSQL DATABASE KURULUMU TAMAMLANDI!")
    print("="*60)
    print("""
✅ Başarıyla oluşturulan:
   • crypto_trading_db database
   • 6 sistem tablosu
   • 3 test coin verisi
   
🚀 Şimdi dashboard'u başlatabilirsiniz:
   python3 run_dashboard.py
   
🌐 Erişim:
   http://localhost:5000
""")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Kurulum başarısız oldu!")
            exit(1)
    except KeyboardInterrupt:
        print("\n\n🔴 Kurulum iptal edildi!")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {str(e)}")
        import traceback
        traceback.print_exc() 