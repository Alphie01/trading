#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Variables ve MSSQL Test Script

Bu script şunları test eder:
- Environment variables kurulumu
- MSSQL Server bağlantısı
- System persistence çalışması
"""

import os
import sys
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def test_environment_variables():
    """Environment variables'ları test eder"""
    print("🔐 Environment Variables Test")
    print("=" * 50)
    
    required_vars = {
        'MSSQL_SERVER': 'MSSQL sunucu adresi',
        'MSSQL_DATABASE': 'MSSQL database adı', 
        'MSSQL_USERNAME': 'MSSQL kullanıcı adı',
        'MSSQL_PASSWORD': 'MSSQL şifresi',
        'FLASK_SECRET_KEY': 'Flask güvenlik anahtarı'
    }
    
    missing_required = []
    
    print("📋 Gerekli Variables:")
    for var, desc in required_vars.items():
        value = os.getenv(var)
        if value:
            if 'PASSWORD' in var or 'SECRET' in var or 'KEY' in var:
                display_value = f"{'*' * (len(value) - 4)}{value[-4:]}" if len(value) > 4 else "****"
            else:
                display_value = value
            print(f"   ✅ {var}: {display_value}")
        else:
            print(f"   ❌ {var}: EKSIK! ({desc})")
            missing_required.append(var)
    
    if missing_required:
        print(f"\n❌ Eksik gerekli variables: {', '.join(missing_required)}")
        return False
    else:
        print("\n✅ Tüm gerekli environment variables mevcut!")
        return True

def test_mssql_connection():
    """MSSQL bağlantısını test eder"""
    print("\n🗄️ MSSQL Connection Test")
    print("=" * 50)
    
    mssql_server = os.getenv('MSSQL_SERVER')
    if not mssql_server:
        print("⚠️ MSSQL_SERVER ayarlanmamış, test atlanıyor")
        return False
    
    try:
        print(f"📍 Server: {mssql_server}")
        print(f"🏪 Database: {os.getenv('MSSQL_DATABASE')}")
        print(f"👤 Username: {os.getenv('MSSQL_USERNAME')}")
        
        from mssql_database import MSSQLTradingDatabase
        
        print("🔗 Bağlantı test ediliyor...")
        db = MSSQLTradingDatabase()
        
        if db.test_connection():
            print("✅ MSSQL bağlantısı başarılı!")
            
            print("🧪 Basit test işlemleri yapılıyor...")
            test_result = db.add_coin('TEST', 'Test Coin')
            if test_result:
                print("   ✅ Test coin ekleme başarılı")
                coins = db.get_active_coins()
                print(f"   📋 Aktif coin sayısı: {len(coins)}")
            
            return True
        else:
            print("❌ MSSQL bağlantısı başarısız!")
            return False
            
    except Exception as e:
        print(f"❌ MSSQL test hatası: {str(e)}")
        return False

def test_system_persistence():
    """System persistence'ı test eder"""
    print("\n💾 System Persistence Test")
    print("=" * 50)
    
    try:
        from system_persistence import SystemPersistence
        
        print("🔧 SystemPersistence başlatılıyor...")
        persistence = SystemPersistence()
        
        print(f"   ✅ Database type: {persistence.db_type}")
        print(f"   🔧 Session ID: {persistence.session_id}")
        
        print("💾 Test state kaydetme...")
        test_coins = ['BTC', 'ETH', 'TEST']
        persistence.save_monitoring_state(
            is_active=True,
            interval_minutes=15,
            active_coins=test_coins,
            session_info={'test_mode': True}
        )
        print("   ✅ Monitoring state kaydedildi")
        
        print("📖 Test state yükleme...")
        loaded_state = persistence.load_monitoring_state()
        print(f"   ✅ State yüklendi: {len(loaded_state['active_coins'])} coin")
        
        return True
        
    except Exception as e:
        print(f"❌ Persistence test hatası: {str(e)}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                🧪 ENVIRONMENT & MSSQL TEST SCRIPT 🧪                ║
║                                                                      ║
║  Bu script sistemin doğru kurulumunu test eder                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    all_tests_passed = True
    
    if not test_environment_variables():
        all_tests_passed = False
    
    if not test_mssql_connection():
        all_tests_passed = False
    
    if not test_system_persistence():
        all_tests_passed = False
    
    print("\n" + "=" * 70)
    if all_tests_passed:
        print("✅ TÜM TESTLER BAŞARILI!")
        print("🚀 Sistem kurulumu tamamlandı")
        print("📱 Başlatmak için: python run_dashboard.py")
    else:
        print("❌ BAZI TESTLER BAŞARISIZ!")
        print("🔧 Lütfen hataları düzeltip tekrar test edin")

if __name__ == "__main__":
    main()
