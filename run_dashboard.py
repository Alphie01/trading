#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kripto Trading Dashboard Başlatıcı

Bu script web dashboard'unu başlatır ve kullanıcı dostu bir arayüz sağlar.
Environment variables, MSSQL ve sistem kontrollerini yapar.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from web_app import main

def print_startup_banner():
    """Başlangıç banner'ı"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║            🌐 KRİPTO TRADİNG DASHBOARD WEB ARAYÜZÜ 🌐             ║
║                                                                    ║
║  📊 Çoklu coin izleme ve analiz                                   ║
║  💰 İşlem geçmişi ve kar/zarar takibi                            ║
║  📈 Gerçek zamanlı portfolio yönetimi                            ║
║  🤖 Otomatik LSTM trading sistemi                                ║
║  📱 Modern ve kullanıcı dostu web arayüzü                        ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    time.sleep(2)

def check_environment_setup():
    """Environment variables kurulumunu kontrol eder"""
    print("🔐 Environment variables kontrol ediliyor...")
    
    # .env dosyası var mı?
    if not os.path.exists('.env'):
        print("⚠️ .env dosyası bulunamadı!")
        if os.path.exists('.env.example'):
            print("📝 .env.example'dan .env oluşturuluyor...")
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ .env dosyası oluşturuldu")
            print("⚙️ Lütfen .env dosyasındaki ayarları düzenleyin!")
        return False
    
    # PostgreSQL bağlantısı kontrolü
    if os.getenv('DATABASE_URL'):
        print("✅ DATABASE_URL tanımlı (PostgreSQL)")
    else:
        print("⚠️ DATABASE_URL tanımlı değil — trading_db bağlantısı başarısız olabilir")

    return True

def check_requirements():
    """Gerekli kütüphaneleri kontrol et"""
    print("📦 Gerekli kütüphaneler kontrol ediliyor...")
    
    required_packages = [
        'flask', 'pandas', 'numpy', 'ccxt', 'tensorflow', 'dotenv',
        'trading_db', 'sqlalchemy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - EKSIK!")
            missing_packages.append(package)

    if 'trading_db' in missing_packages:
        print("📦 Paylaşılan veri katmanı: pip install -e packages/database")

    if missing_packages:
        print(f"\n⚠️ Eksik kütüphaneler bulundu: {', '.join(missing_packages)}")
        print("📦 Kurulum için: pip install -r requirements.txt")
        return False
    
    print("✅ Tüm gerekli kütüphaneler mevcut!")
    return True

def show_access_info():
    """Erişim bilgilerini göster"""
    print("""
🌐 WEB DASHBOARD ERİŞİM BİLGİLERİ:

   📱 Ana Dashboard:    http://localhost:5000
   📊 Portfolio:        http://localhost:5000/portfolio  
   ⚙️ Ayarlar:          http://localhost:5000/settings

🔧 ÖZELLİKLER:
   • Coin ekleme/çıkarma
   • Gerçek zamanlı analiz
   • İşlem takibi
   • Kar/zarar analizi
   • Otomatik trading
   • Model cache sistemi

⚠️ UYARI:
   • İlk kullanımda ayarlar sayfasından API anahtarlarını yapılandırın
   • Otomatik trading için Binance API anahtarı gereklidir
   • Haber analizi için NewsAPI anahtarı önerilir
   • Whale takibi için Whale Alert API anahtarı önerilir

🔴 DURDURMA: Ctrl+C tuşlarına basın
""")

if __name__ == "__main__":
    try:
        # Banner göster
        print_startup_banner()
        
        # Environment setup kontrol et
        if not check_environment_setup():
            print("\n❌ Environment kurulumu tamamlanmadı.")
            print("🔧 .env dosyasını düzenleyip tekrar çalıştırın.")
            sys.exit(1)
        
        # Kütüphaneleri kontrol et
        if not check_requirements():
            print("\n❌ Gerekli kütüphaneler eksik. Çıkılıyor...")
            sys.exit(1)
        
        # Sistem bilgilerini göster
        print(f"\n📋 Sistem Konfigürasyonu:")
        print(f"   🗄️ Database: {'MSSQL' if os.getenv('MSSQL_SERVER') else 'SQLite'}")
        print(f"   🌐 Host: {os.getenv('FLASK_HOST', '0.0.0.0')}")
        print(f"   🚪 Port: {os.getenv('FLASK_PORT', '5000')}")
        
        # Erişim bilgilerini göster
        show_access_info()
        
        # Web dashboard'unu başlat
        print("🚀 Web dashboard başlatılıyor...")
        print("⏳ Lütfen bekleyin...")
        
        main()
        
    except KeyboardInterrupt:
        print("\n\n🔴 Dashboard durduruldu. Görüşürüz! 👋")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Başlatma hatası: {str(e)}")
        print("💡 Lütfen requirements.txt'yi kontrol edin ve gerekli kütüphaneleri kurun.")
        import traceback
        traceback.print_exc()
        sys.exit(1) 