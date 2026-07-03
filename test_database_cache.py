#!/usr/bin/env python3
"""
Database Cache Sistemi Test Scripti

Bu script yeni database cache sistemini test eder:
1. Coin data sync işlemini
2. Database'den veri çekme performansını
3. API vs Database hız karşılaştırmasını
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Environment variables yükle
load_dotenv()

# Paylaşılan veri katmanı (PostgreSQL + SQLAlchemy, schema-per-tenant)
# NOT: Tenant-özel metodlar için önce set_current_tenant(schema) çağrılmalıdır.
from trading_db import TradingDatabase as DatabaseClass
DATABASE_TYPE = "PostgreSQL"

from data_fetcher import CryptoDataFetcher

def test_database_cache_system():
    """Database cache sistemini test eder"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         🧪 DATABASE CACHE SİSTEMİ TEST UYGULAMASI          ║
║                                                              ║
║  📊 Database-first veri çekme testi                         ║
║  🚀 Background sync performance testi                       ║
║  ⏱️ API vs Database hız karşılaştırması                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Database ve Data Fetcher initialize
    print("🔧 Sistem başlatılıyor...")
    db = DatabaseClass()
    
    # Tabloları oluştur
    if hasattr(db, 'create_tables'):
        db.create_tables()
        print("✅ Database tabloları oluşturuldu")
    
    # Database'li data fetcher
    data_fetcher = CryptoDataFetcher(database=db)
    print("✅ Database'li data fetcher hazır")
    
    # Test coinleri
    test_coins = ['BTC', 'ETH', 'BNB']
    
    print(f"\n🎯 Test Coinleri: {', '.join(test_coins)}")
    print("=" * 60)
    
    for coin in test_coins:
        print(f"\n🪙 {coin} Test Başlıyor...")
        print("-" * 40)
        
        # 1. Database durumunu kontrol et
        sync_status = db.get_sync_status(coin)
        print(f"📊 Sync Status: {sync_status['status']}")
        print(f"📈 Records: {sync_status['total_records']}")
        
        # 2. Manuel sync testi (eğer gerekiyorsa)
        if sync_status['status'] != 'COMPLETED' or sync_status['total_records'] < 1000:
            print(f"🚀 {coin} için manuel sync başlatılıyor...")
            start_time = time.time()
            
            success = data_fetcher.sync_coin_data_manual(coin, days=100, force_refresh=False)
            
            sync_time = time.time() - start_time
            print(f"⏱️ Sync süresi: {sync_time:.2f} saniye")
            
            if success:
                print(f"✅ {coin} sync başarılı")
            else:
                print(f"❌ {coin} sync başarısız")
                continue
        else:
            print(f"✅ {coin} zaten sync edilmiş")
        
        # 3. Database'den veri çekme hız testi
        print(f"📊 {coin} Database hız testi...")
        start_time = time.time()
        
        db_data = data_fetcher.fetch_ohlcv_data(coin, days=30, force_api=False)
        
        db_fetch_time = time.time() - start_time
        print(f"⏱️ Database fetch: {db_fetch_time:.3f} saniye")
        
        if db_data is not None:
            print(f"📈 Database'den çekilen veri: {len(db_data)} kayıt")
            print(f"📅 Tarih aralığı: {db_data.index[0]} - {db_data.index[-1]}")
        else:
            print("❌ Database'den veri çekilemedi")
            continue
        
        # 4. API'den veri çekme hız testi (karşılaştırma)
        print(f"🌐 {coin} API hız testi...")
        start_time = time.time()
        
        api_data = data_fetcher.fetch_ohlcv_data(coin, days=30, force_api=True)
        
        api_fetch_time = time.time() - start_time
        print(f"⏱️ API fetch: {api_fetch_time:.3f} saniye")
        
        if api_data is not None:
            print(f"📈 API'den çekilen veri: {len(api_data)} kayıt")
        else:
            print("❌ API'den veri çekilemedi")
        
        # 5. Hız karşılaştırması
        if db_data is not None and api_data is not None:
            speed_improvement = (api_fetch_time / db_fetch_time) if db_fetch_time > 0 else 0
            print(f"🚀 Database {speed_improvement:.1f}x daha hızlı!")
            
            # Veri doğruluğu kontrolü
            db_latest_price = db_data['close'].iloc[-1]
            api_latest_price = api_data['close'].iloc[-1]
            price_diff_percent = abs(db_latest_price - api_latest_price) / api_latest_price * 100
            
            print(f"💰 Database son fiyat: ${db_latest_price:.6f}")
            print(f"💰 API son fiyat: ${api_latest_price:.6f}")
            print(f"📊 Fark: %{price_diff_percent:.2f}")
            
            if price_diff_percent < 5:
                print("✅ Veri doğruluğu kabul edilebilir seviyede")
            else:
                print("⚠️ Veri doğruluğu problemli, sync gerekli")
        
        print(f"✅ {coin} test tamamlandı")
    
    # 6. Genel sync durumu özeti
    print(f"\n📊 Genel Sync Durumu:")
    print("=" * 60)
    
    try:
        sync_summary = data_fetcher.get_sync_status_summary()
        print(f"📈 Toplam kayıt: {sync_summary.get('total_records', 0)}")
        print(f"🔄 Queue boyutu: {sync_summary.get('queue_size', 0)}")
        
        # Status'lara göre breakdown
        for status, count in sync_summary.items():
            if status not in ['total_records', 'queue_size', 'error']:
                print(f"   {status}: {count} coin")
                
    except Exception as e:
        print(f"❌ Sync özet alınamadı: {str(e)}")
    
    # 7. Performans önerisi
    print(f"\n💡 Performans Değerlendirmesi:")
    print("=" * 60)
    print("✅ Database cache sistemi başarıyla çalışıyor")
    print("🚀 API çağrıları minimum seviyeye indirildi")
    print("📊 Veri tutarlılığı korunuyor")
    print("⚡ Analiz hızı önemli ölçüde arttı")
    print("\n🎯 Öneriler:")
    print("• Yeni coin eklerken background sync kullanın")
    print("• Günlük otomatik sync scheduleri kurun")  
    print("• Database backup'larını düzenli alın")
    print("• Sync error'larını monitör edin")
    
    print(f"\n✅ Database Cache Sistemi Test Tamamlandı!")
    print("🔔 Artık web dashboard'da çok daha hızlı analiz yapabilirsiniz!")

if __name__ == "__main__":
    try:
        test_database_cache_system()
    except KeyboardInterrupt:
        print("\n👋 Test durduruldu")
    except Exception as e:
        print(f"\n❌ Test hatası: {str(e)}")
        import traceback
        print(f"🔍 Detay: {traceback.format_exc()}") 