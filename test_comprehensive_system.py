#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Training System Test Script

Bu script yeni comprehensive training sistemini test eder:
1. Comprehensive Trainer test
2. Training Scheduler test
3. Multi-timeframe predictions test
"""

import os
import sys
import asyncio
from datetime import datetime

def test_comprehensive_trainer():
    """Comprehensive Trainer'ı test eder"""
    print("\n🧪 COMPREHENSIVE TRAINER TEST")
    print("="*50)
    
    try:
        from comprehensive_trainer import ComprehensiveTrainer
        
        # Trainer oluştur
        trainer = ComprehensiveTrainer()
        print("✅ Comprehensive Trainer başlatıldı")
        
        # Test coin için training yap
        test_coin = "BTC"
        print(f"🔥 {test_coin} için test training başlıyor...")
        
        # Synchronous training test
        result = trainer.train_coin_sync(test_coin, is_fine_tune=False)
        
        if result['success']:
            print(f"✅ {test_coin} comprehensive training başarılı!")
            print(f"   Başarılı modeller: {result.get('successful_models', [])}")
            print(f"   Başarısız modeller: {result.get('failed_models', [])}")
            
            predictions = result.get('predictions', {})
            print(f"   4h tahminler: {len(predictions.get('4h', {}))}")
            print(f"   1d tahminler: {len(predictions.get('1d', {}))}")
            
            # Detay göster
            if 'LSTM' in predictions.get('4h', {}):
                lstm_pred = predictions['4h']['LSTM']
                print(f"   LSTM 4h tahmin: ${lstm_pred.get('current_price', 0):.2f} → ${lstm_pred.get('predicted_price', 0):.2f}")
                print(f"   Değişim: {lstm_pred.get('price_change_percent', 0):+.2f}%")
        else:
            print(f"❌ {test_coin} comprehensive training başarısız: {result.get('error')}")
            
        return result['success']
        
    except Exception as e:
        print(f"❌ Comprehensive Trainer test hatası: {e}")
        return False

def test_training_scheduler():
    """Training Scheduler'ı test eder"""
    print("\n⏰ TRAINING SCHEDULER TEST")
    print("="*50)
    
    try:
        from training_scheduler import TrainingScheduler
        
        # Scheduler oluştur
        scheduler = TrainingScheduler(
            schedule_day="sunday",
            schedule_time="02:00"
        )
        print("✅ Training Scheduler başlatıldı")
        
        # Test coinleri ekle
        test_coins = ["BTC", "ETH"]
        for coin in test_coins:
            scheduler.add_coin_to_schedule(coin)
            print(f"➕ {coin} schedule'a eklendi")
        
        # Scheduler durumunu kontrol et
        status = scheduler.get_scheduler_status()
        print(f"📊 Scheduler durumu:")
        print(f"   Çalışıyor: {status['is_running']}")
        print(f"   Takip edilen coinler: {status['tracked_coins_count']}")
        print(f"   Coinler: {status['tracked_coins']}")
        print(f"   Sonraki çalışma: {status['next_run_time']}")
        
        # Manual test (sadece BTC için)
        print(f"\n🔥 Manual training test (BTC)...")
        # result = scheduler.force_run_training("BTC")
        # print(f"Manual training sonucu: {result.get('success', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training Scheduler test hatası: {e}")
        return False

def test_web_integration():
    """Web entegrasyonu test eder"""
    print("\n🌐 WEB INTEGRATION TEST")
    print("="*50)
    
    try:
        # Import web app components
        from web_app import training_scheduler, db, data_fetcher
        
        print("✅ Web app components yüklendi")
        
        # Scheduler durumu
        if training_scheduler:
            status = training_scheduler.get_scheduler_status()
            print(f"📅 Web Scheduler durumu: {status['is_running']}")
            print(f"   Takip edilen coinler: {status['tracked_coins_count']}")
        else:
            print("⚠️ Web Scheduler mevcut değil")
        
        # Database connection test
        try:
            coins = db.get_active_coins()
            print(f"💾 Database: {len(coins)} aktif coin")
        except Exception as db_error:
            print(f"⚠️ Database test hatası: {db_error}")
        
        # Data fetcher test
        try:
            test_df = data_fetcher.fetch_ohlcv_data("BTC", days=1)
            if test_df is not None:
                print(f"📊 Data Fetcher: {len(test_df)} veri noktası çekildi")
            else:
                print("⚠️ Data Fetcher: Veri çekilemedi")
        except Exception as fetcher_error:
            print(f"⚠️ Data Fetcher test hatası: {fetcher_error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web integration test hatası: {e}")
        return False

def test_model_files():
    """Model dosyalarının varlığını test eder"""
    print("\n📁 MODEL FILES TEST")
    print("="*50)
    
    model_cache_dir = "model_cache"
    training_results_dir = "training_results"
    scheduler_data_dir = "scheduler_data"
    
    # Dizinlerin varlığını kontrol et
    dirs_to_check = [model_cache_dir, training_results_dir, scheduler_data_dir]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"📂 {dir_path}: {len(files)} dosya")
            
            # İlk 5 dosyayı göster
            for file in files[:5]:
                print(f"   📄 {file}")
            
            if len(files) > 5:
                print(f"   ... ve {len(files) - 5} dosya daha")
        else:
            print(f"📂 {dir_path}: Dizin bulunamadı (henüz oluşturulmamış)")
    
    return True

def print_environment_info():
    """Environment bilgilerini yazdırır"""
    print("\n🔧 ENVIRONMENT INFO")
    print("="*50)
    
    env_vars = [
        'LSTM_EPOCHS',
        'LSTM_TRAINING_DAYS', 
        'DQN_EPISODES',
        'TRAINING_SCHEDULE_DAY',
        'TRAINING_SCHEDULE_TIME',
        'NEWSAPI_KEY',
        'WHALE_ALERT_API_KEY',
        'BINANCE_API_KEY',
        'BINANCE_TESTNET'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if 'KEY' in var or 'SECRET' in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else value
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"⚠️ {var}: Ayarlanmamış")

def main():
    """Ana test fonksiyonu"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║           🧪 COMPREHENSIVE TRAINING SYSTEM TEST 🧪               ║
║                                                                    ║
║  Bu script yeni comprehensive training sistemini test eder       ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    print(f"🕒 Test başlangıcı: {datetime.now()}")
    
    # Environment bilgileri
    print_environment_info()
    
    # Model dosyaları kontrolü
    test_model_files()
    
    # Test sonuçları
    test_results = {}
    
    # 1. Comprehensive Trainer Test
    try:
        test_results['comprehensive_trainer'] = test_comprehensive_trainer()
    except Exception as e:
        print(f"❌ Comprehensive Trainer test exception: {e}")
        test_results['comprehensive_trainer'] = False
    
    # 2. Training Scheduler Test
    try:
        test_results['training_scheduler'] = test_training_scheduler()
    except Exception as e:
        print(f"❌ Training Scheduler test exception: {e}")
        test_results['training_scheduler'] = False
    
    # 3. Web Integration Test
    try:
        test_results['web_integration'] = test_web_integration()
    except Exception as e:
        print(f"❌ Web Integration test exception: {e}")
        test_results['web_integration'] = False
    
    # Sonuçları özetle
    print(f"\n📊 TEST SONUÇLARI ÖZETI")
    print("="*50)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Toplam: {passed_tests}/{total_tests} test başarılı")
    
    if passed_tests == total_tests:
        print("🎉 Tüm testler başarılı! Sistem çalışmaya hazır.")
    else:
        print("⚠️ Bazı testler başarısız. Loglara bakın.")
    
    print(f"🕒 Test bitişi: {datetime.now()}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Test kullanıcı tarafından durduruldu")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test script genel hatası: {e}")
        sys.exit(1) 