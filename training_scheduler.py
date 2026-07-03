#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Scheduler

Haftalık otomatik fine-tune sistemini yöneten modül.
Cache edilmiş modellerin üzerine fine-tune işlemi yapar.
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import schedule
import logging

from comprehensive_trainer import ComprehensiveTrainer

try:
    from trading_db import TradingDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

class TrainingScheduler:
    """
    Haftalık fine-tune scheduler sınıfı
    """
    
    def __init__(self, 
                 schedule_day: str = "sunday",
                 schedule_time: str = "02:00",
                 enable_notifications: bool = True):
        """
        Training Scheduler'ı başlatır
        
        Args:
            schedule_day: Hangi gün çalışacağı (default: sunday)
            schedule_time: Hangi saatte çalışacağı (default: 02:00)
            enable_notifications: Bildirimler aktif mi
        """
        self.schedule_day = schedule_day.lower()
        self.schedule_time = schedule_time
        self.enable_notifications = enable_notifications
        self.is_running = False
        self.scheduler_thread = None
        
        # Comprehensive trainer
        self.trainer = ComprehensiveTrainer()
        
        # Database connection
        if DATABASE_AVAILABLE:
            self.db = TradingDatabase()
        else:
            self.db = None
        
        # Tracked coins
        self.tracked_coins = set()
        
        # Scheduler ayarları
        self.max_concurrent_training = 3  # Aynı anda max 3 coin eğit
        self.training_timeout = 7200  # 2 saat timeout
        
        # Log setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training_scheduler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        print("⏰ Training Scheduler başlatıldı!")
        print(f"📅 Schedule: Her {schedule_day} {schedule_time}")
        print(f"💾 Database: {'✅' if self.db else '❌'}")
        
        # Load tracked coins
        self._load_tracked_coins()
        
        # Set up schedule
        self._setup_schedule()
    
    def _load_tracked_coins(self):
        """Takip edilen coinleri yükler"""
        try:
            if self.db:
                # Database'den coin listesi al
                coins = self.db.get_tracked_coins()
                self.tracked_coins = set(coins) if coins else set()
            else:
                # JSON dosyasından yükle
                if os.path.exists("tracked_coins.json"):
                    with open("tracked_coins.json", 'r') as f:
                        data = json.load(f)
                        self.tracked_coins = set(data.get('coins', []))
                else:
                    self.tracked_coins = set()
            
            print(f"📊 Takip edilen coinler yüklendi: {len(self.tracked_coins)} coin")
            if self.tracked_coins:
                print(f"   Coinler: {', '.join(list(self.tracked_coins)[:10])}{'...' if len(self.tracked_coins) > 10 else ''}")
                
        except Exception as e:
            self.logger.error(f"Tracked coins yükleme hatası: {e}")
            self.tracked_coins = set()
    
    def _setup_schedule(self):
        """Schedule ayarlarını yapar"""
        try:
            # Haftalık schedule
            if self.schedule_day == "monday":
                schedule.every().monday.at(self.schedule_time).do(self._run_weekly_training)
            elif self.schedule_day == "tuesday":
                schedule.every().tuesday.at(self.schedule_time).do(self._run_weekly_training)
            elif self.schedule_day == "wednesday":
                schedule.every().wednesday.at(self.schedule_time).do(self._run_weekly_training)
            elif self.schedule_day == "thursday":
                schedule.every().thursday.at(self.schedule_time).do(self._run_weekly_training)
            elif self.schedule_day == "friday":
                schedule.every().friday.at(self.schedule_time).do(self._run_weekly_training)
            elif self.schedule_day == "saturday":
                schedule.every().saturday.at(self.schedule_time).do(self._run_weekly_training)
            else:  # sunday (default)
                schedule.every().sunday.at(self.schedule_time).do(self._run_weekly_training)
            
            print(f"✅ Schedule ayarlandı: Her {self.schedule_day} {self.schedule_time}")
            
        except Exception as e:
            self.logger.error(f"Schedule ayarlama hatası: {e}")
    
    def add_coin_to_schedule(self, coin_symbol: str):
        """Schedule'a yeni coin ekler"""
        try:
            coin_symbol = coin_symbol.upper()
            self.tracked_coins.add(coin_symbol)
            self._save_tracked_coins()
            
            print(f"➕ {coin_symbol} haftalık schedule'a eklendi")
            self.logger.info(f"Coin added to schedule: {coin_symbol}")
            
        except Exception as e:
            self.logger.error(f"Coin ekleme hatası: {e}")
    
    def remove_coin_from_schedule(self, coin_symbol: str):
        """Schedule'dan coin çıkarır"""
        try:
            coin_symbol = coin_symbol.upper()
            self.tracked_coins.discard(coin_symbol)
            self._save_tracked_coins()
            
            print(f"➖ {coin_symbol} haftalık schedule'dan çıkarıldı")
            self.logger.info(f"Coin removed from schedule: {coin_symbol}")
            
        except Exception as e:
            self.logger.error(f"Coin çıkarma hatası: {e}")
    
    def _save_tracked_coins(self):
        """Takip edilen coinleri kaydeder"""
        try:
            if self.db:
                # Database'e kaydet (db method gerekli)
                pass
            
            # JSON backup
            os.makedirs("scheduler_data", exist_ok=True)
            with open("scheduler_data/tracked_coins.json", 'w') as f:
                json.dump({
                    'coins': list(self.tracked_coins),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Tracked coins kaydetme hatası: {e}")
    
    def start_scheduler(self):
        """Scheduler'ı başlatır"""
        if self.is_running:
            print("⚠️ Scheduler zaten çalışıyor!")
            return
        
        self.is_running = True
        
        def run_scheduler():
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Her dakika kontrol et
                except Exception as e:
                    self.logger.error(f"Scheduler run error: {e}")
                    time.sleep(300)  # Hata durumunda 5 dakika bekle
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        print("✅ Training Scheduler başlatıldı!")
        print(f"⏰ Sonraki çalışma: {self._get_next_run_time()}")
        self.logger.info("Training Scheduler started")
    
    def stop_scheduler(self):
        """Scheduler'ı durdurur"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        print("🛑 Training Scheduler durduruldu!")
        self.logger.info("Training Scheduler stopped")
    
    def _get_next_run_time(self) -> str:
        """Sonraki çalışma zamanını döndürür"""
        try:
            jobs = schedule.jobs
            if jobs:
                next_run = min(job.next_run for job in jobs)
                return next_run.strftime('%Y-%m-%d %H:%M:%S')
            return "Bilinmiyor"
        except:
            return "Bilinmiyor"
    
    def _run_weekly_training(self):
        """Haftalık fine-tune training'i çalıştırır"""
        try:
            start_time = datetime.now()
            self.logger.info("=== HAFTALIK FINE-TUNE BAŞLADI ===")
            print(f"\n🔥 Haftalık Fine-tune başlıyor: {start_time}")
            
            if not self.tracked_coins:
                self.logger.warning("Takip edilen coin bulunamadı, training atlanıyor")
                return
            
            # Training sonuçları
            training_results = {
                'start_time': start_time.isoformat(),
                'coins_attempted': list(self.tracked_coins),
                'successful_coins': [],
                'failed_coins': [],
                'results': {}
            }
            
            # Her coin için fine-tune yap
            for coin_symbol in self.tracked_coins:
                try:
                    print(f"\n🔄 {coin_symbol} fine-tune başlıyor...")
                    self.logger.info(f"Starting fine-tune for {coin_symbol}")
                    
                    # Cache'de mevcut model var mı kontrol et
                    if not self._has_existing_model(coin_symbol):
                        print(f"⚠️ {coin_symbol} için mevcut model bulunamadı, ilk eğitim yapılacak")
                        self.logger.warning(f"No existing model for {coin_symbol}, doing first training")
                        
                        # İlk eğitim yap
                        result = self.trainer.train_coin_sync(coin_symbol, is_fine_tune=False)
                    else:
                        print(f"📂 {coin_symbol} için mevcut model bulundu, fine-tune yapılıyor")
                        self.logger.info(f"Existing model found for {coin_symbol}, doing fine-tune")
                        
                        # Fine-tune yap
                        result = self.trainer.train_coin_sync(coin_symbol, is_fine_tune=True)
                    
                    if result.get('success', False):
                        training_results['successful_coins'].append(coin_symbol)
                        training_results['results'][coin_symbol] = {
                            'status': 'success',
                            'successful_models': result.get('successful_models', []),
                            'failed_models': result.get('failed_models', []),
                            'predictions_4h': len(result.get('predictions', {}).get('4h', {})),
                            'predictions_1d': len(result.get('predictions', {}).get('1d', {}))
                        }
                        print(f"✅ {coin_symbol} fine-tune başarılı!")
                        self.logger.info(f"Fine-tune successful for {coin_symbol}")
                    else:
                        training_results['failed_coins'].append(coin_symbol)
                        training_results['results'][coin_symbol] = {
                            'status': 'failed',
                            'error': result.get('error', 'Bilinmeyen hata')
                        }
                        print(f"❌ {coin_symbol} fine-tune başarısız: {result.get('error')}")
                        self.logger.error(f"Fine-tune failed for {coin_symbol}: {result.get('error')}")
                        
                except Exception as e:
                    training_results['failed_coins'].append(coin_symbol)
                    training_results['results'][coin_symbol] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"❌ {coin_symbol} fine-tune exception: {e}")
                    self.logger.error(f"Fine-tune exception for {coin_symbol}: {e}")
            
            # Training tamamlandı
            end_time = datetime.now()
            duration = end_time - start_time
            training_results['end_time'] = end_time.isoformat()
            training_results['duration_minutes'] = duration.total_seconds() / 60
            
            # Sonuçları kaydet
            self._save_weekly_training_results(training_results)
            
            # Özet rapor
            self._print_training_summary(training_results)
            
            self.logger.info("=== HAFTALIK FINE-TUNE TAMAMLANDI ===")
            
        except Exception as e:
            self.logger.error(f"Haftalık training genel hatası: {e}")
            print(f"❌ Haftalık training genel hatası: {e}")
    
    def _has_existing_model(self, coin_symbol: str) -> bool:
        """Coin için mevcut model var mı kontrol eder"""
        try:
            model_files = [
                f"model_cache/lstm_{coin_symbol.lower()}_comprehensive.h5",
                f"model_cache/dqn_{coin_symbol.lower()}_comprehensive.h5",
                f"model_cache/hybrid_{coin_symbol.lower()}_comprehensive.h5"
            ]
            
            # En az bir model dosyası varsa True döndür
            return any(os.path.exists(file) for file in model_files)
            
        except Exception as e:
            self.logger.error(f"Model varlık kontrolü hatası: {e}")
            return False
    
    def _save_weekly_training_results(self, results: Dict):
        """Haftalık training sonuçlarını kaydeder"""
        try:
            # JSON dosyası olarak kaydet
            os.makedirs("scheduler_data/weekly_results", exist_ok=True)
            filename = f"scheduler_data/weekly_results/weekly_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 Haftalık sonuçlar kaydedildi: {filename}")
            self.logger.info(f"Weekly training results saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Haftalık sonuç kaydetme hatası: {e}")
    
    def _print_training_summary(self, results: Dict):
        """Training özet raporunu yazdırır"""
        try:
            print(f"\n📊 HAFTALIK FINE-TUNE RAPORU")
            print("="*50)
            print(f"⏰ Başlangıç: {results['start_time']}")
            print(f"⏱️ Süre: {results['duration_minutes']:.1f} dakika")
            print(f"🎯 Toplam coin: {len(results['coins_attempted'])}")
            print(f"✅ Başarılı: {len(results['successful_coins'])}")
            print(f"❌ Başarısız: {len(results['failed_coins'])}")
            
            if results['successful_coins']:
                print(f"\n✅ Başarılı Coinler:")
                for coin in results['successful_coins']:
                    coin_result = results['results'][coin]
                    print(f"   {coin}: {len(coin_result['successful_models'])} model, "
                          f"4h={coin_result['predictions_4h']}, 1d={coin_result['predictions_1d']}")
            
            if results['failed_coins']:
                print(f"\n❌ Başarısız Coinler:")
                for coin in results['failed_coins']:
                    error = results['results'][coin].get('error', 'Bilinmeyen')
                    print(f"   {coin}: {error[:100]}...")
            
            print("="*50)
            
        except Exception as e:
            self.logger.error(f"Özet rapor yazdırma hatası: {e}")
    
    def get_scheduler_status(self) -> Dict:
        """Scheduler durumunu döndürür"""
        return {
            'is_running': self.is_running,
            'schedule_day': self.schedule_day,
            'schedule_time': self.schedule_time,
            'tracked_coins_count': len(self.tracked_coins),
            'tracked_coins': list(self.tracked_coins),
            'next_run_time': self._get_next_run_time()
        }
    
    def force_run_training(self, coin_symbol: Optional[str] = None):
        """Manual olarak training'i başlatır"""
        try:
            if coin_symbol:
                # Tek coin için
                coin_symbol = coin_symbol.upper()
                print(f"🔥 Manual fine-tune başlıyor: {coin_symbol}")
                self.logger.info(f"Manual training started for {coin_symbol}")
                
                if self._has_existing_model(coin_symbol):
                    result = self.trainer.train_coin_sync(coin_symbol, is_fine_tune=True)
                else:
                    result = self.trainer.train_coin_sync(coin_symbol, is_fine_tune=False)
                
                if result.get('success', False):
                    print(f"✅ {coin_symbol} manual training başarılı!")
                else:
                    print(f"❌ {coin_symbol} manual training başarısız: {result.get('error')}")
                
                return result
            else:
                # Tüm coinler için
                print(f"🔥 Manual haftalık training başlıyor...")
                self.logger.info("Manual weekly training started")
                self._run_weekly_training()
                
        except Exception as e:
            print(f"❌ Manual training hatası: {e}")
            self.logger.error(f"Manual training error: {e}")
            return {'success': False, 'error': str(e)}

# Singleton scheduler instance
_scheduler_instance = None

def get_scheduler() -> TrainingScheduler:
    """Global scheduler instance döndürür"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = TrainingScheduler()
    return _scheduler_instance

def init_scheduler(schedule_day: str = "sunday", 
                  schedule_time: str = "02:00",
                  enable_notifications: bool = True) -> TrainingScheduler:
    """Scheduler'ı initialize eder"""
    global _scheduler_instance
    _scheduler_instance = TrainingScheduler(
        schedule_day=schedule_day,
        schedule_time=schedule_time,
        enable_notifications=enable_notifications
    )
    return _scheduler_instance

if __name__ == "__main__":
    # Test
    scheduler = TrainingScheduler()
    scheduler.add_coin_to_schedule("BTC")
    scheduler.add_coin_to_schedule("ETH")
    
    print("Scheduler durumu:", scheduler.get_scheduler_status())
    
    # Manual test
    # scheduler.force_run_training("BTC") 