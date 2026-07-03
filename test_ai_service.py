#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Service Test Script
AI Service'in düzgün çalışıp çalışmadığını test eder
"""

import requests
import json
import time
from datetime import datetime

class AIServiceTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_check(self):
        """Health check testi"""
        print("🔍 AI Service health check...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Health check başarılı: {result['status']}")
                return True
            else:
                print(f"❌ Health check başarısız: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check hatası: {e}")
            return False
    
    def test_analyze_coin(self, coin_symbol="BTC"):
        """Coin analizi testi"""
        print(f"📊 {coin_symbol} analizi test ediliyor...")
        try:
            payload = {
                "coin_symbol": coin_symbol,
                "analysis_type": "lstm",  # Basit test için
                "use_news": False,
                "use_whale": False
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    print(f"✅ {coin_symbol} analizi başarılı")
                    print(f"   Mevcut fiyat: ${result.get('current_price', 0):.2f}")
                    print(f"   Model tipi: {result.get('model_type')}")
                    return True
                else:
                    print(f"❌ {coin_symbol} analizi başarısız: {result.get('error')}")
                    return False
            else:
                print(f"❌ Analiz API hatası: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Analiz testi hatası: {e}")
            return False
    
    def test_model_status(self, coin_symbol="BTC"):
        """Model durumu testi"""
        print(f"🔧 {coin_symbol} model durumu kontrol ediliyor...")
        try:
            response = self.session.get(
                f"{self.base_url}/models/{coin_symbol}",
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Model durumu alındı: {coin_symbol}")
                
                models = result.get('models', {})
                for model_type, status in models.items():
                    exists = "✅" if status.get('exists', False) else "❌"
                    print(f"   {model_type}: {exists}")
                
                return True
            else:
                print(f"❌ Model durumu alınamadı: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Model durumu hatası: {e}")
            return False
    
    def test_training(self, coin_symbol="BTC"):
        """Model eğitimi testi (background)"""
        print(f"🎯 {coin_symbol} model eğitimi test ediliyor...")
        try:
            payload = {
                "coin_symbol": coin_symbol,
                "training_type": "lstm",
                "is_fine_tune": False,
                "epochs": 5,  # Hızlı test için az epoch
                "data_days": 50
            }
            
            response = self.session.post(
                f"{self.base_url}/train",
                json=payload,
                timeout=10  # Background olduğu için kısa timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    print(f"✅ {coin_symbol} eğitimi başlatıldı")
                    print(f"   Background: {result.get('results', {}).get('background', False)}")
                    return True
                else:
                    print(f"❌ {coin_symbol} eğitimi başarısız: {result.get('error')}")
                    return False
            else:
                print(f"❌ Eğitim API hatası: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Eğitim testi hatası: {e}")
            return False
    
    def run_all_tests(self):
        """Tüm testleri çalıştır"""
        print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                🧪 AI SERVICE TEST SUITE 🧪                       ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
        
        print(f"🚀 AI Service Test Başlıyor: {self.base_url}")
        print(f"⏰ Test zamanı: {datetime.now()}")
        
        results = {}
        
        # Test 1: Health Check
        results['health_check'] = self.test_health_check()
        time.sleep(1)
        
        # Test 2: Model Status
        results['model_status'] = self.test_model_status()
        time.sleep(1)
        
        # Test 3: Coin Analysis
        results['coin_analysis'] = self.test_analyze_coin()
        time.sleep(1)
        
        # Test 4: Training (Optional - comment out if not needed)
        # results['training'] = self.test_training()
        
        # Sonuçları özetle
        print("\n" + "="*60)
        print("📋 TEST SONUÇLARI")
        print("="*60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:20} : {status}")
            if result:
                passed += 1
        
        print(f"\n🎯 Genel Başarı: {passed}/{total} ({(passed/total)*100:.1f}%)")
        
        if passed == total:
            print("🎉 Tüm testler başarılı! AI Service hazır.")
        else:
            print("⚠️ Bazı testler başarısız. Lütfen logları kontrol edin.")
        
        return passed == total

def main():
    """Ana test fonksiyonu"""
    import sys
    
    # AI Service URL'i al
    ai_service_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        ai_service_url = sys.argv[1]
    
    # Tester oluştur ve çalıştır
    tester = AIServiceTester(ai_service_url)
    success = tester.run_all_tests()
    
    # Exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
