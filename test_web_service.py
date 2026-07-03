#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Service Test Script
Web Service'in AI Service ile iletişim kurabildiğini test eder
"""

import requests
import json
import time
from datetime import datetime

class WebServiceTester:
    def __init__(self, web_url="http://localhost:5000", ai_url="http://localhost:8000"):
        self.web_url = web_url
        self.ai_url = ai_url
        self.session = requests.Session()
        
    def test_web_health(self):
        """Web service health check"""
        print("🌐 Web Service health check...")
        try:
            response = self.session.get(f"{self.web_url}/", timeout=10)
            if response.status_code in [200, 302]:  # 302 redirect to login is OK
                print("✅ Web Service erişilebilir")
                return True
            else:
                print(f"❌ Web Service hatası: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Web Service bağlantı hatası: {e}")
            return False
    
    def test_ai_service_connection(self):
        """Web Service'ten AI Service bağlantısı"""
        print("🔗 AI Service bağlantısı kontrol ediliyor...")
        try:
            response = self.session.get(f"{self.web_url}/api/ai_service_status", timeout=30)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'healthy':
                    print("✅ AI Service bağlantısı başarılı")
                    return True
                else:
                    print(f"❌ AI Service sağlıksız: {result}")
                    return False
            elif response.status_code == 401:
                print("⚠️ Authentication gerekli - bu normal")
                return True  # Login gerekliliği normal
            else:
                print(f"❌ API status hatası: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ AI Service bağlantı testi hatası: {e}")
            return False
    
    def test_login(self, username="admin", password="admin"):
        """Login testi"""
        print("🔑 Login testi...")
        try:
            login_data = {
                'username': username,
                'password': password
            }
            
            response = self.session.post(
                f"{self.web_url}/login",
                data=login_data,
                timeout=10,
                allow_redirects=False
            )
            
            if response.status_code in [200, 302]:
                # Dashboard'a yönlendirme kontrolü
                dashboard_response = self.session.get(f"{self.web_url}/dashboard", timeout=10)
                if dashboard_response.status_code == 200:
                    print("✅ Login başarılı")
                    return True
                else:
                    print(f"❌ Dashboard erişimi başarısız: {dashboard_response.status_code}")
                    return False
            else:
                print(f"❌ Login başarısız: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Login testi hatası: {e}")
            return False
    
    def test_coin_analysis_web(self, coin_symbol="BTC"):
        """Web üzerinden coin analizi"""
        print(f"📊 Web üzerinden {coin_symbol} analizi...")
        try:
            payload = {
                "coin_symbol": coin_symbol,
                "analysis_type": "lstm"
            }
            
            response = self.session.post(
                f"{self.web_url}/api/analyze_coin",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    print(f"✅ {coin_symbol} web analizi başarılı")
                    print(f"   Model tipi: {result.get('model_type')}")
                    return True
                else:
                    print(f"❌ {coin_symbol} web analizi başarısız: {result.get('error')}")
                    return False
            elif response.status_code == 401:
                print("❌ Authentication hatası - login gerekli")
                return False
            else:
                print(f"❌ Web analizi API hatası: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Web analizi testi hatası: {e}")
            return False
    
    def test_model_status_web(self, coin_symbol="BTC"):
        """Web üzerinden model durumu"""
        print(f"🔧 Web üzerinden {coin_symbol} model durumu...")
        try:
            response = self.session.get(
                f"{self.web_url}/api/model_status/{coin_symbol}",
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Model durumu alındı: {coin_symbol}")
                return True
            elif response.status_code == 401:
                print("❌ Authentication hatası - login gerekli")
                return False
            else:
                print(f"❌ Model durumu hatası: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Model durumu testi hatası: {e}")
            return False
    
    def test_direct_ai_service(self):
        """Direkt AI Service bağlantısı (karşılaştırma için)"""
        print("🎯 Direkt AI Service testi...")
        try:
            response = requests.get(f"{self.ai_url}/health", timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'healthy':
                    print("✅ Direkt AI Service bağlantısı OK")
                    return True
            
            print(f"❌ Direkt AI Service bağlantısı başarısız")
            return False
            
        except Exception as e:
            print(f"❌ Direkt AI Service hatası: {e}")
            return False
    
    def run_all_tests(self):
        """Tüm testleri çalıştır"""
        print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                🧪 WEB SERVICE TEST SUITE 🧪                      ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
        
        print(f"🚀 Web Service Test Başlıyor:")
        print(f"   Web URL: {self.web_url}")
        print(f"   AI URL: {self.ai_url}")
        print(f"⏰ Test zamanı: {datetime.now()}")
        
        results = {}
        
        # Test 1: Web Health
        results['web_health'] = self.test_web_health()
        time.sleep(1)
        
        # Test 2: Direct AI Service (karşılaştırma)
        results['direct_ai'] = self.test_direct_ai_service()
        time.sleep(1)
        
        # Test 3: Login
        results['login'] = self.test_login()
        time.sleep(1)
        
        # Eğer login başarılıysa authentication gerektiren testleri çalıştır
        if results['login']:
            # Test 4: AI Service Connection via Web
            results['ai_connection'] = self.test_ai_service_connection()
            time.sleep(1)
            
            # Test 5: Model Status via Web
            results['model_status'] = self.test_model_status_web()
            time.sleep(1)
            
            # Test 6: Coin Analysis via Web
            results['coin_analysis'] = self.test_coin_analysis_web()
        else:
            print("⚠️ Login başarısız - authentication gerektiren testler atlanıyor")
        
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
            print("🎉 Tüm testler başarılı! Web Service hazır.")
        else:
            print("⚠️ Bazı testler başarısız. Lütfen logları kontrol edin.")
        
        return passed == total

def main():
    """Ana test fonksiyonu"""
    import sys
    
    # Service URL'lerini al
    web_url = "http://localhost:5000"
    ai_url = "http://localhost:8000"
    
    if len(sys.argv) > 1:
        web_url = sys.argv[1]
    if len(sys.argv) > 2:
        ai_url = sys.argv[2]
    
    # Tester oluştur ve çalıştır
    tester = WebServiceTester(web_url, ai_url)
    success = tester.run_all_tests()
    
    # Exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
