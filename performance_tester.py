#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kripto Para LSTM Tahmin Sistemi - Gerçek Zamanlı Performans Testi

Bu modül sisteminizin tahminlerini 24 saat boyunca gerçek fiyatlarla 
karşılaştırarak performansını test eder.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import threading
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import CryptoDataFetcher
from data_preprocessor import CryptoDataPreprocessor
from lstm_model import CryptoLSTMModel
from predictor import CryptoPricePredictor
from news_analyzer import CryptoNewsAnalyzer
from whale_tracker import CryptoWhaleTracker

class CryptoPerformanceTester:
    """
    Gerçek zamanlı performans test sistemi
    """
    
    def __init__(self, coin_symbol, test_duration_hours=24, prediction_interval_minutes=60):
        """
        Performance tester'ı başlatır
        
        Args:
            coin_symbol (str): Test edilecek coin sembolü
            test_duration_hours (int): Test süresi (saat)
            prediction_interval_minutes (int): Tahmin aralığı (dakika)
        """
        self.coin_symbol = coin_symbol.upper()
        self.test_duration_hours = test_duration_hours
        self.prediction_interval_minutes = prediction_interval_minutes
        self.prediction_interval_seconds = prediction_interval_minutes * 60
        
        # Test verileri
        self.test_results = []
        self.performance_metrics = {}
        self.is_running = False
        
        # Model bileşenleri
        self.fetcher = None
        self.preprocessor = None
        self.model = None
        self.predictor = None
        self.news_analyzer = None
        self.whale_tracker = None
        
        # Dosya isimleri
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file = f"test_results_{self.coin_symbol}_{timestamp}.json"
        self.log_file = f"test_log_{self.coin_symbol}_{timestamp}.txt"
        
        print(f"🔬 Performance Tester başlatıldı: {self.coin_symbol}")
        print(f"📊 Test süresi: {test_duration_hours} saat")
        print(f"⏱️ Tahmin aralığı: {prediction_interval_minutes} dakika")
    
    def setup_model(self, use_news=True, use_whale=True, newsapi_key=None, whale_api_key=None):
        """
        Model ve analiz bileşenlerini hazırlar
        
        Args:
            use_news (bool): Haber analizi kullanılsın mı
            use_whale (bool): Whale analizi kullanılsın mı
            newsapi_key (str): NewsAPI anahtarı
            whale_api_key (str): Whale Alert API anahtarı
        """
        print("🔧 Model hazırlanıyor...")
        
        try:
            # 1. Veri çekici
            self.fetcher = CryptoDataFetcher()
            
            # 2. İlk veri çekme ve model eğitimi için
            print("📊 İlk veri çekiliyor...")
            initial_data = self.fetcher.fetch_ohlcv_data(self.coin_symbol)
            
            if initial_data is None:
                raise ValueError(f"❌ {self.coin_symbol} için veri çekilemedi!")
            
            # 3. Haber analizi (eğer isteniyorsa)
            sentiment_df = None
            if use_news:
                print("📰 Haber analizi hazırlanıyor...")
                self.news_analyzer = CryptoNewsAnalyzer(newsapi_key)
                
                all_news = self.news_analyzer.fetch_all_news(self.coin_symbol, days=30)
                if all_news:
                    news_sentiment_df = self.news_analyzer.analyze_news_sentiment_batch(all_news)
                    if not news_sentiment_df.empty:
                        sentiment_df = self.news_analyzer.create_daily_sentiment_features(news_sentiment_df, initial_data)
                        print(f"✅ {len(all_news)} haber analiz edildi")
            
            # 4. Whale analizi (eğer isteniyorsa)
            whale_features = None
            if use_whale:
                print("🐋 Whale analizi hazırlanıyor...")
                self.whale_tracker = CryptoWhaleTracker(whale_api_key)
                
                whale_transactions = self.whale_tracker.fetch_whale_alert_transactions(self.coin_symbol, hours=48)
                if whale_transactions:
                    whale_analysis = self.whale_tracker.analyze_whale_transactions(whale_transactions)
                    whale_features = self.whale_tracker.create_whale_features(whale_analysis, 48)
                    print(f"✅ {len(whale_transactions)} whale transfer analiz edildi")
            
            # 5. Veri ön işleme
            print("⚙️ Veri ön işleme...")
            self.preprocessor = CryptoDataPreprocessor()
            processed_df = self.preprocessor.prepare_data(initial_data, True, sentiment_df, whale_features)
            
            # 6. Model eğitimi
            print("🧠 LSTM model eğitiliyor...")
            scaled_data = self.preprocessor.scale_data(processed_df)
            X, y = self.preprocessor.create_sequences(scaled_data, 60)
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(X, y)
            
            # Model oluştur ve eğit
            self.model = CryptoLSTMModel(60, X_train.shape[2])
            self.model.build_model([64, 64, 32], 0.3, 0.001)
            self.model.train_model(X_train, y_train, X_val, y_val, batch_size=32)  # Epochs environment'tan
            
            # 7. Predictor oluştur
            self.predictor = CryptoPricePredictor(self.model, self.preprocessor, 
                                               self.news_analyzer, self.whale_tracker)
            
            print("✅ Model hazırlığı tamamlandı!")
            
            # İlk değerlendirme
            metrics, _ = self.model.evaluate_model(X_test, y_test)
            print(f"📈 Model performansı - MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
            
        except Exception as e:
            print(f"❌ Model hazırlık hatası: {str(e)}")
            raise
    
    def log_message(self, message):
        """
        Mesajı hem ekrana hem dosyaya yazar
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        
        # Dosyaya yaz
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def make_prediction(self):
        """
        Mevcut verilerle tahmin yapar
        
        Returns:
            dict: Tahmin sonucu
        """
        try:
            # Güncel veri çek
            current_data = self.fetcher.fetch_ohlcv_data(self.coin_symbol)
            
            if current_data is None:
                return None
            
            # Haber analizi güncelle (eğer varsa)
            sentiment_df = None
            if self.news_analyzer:
                recent_news = self.news_analyzer.fetch_all_news(self.coin_symbol, days=7)
                if recent_news:
                    news_sentiment_df = self.news_analyzer.analyze_news_sentiment_batch(recent_news)
                    if not news_sentiment_df.empty:
                        sentiment_df = self.news_analyzer.create_daily_sentiment_features(news_sentiment_df, current_data)
            
            # Whale analizi güncelle (eğer varsa)
            whale_features = None
            if self.whale_tracker:
                whale_transactions = self.whale_tracker.fetch_whale_alert_transactions(self.coin_symbol, hours=24)
                if whale_transactions:
                    whale_analysis = self.whale_tracker.analyze_whale_transactions(whale_transactions)
                    whale_features = self.whale_tracker.create_whale_features(whale_analysis, 24)
            
            # Veriyi hazırla
            processed_df = self.preprocessor.prepare_data(current_data, True, sentiment_df, whale_features)
            
            # Tahmin yap
            prediction_result = self.predictor.predict_next_price(processed_df, 60)
            
            if prediction_result:
                return {
                    'timestamp': datetime.now(),
                    'current_price': prediction_result['current_price'],
                    'predicted_price': prediction_result['predicted_price'],
                    'price_change_percent': prediction_result['price_change_percent'],
                    'confidence': prediction_result['confidence'],
                    'prediction_target_time': datetime.now() + timedelta(hours=4),  # 4 saatlik tahmin
                    'features_used': processed_df.shape[1]
                }
            
            return None
            
        except Exception as e:
            self.log_message(f"❌ Tahmin hatası: {str(e)}")
            return None
    
    def get_current_price(self):
        """
        Mevcut fiyatı çeker
        
        Returns:
            float: Mevcut fiyat
        """
        try:
            current_data = self.fetcher.fetch_ohlcv_data(self.coin_symbol, days=1)
            if current_data is not None and not current_data.empty:
                return float(current_data['close'].iloc[-1])
            return None
        except Exception as e:
            self.log_message(f"❌ Fiyat çekme hatası: {str(e)}")
            return None
    
    def calculate_prediction_accuracy(self, prediction, actual_price):
        """
        Tahmin doğruluğunu hesaplar
        
        Args:
            prediction (dict): Tahmin verisi
            actual_price (float): Gerçek fiyat
        
        Returns:
            dict: Doğruluk metrikleri
        """
        predicted_price = prediction['predicted_price']
        current_price = prediction['current_price']
        
        # Mutlak hata
        absolute_error = abs(predicted_price - actual_price)
        
        # Yüzde hata
        percentage_error = (absolute_error / actual_price) * 100
        
        # Yön doğruluğu
        predicted_direction = 1 if predicted_price > current_price else -1
        actual_direction = 1 if actual_price > current_price else -1
        direction_correct = predicted_direction == actual_direction
        
        # Price change accuracy
        predicted_change = ((predicted_price - current_price) / current_price) * 100
        actual_change = ((actual_price - current_price) / current_price) * 100
        change_error = abs(predicted_change - actual_change)
        
        return {
            'absolute_error': absolute_error,
            'percentage_error': percentage_error,
            'direction_correct': direction_correct,
            'predicted_change': predicted_change,
            'actual_change': actual_change,
            'change_error': change_error
        }
    
    def run_test(self):
        """
        24 saatlik test sürecini başlatır
        """
        self.log_message(f"🚀 {self.coin_symbol} için 24 saatlik test başlıyor...")
        self.log_message(f"⏱️ Tahmin aralığı: {self.prediction_interval_minutes} dakika")
        
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=self.test_duration_hours)
        
        prediction_count = 0
        
        try:
            while self.is_running and datetime.now() < end_time:
                # Tahmin yap
                self.log_message(f"🔮 Tahmin #{prediction_count + 1} yapılıyor...")
                prediction = self.make_prediction()
                
                if prediction:
                    # Tahminni kaydet
                    test_entry = {
                        'prediction_id': prediction_count + 1,
                        'prediction_time': prediction['timestamp'].isoformat(),
                        'current_price': prediction['current_price'],
                        'predicted_price': prediction['predicted_price'],
                        'price_change_percent': prediction['price_change_percent'],
                        'confidence': prediction['confidence'],
                        'target_time': prediction['prediction_target_time'].isoformat(),
                        'features_used': prediction['features_used'],
                        'actual_price': None,  # Sonra doldurulacak
                        'accuracy_metrics': None  # Sonra doldurulacak
                    }
                    
                    self.test_results.append(test_entry)
                    prediction_count += 1
                    
                    self.log_message(f"✅ Tahmin kaydedildi:")
                    self.log_message(f"   Mevcut: ${prediction['current_price']:.6f}")
                    self.log_message(f"   Tahmin: ${prediction['predicted_price']:.6f}")
                    self.log_message(f"   Değişim: {prediction['price_change_percent']:+.2f}%")
                    self.log_message(f"   Güven: {prediction['confidence']:.1f}%")
                    
                    # Sonuçları dosyaya kaydet
                    self.save_results()
                
                else:
                    self.log_message("⚠️ Tahmin yapılamadı, tekrar deneniyor...")
                
                # Bekleme
                self.log_message(f"⏳ {self.prediction_interval_minutes} dakika bekleniyor...")
                time.sleep(self.prediction_interval_seconds)
            
            self.log_message("✅ Test süresi tamamlandı!")
            
        except KeyboardInterrupt:
            self.log_message("⚠️ Test kullanıcı tarafından durduruldu!")
            self.is_running = False
        
        except Exception as e:
            self.log_message(f"❌ Test hatası: {str(e)}")
            self.is_running = False
        
        finally:
            # Final analiz
            self.log_message("📊 Final analiz başlıyor...")
            self.update_accuracy_metrics()
            self.generate_final_report()
    
    def update_accuracy_metrics(self):
        """
        Geçmiş tahminler için doğruluk metriklerini günceller
        """
        self.log_message("🔍 Geçmiş tahminlerin doğruluğu kontrol ediliyor...")
        
        current_time = datetime.now()
        
        for i, result in enumerate(self.test_results):
            # Tahmin zamanı geçmişse ve henüz gerçek fiyat alınmamışsa
            target_time = datetime.fromisoformat(result['target_time'])
            
            if current_time >= target_time and result['actual_price'] is None:
                # Gerçek fiyatı al
                actual_price = self.get_current_price()
                
                if actual_price:
                    # Doğruluk metriklerini hesapla
                    accuracy = self.calculate_prediction_accuracy(result, actual_price)
                    
                    # Sonucu güncelle
                    self.test_results[i]['actual_price'] = actual_price
                    self.test_results[i]['accuracy_metrics'] = accuracy
                    
                    self.log_message(f"✅ Tahmin #{result['prediction_id']} doğruluğu:")
                    self.log_message(f"   Tahmin: ${result['predicted_price']:.6f}")
                    self.log_message(f"   Gerçek: ${actual_price:.6f}")
                    self.log_message(f"   Hata: %{accuracy['percentage_error']:.2f}")
                    self.log_message(f"   Yön: {'✅' if accuracy['direction_correct'] else '❌'}")
        
        # Sonuçları kaydet
        self.save_results()
    
    def calculate_overall_performance(self):
        """
        Genel performans metriklerini hesaplar
        
        Returns:
            dict: Performans metrikleri
        """
        completed_predictions = [r for r in self.test_results if r['actual_price'] is not None]
        
        if not completed_predictions:
            return {
                'total_predictions': len(self.test_results),
                'completed_predictions': 0,
                'average_error': 0,
                'direction_accuracy': 0,
                'best_prediction': None,
                'worst_prediction': None
            }
        
        # Metrikler
        errors = [r['accuracy_metrics']['percentage_error'] for r in completed_predictions]
        direction_corrects = [r['accuracy_metrics']['direction_correct'] for r in completed_predictions]
        
        # En iyi ve en kötü tahmin
        best_idx = errors.index(min(errors))
        worst_idx = errors.index(max(errors))
        
        return {
            'total_predictions': len(self.test_results),
            'completed_predictions': len(completed_predictions),
            'average_error': np.mean(errors),
            'median_error': np.median(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'direction_accuracy': (sum(direction_corrects) / len(direction_corrects)) * 100,
            'best_prediction': {
                'id': completed_predictions[best_idx]['prediction_id'],
                'error': errors[best_idx],
                'predicted': completed_predictions[best_idx]['predicted_price'],
                'actual': completed_predictions[best_idx]['actual_price']
            },
            'worst_prediction': {
                'id': completed_predictions[worst_idx]['prediction_id'],
                'error': errors[worst_idx],
                'predicted': completed_predictions[worst_idx]['predicted_price'],
                'actual': completed_predictions[worst_idx]['actual_price']
            }
        }
    
    def generate_final_report(self):
        """
        Final test raporunu oluşturur
        """
        performance = self.calculate_overall_performance()
        
        # Konsol raporu
        report = f"""
{'='*70}
📊 {self.coin_symbol} - 24 SAATLİK PERFORMANS TEST RAPORU
{'='*70}

🔢 GENEL İSTATİSTİKLER:
• Toplam Tahmin Sayısı: {performance['total_predictions']}
• Tamamlanan Tahmin: {performance['completed_predictions']}
• Test Tamamlanma Oranı: {(performance['completed_predictions']/performance['total_predictions']*100):.1f}%

📈 DOĞRULUK METRİKLERİ:
• Ortalama Hata: %{performance['average_error']:.2f}
• Medyan Hata: %{performance['median_error']:.2f}
• En Düşük Hata: %{performance['min_error']:.2f}
• En Yüksek Hata: %{performance['max_error']:.2f}
• Yön Doğruluğu: %{performance['direction_accuracy']:.1f}

🏆 EN İYİ TAHMİN:
• Tahmin #{performance['best_prediction']['id']}
• Hata: %{performance['best_prediction']['error']:.2f}
• Tahmin: ${performance['best_prediction']['predicted']:.6f}
• Gerçek: ${performance['best_prediction']['actual']:.6f}

😞 EN KÖTÜ TAHMİN:
• Tahmin #{performance['worst_prediction']['id']}
• Hata: %{performance['worst_prediction']['error']:.2f}
• Tahmin: ${performance['worst_prediction']['predicted']:.6f}
• Gerçek: ${performance['worst_prediction']['actual']:.6f}

📁 DOSYALAR:
• Test Sonuçları: {self.results_file}
• Test Logları: {self.log_file}
• Grafik: {self.coin_symbol}_performance_chart.png

{'='*70}
"""
        
        print(report)
        self.log_message("📋 Final rapor oluşturuldu!")
        
        # Performans grafiği oluştur
        self.create_performance_chart()
        
        return performance
    
    def create_performance_chart(self):
        """
        Performans grafiği oluşturur
        """
        try:
            completed_predictions = [r for r in self.test_results if r['actual_price'] is not None]
            
            if not completed_predictions:
                self.log_message("⚠️ Grafik için yeterli veri yok")
                return
            
            # Veri hazırlama
            prediction_ids = [r['prediction_id'] for r in completed_predictions]
            predicted_prices = [r['predicted_price'] for r in completed_predictions]
            actual_prices = [r['actual_price'] for r in completed_predictions]
            errors = [r['accuracy_metrics']['percentage_error'] for r in completed_predictions]
            
            # Grafik oluştur
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{self.coin_symbol} - 24 Saatlik Performance Test Sonuçları', fontsize=16)
            
            # 1. Tahmin vs Gerçek Fiyatlar
            ax1.plot(prediction_ids, predicted_prices, 'b-o', label='Tahmin Edilen', linewidth=2)
            ax1.plot(prediction_ids, actual_prices, 'r-s', label='Gerçek Fiyat', linewidth=2)
            ax1.set_xlabel('Tahmin Numarası')
            ax1.set_ylabel('Fiyat ($)')
            ax1.set_title('Tahmin vs Gerçek Fiyatlar')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Hata Yüzdeleri
            ax2.bar(prediction_ids, errors, color='orange', alpha=0.7)
            ax2.set_xlabel('Tahmin Numarası')
            ax2.set_ylabel('Hata (%)')
            ax2.set_title('Tahmin Hata Yüzdeleri')
            ax2.axhline(y=np.mean(errors), color='red', linestyle='--', label=f'Ortalama: {np.mean(errors):.2f}%')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Hata Dağılımı
            ax3.hist(errors, bins=10, color='green', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Hata (%)')
            ax3.set_ylabel('Frekans')
            ax3.set_title('Hata Dağılımı')
            ax3.axvline(x=np.mean(errors), color='red', linestyle='--', label=f'Ortalama: {np.mean(errors):.2f}%')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Yön Doğruluğu
            direction_corrects = [r['accuracy_metrics']['direction_correct'] for r in completed_predictions]
            correct_count = sum(direction_corrects)
            wrong_count = len(direction_corrects) - correct_count
            
            ax4.pie([correct_count, wrong_count], 
                   labels=[f'Doğru ({correct_count})', f'Yanlış ({wrong_count})'],
                   colors=['green', 'red'],
                   autopct='%1.1f%%',
                   startangle=90)
            ax4.set_title('Yön Tahmin Doğruluğu')
            
            plt.tight_layout()
            
            # Grafiği kaydet
            chart_filename = f"{self.coin_symbol}_performance_chart.png"
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            self.log_message(f"📊 Performans grafiği kaydedildi: {chart_filename}")
            
            plt.close()
            
        except Exception as e:
            self.log_message(f"❌ Grafik oluşturma hatası: {str(e)}")
    
    def save_results(self):
        """
        Test sonuçlarını JSON dosyasına kaydeder
        """
        try:
            results_data = {
                'test_info': {
                    'coin_symbol': self.coin_symbol,
                    'test_duration_hours': self.test_duration_hours,
                    'prediction_interval_minutes': self.prediction_interval_minutes,
                    'start_time': datetime.now().isoformat(),
                    'total_predictions': len(self.test_results)
                },
                'predictions': self.test_results,
                'performance_metrics': self.calculate_overall_performance()
            }
            
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.log_message(f"❌ Sonuç kaydetme hatası: {str(e)}")
    
    def stop_test(self):
        """
        Test sürecini durdurur
        """
        self.is_running = False
        self.log_message("🛑 Test durduruldu!")

def main():
    """
    Ana test fonksiyonu
    """
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║            🔬 KRİPTO LSTM PERFORMANS TEST SİSTEMİ 🔬              ║
║                                                                    ║
║  Bu sistem seçtiğiniz coin'i 24 saat boyunca izleyerek           ║
║  tahminlerinizin doğruluğunu gerçek fiyatlarla karşılaştırır.     ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    # Kullanıcı girişi
    print("\n🎯 TEST PARAMETRELERİ")
    print("="*50)
    
    # Coin seçimi
    coin_symbol = input("📊 Test edilecek coin (örn: BTC, ETH): ").strip().upper()
    if not coin_symbol:
        coin_symbol = "BTC"
    
    # Test süresi
    try:
        duration = input("⏱️ Test süresi - saat (varsayılan: 24): ").strip()
        duration = int(duration) if duration else 24
    except:
        duration = 24
    
    # Tahmin aralığı
    try:
        interval = input("🔄 Tahmin aralığı - dakika (varsayılan: 60): ").strip()
        interval = int(interval) if interval else 60
    except:
        interval = 60
    
    # Özellik seçimi
    use_news = input("📰 Haber analizi kullanılsın mı? (y/n, varsayılan: y): ").strip().lower() != 'n'
    use_whale = input("🐋 Whale analizi kullanılsın mı? (y/n, varsayılan: y): ").strip().lower() != 'n'
    
    # API anahtarları
    newsapi_key = None
    whale_api_key = None
    
    if use_news:
        newsapi_key = input("🔑 NewsAPI anahtarı (opsiyonel): ").strip() or None
    
    if use_whale:
        whale_api_key = input("🔑 Whale Alert API anahtarı (opsiyonel): ").strip() or None
    
    # Test başlat
    try:
        tester = CryptoPerformanceTester(coin_symbol, duration, interval)
        
        print(f"\n🔧 {coin_symbol} için model hazırlanıyor...")
        tester.setup_model(use_news, use_whale, newsapi_key, whale_api_key)
        
        print(f"\n🚀 Test başlıyor! ({duration} saat, {interval} dakika aralık)")
        print("💡 Test'i durdurmak için Ctrl+C basın\n")
        
        # Test başlat
        tester.run_test()
        
    except KeyboardInterrupt:
        print("\n🛑 Test kullanıcı tarafından durduruldu!")
        if 'tester' in locals():
            tester.stop_test()
    
    except Exception as e:
        print(f"\n❌ Test hatası: {str(e)}")
    
    print("\n👋 Test tamamlandı!")

if __name__ == "__main__":
    main() 