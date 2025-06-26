#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hızlı Performans Testi - Kripto LSTM Sistemi

Bu script sisteminizi hızlıca test etmek için kullanılır.
Kısa süreli (1-2 saat) test yapar ve sonuçları gösterir.
"""

import warnings
warnings.filterwarnings('ignore')

from performance_tester import CryptoPerformanceTester
import time
from datetime import datetime

def quick_test_demo():
    """
    Hızlı demo test - 15 dakika aralıklarla 2 saat test
    """
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                🚀 HIZLI PERFORMANS TESTİ 🚀                      ║
║                                                                    ║
║  Bu test sisteminizi hızlıca kontrol etmek için tasarlanmıştır.   ║
║  2 saat boyunca 15 dakika aralıklarla tahmin yapar.               ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    # Sabit parametreler (hızlı test için)
    coin_symbol = input("📊 Test edilecek coin (varsayılan: BTC): ").strip().upper()
    if not coin_symbol:
        coin_symbol = "BTC"
    
    print(f"\n🎯 {coin_symbol} için hızlı test başlıyor!")
    print("⏱️ Test süresi: 2 saat")
    print("🔄 Tahmin aralığı: 15 dakika")
    print("📰 Haber analizi: Aktif (demo modu)")
    print("🐋 Whale analizi: Aktif (demo modu)")
    
    confirm = input("\n✅ Test'i başlatmak istiyor musunuz? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Test iptal edildi.")
        return
    
    try:
        # Performance tester oluştur
        tester = CryptoPerformanceTester(
            coin_symbol=coin_symbol,
            test_duration_hours=2,  # 2 saat
            prediction_interval_minutes=15  # 15 dakika aralık
        )
        
        print(f"\n🔧 {coin_symbol} için model hazırlanıyor...")
        print("⚠️ Bu işlem birkaç dakika sürebilir...")
        
        # Model setup (haber ve whale demo modu)
        tester.setup_model(
            use_news=True,
            use_whale=True,
            newsapi_key=None,  # Demo modu
            whale_api_key=None  # Demo modu
        )
        
        print(f"\n🚀 Hızlı test başlıyor!")
        print("💡 Test'i durdurmak için Ctrl+C basın")
        print(f"📋 Test sonuçları: {tester.results_file}")
        print(f"📝 Test logları: {tester.log_file}\n")
        
        # Test başlat
        tester.run_test()
        
    except KeyboardInterrupt:
        print("\n🛑 Test kullanıcı tarafından durduruldu!")
        if 'tester' in locals():
            tester.stop_test()
    
    except Exception as e:
        print(f"\n❌ Test hatası: {str(e)}")
        print("💡 İnternet bağlantınızı kontrol edin ve tekrar deneyin.")
    
    print("\n👋 Hızlı test tamamlandı!")

def mini_test():
    """
    Mini test - Sadece 3 tahmin yaparak hızlı sonuç alır
    """
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                     ⚡ MİNİ TEST ⚡                                ║
║                                                                    ║
║  Sadece 3 tahmin yaparak sisteminizi hızlıca test eder.           ║
║  Her tahmin arasında 5 dakika bekler.                             ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    coin_symbol = input("📊 Test edilecek coin (varsayılan: BTC): ").strip().upper()
    if not coin_symbol:
        coin_symbol = "BTC"
    
    print(f"\n🎯 {coin_symbol} için mini test başlıyor!")
    print("⏱️ Test süresi: 15 dakika (3 tahmin)")
    print("🔄 Tahmin aralığı: 5 dakika")
    
    try:
        from data_fetcher import CryptoDataFetcher
        from data_preprocessor import CryptoDataPreprocessor
        from lstm_model import CryptoLSTMModel
        from predictor import CryptoPricePredictor
        from news_analyzer import CryptoNewsAnalyzer
        from whale_tracker import CryptoWhaleTracker
        
        print("\n🔧 Model hazırlanıyor...")
        
        # 1. Veri çek
        fetcher = CryptoDataFetcher()
        data = fetcher.fetch_ohlcv_data(coin_symbol)
        
        if data is None:
            print(f"❌ {coin_symbol} verisi çekilemedi!")
            return
        
        # 2. Hızlı haber analizi
        news_analyzer = CryptoNewsAnalyzer()
        all_news = news_analyzer.fetch_all_news(coin_symbol, days=7)
        sentiment_df = None
        
        if all_news:
            news_sentiment_df = news_analyzer.analyze_news_sentiment_batch(all_news)
            if not news_sentiment_df.empty:
                sentiment_df = news_analyzer.create_daily_sentiment_features(news_sentiment_df, data)
                print(f"📰 {len(all_news)} haber analiz edildi")
        
        # 3. Hızlı whale analizi
        whale_tracker = CryptoWhaleTracker()
        whale_transactions = whale_tracker.fetch_whale_alert_transactions(coin_symbol, hours=24)
        whale_features = None
        
        if whale_transactions:
            whale_analysis = whale_tracker.analyze_whale_transactions(whale_transactions)
            whale_features = whale_tracker.create_whale_features(whale_analysis, 24)
            print(f"🐋 {len(whale_transactions)} whale transfer analiz edildi")
        
        # 4. Veri hazırla
        preprocessor = CryptoDataPreprocessor()
        processed_df = preprocessor.prepare_data(data, True, sentiment_df, whale_features)
        
        # 5. Hızlı model eğitimi
        print("🧠 Model eğitiliyor (hızlı mod)...")
        scaled_data = preprocessor.scale_data(processed_df)
        X, y = preprocessor.create_sequences(scaled_data, 30)  # Kısa sekans
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        model = CryptoLSTMModel(30, X_train.shape[2])
        model.build_model([32, 32], 0.2, 0.001)
        model.train_model(X_train, y_train, X_val, y_val, batch_size=16)  # Epochs environment'tan, hızlı eğitim
        
        # 6. Predictor oluştur
        predictor = CryptoPricePredictor(model, preprocessor, news_analyzer, whale_tracker)
        
        print("✅ Model hazırlığı tamamlandı!\n")
        
        # 7. 3 tahmin yap
        predictions = []
        
        for i in range(3):
            print(f"🔮 Tahmin #{i+1} yapılıyor...")
            
            # Güncel veri çek
            current_data = fetcher.fetch_ohlcv_data(coin_symbol)
            if current_data is not None:
                
                # Güncel analiz
                current_sentiment_df = None
                if news_analyzer:
                    recent_news = news_analyzer.fetch_all_news(coin_symbol, days=3)
                    if recent_news:
                        current_news_sentiment = news_analyzer.analyze_news_sentiment_batch(recent_news)
                        if not current_news_sentiment.empty:
                            current_sentiment_df = news_analyzer.create_daily_sentiment_features(current_news_sentiment, current_data)
                
                current_whale_features = None
                if whale_tracker:
                    current_whale_tx = whale_tracker.fetch_whale_alert_transactions(coin_symbol, hours=12)
                    if current_whale_tx:
                        current_whale_analysis = whale_tracker.analyze_whale_transactions(current_whale_tx)
                        current_whale_features = whale_tracker.create_whale_features(current_whale_analysis, 12)
                
                # Tahmin yap
                current_processed = preprocessor.prepare_data(current_data, True, current_sentiment_df, current_whale_features)
                prediction = predictor.predict_next_price(current_processed, 30)
                
                if prediction:
                    predictions.append({
                        'id': i+1,
                        'timestamp': datetime.now(),
                        'current_price': prediction['current_price'],
                        'predicted_price': prediction['predicted_price'],
                        'change_percent': prediction['price_change_percent'],
                        'confidence': prediction['confidence']
                    })
                    
                    print(f"✅ Tahmin #{i+1}:")
                    print(f"   Mevcut: ${prediction['current_price']:.6f}")
                    print(f"   Tahmin: ${prediction['predicted_price']:.6f}")
                    print(f"   Değişim: {prediction['price_change_percent']:+.2f}%")
                    print(f"   Güven: {prediction['confidence']:.1f}%")
                else:
                    print(f"❌ Tahmin #{i+1} başarısız!")
            
            # 5 dakika bekle (son tahminde bekleme)
            if i < 2:
                print("⏳ 5 dakika bekleniyor...\n")
                time.sleep(300)  # 5 dakika
        
        # 8. Sonuçları göster
        if predictions:
            print("\n" + "="*50)
            print("📊 MİNİ TEST SONUÇLARI")
            print("="*50)
            
            for pred in predictions:
                direction = "📈" if pred['change_percent'] > 0 else "📉"
                print(f"""
Tahmin #{pred['id']} - {pred['timestamp'].strftime('%H:%M:%S')}:
  💰 Mevcut: ${pred['current_price']:.6f}
  🎯 Tahmin: ${pred['predicted_price']:.6f}
  {direction} Değişim: {pred['change_percent']:+.2f}%
  🎲 Güven: {pred['confidence']:.1f}%""")
            
            # Ortalama bilgiler
            avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
            avg_change = sum(abs(p['change_percent']) for p in predictions) / len(predictions)
            
            print(f"""
📈 ÖZET:
  • Toplam Tahmin: {len(predictions)}
  • Ortalama Güven: {avg_confidence:.1f}%
  • Ortalama Değişim: ±{avg_change:.2f}%
  • Test Coin: {coin_symbol}
            """)
            
            print("💡 Bu tahminler 4 saatlik zaman dilimi içindir.")
            print("⚠️  Yatırım tavsiyesi değildir!")
        
    except Exception as e:
        print(f"\n❌ Mini test hatası: {str(e)}")
    
    print("\n👋 Mini test tamamlandı!")

def main():
    """
    Ana test seçim menüsü
    """
    while True:
        print("""
🔬 HIZLI TEST SİSTEMİ

1. 🚀 Hızlı Test (2 saat, 15 dakika aralık)
2. ⚡ Mini Test (15 dakika, 3 tahmin)
3. 📊 Tam Test (performance_tester.py kullan)
4. 🚪 Çıkış

Bu testler sisteminizin çalışıp çalışmadığını kontrol etmek içindir.
""")
        
        choice = input("Seçiminiz (1-4): ").strip()
        
        if choice == '1':
            quick_test_demo()
            break
        elif choice == '2':
            mini_test()
            break
        elif choice == '3':
            print("💡 Tam test için şu komutu çalıştırın:")
            print("   python performance_tester.py")
            break
        elif choice == '4':
            print("👋 Görüşmek üzere!")
            break
        else:
            print("❌ Geçersiz seçim! 1-4 arasında bir sayı girin.")
            continue

if __name__ == "__main__":
    main() 