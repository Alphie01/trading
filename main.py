#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kripto Para LSTM Fiyat Tahmini Uygulaması

Bu uygulama:
1. Kullanıcıdan coin ismini alır
2. Binance'den 100 günlük 4 saatlik mum verilerini çeker
3. Verileri ön işleme tabi tutar ve teknik göstergeler ekler
4. LSTM modeli eğitir
5. Bir sonraki 4 saatlik kapanış fiyatını tahmin eder
6. Analiz raporu ve grafikler oluşturur

Yazar: Kripto Analiz AI
Tarih: 2024
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Kendi modüllerimizi import et
from data_fetcher import CryptoDataFetcher
from data_preprocessor import CryptoDataPreprocessor
from lstm_model import CryptoLSTMModel
from predictor import CryptoPricePredictor
from news_analyzer import CryptoNewsAnalyzer
from whale_tracker import CryptoWhaleTracker

# Model cache sistemi (opsiyonel)
try:
    from model_cache import CachedModelManager
    CACHE_AVAILABLE = True
    print("✅ Model cache sistemi aktif - Eğitim süresi optimize edilecek!")
except ImportError:
    CACHE_AVAILABLE = False
    print("⚠️ Model cache sistemi mevcut değil - Normal eğitim modu")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

def print_banner():
    """
    Uygulama başlangıç banner'ını yazdırır
    """
    banner = """
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║            🚀 KRİPTO PARA LSTM FİYAT TAHMİN SİSTEMİ 🚀            ║
    ║                                                                    ║
    ║  📈 Binance verilerini kullanarak gelişmiş LSTM modeli ile        ║
    ║     kripto para fiyat tahminleri yapan yapay zeka sistemi         ║
    ║                                                                    ║
    ║  ⚡ Özellikler:                                                    ║
    ║     • 100 günlük tarihsel veri analizi                           ║
    ║     • Teknik analiz göstergeleri                                  ║
    ║     • Derin öğrenme LSTM modeli                                   ║
    ║     • Gerçek zamanlı fiyat tahminleri                            ║
    ║     • Detaylı analiz raporları                                    ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def get_user_input():
    """
    Kullanıcıdan coin ismini alır
    
    Returns:
        str: Coin sembolü
    """
    print("\n" + "="*60)
    print("                    COIN SEÇİMİ")
    print("="*60)
    
    while True:
        coin = input("\n🔹 Analiz etmek istediğiniz coin ismini girin (örn: BTC, ETH, ADA): ").strip().upper()
        
        if not coin:
            print("❌ Lütfen geçerli bir coin ismi girin!")
            continue
        
        # Bazı popüler coinleri öner
        popular_coins = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'AVAX', 'MATIC', 'LINK']
        
        if coin in popular_coins:
            print(f"✅ Popüler coin seçildi: {coin}")
        else:
            print(f"⚠️  Dikkat: {coin} popüler listede değil, yine de devam ediliyor...")
        
        return coin

def configure_model_parameters():
    """
    Kullanıcıdan model parametrelerini alır
    
    Returns:
        dict: Model parametreleri
    """
    print("\n" + "="*60)
    print("                MODEL PARAMETRELERİ")
    print("="*60)
    
    print("\n🔧 Model ayarları (Enter = varsayılan değer)")
    
    # Sequence length
    try:
        seq_length = input("📊 Sekans uzunluğu (varsayılan: 60): ").strip()
        seq_length = int(seq_length) if seq_length else 60
    except:
        seq_length = 60
    
    # Epochs (Environment variable desteği ile)
    try:
        epochs_input = input("🔄 Epoch sayısı (Enter = environment'tan al): ").strip()
        if epochs_input:
            epochs = int(epochs_input)
        else:
            epochs = int(os.getenv('LSTM_EPOCHS', 30))  # Environment'tan al
            print(f"   📋 Environment'tan alındı: {epochs} epoch")
    except:
        epochs = int(os.getenv('LSTM_EPOCHS', 30))
    
    # Batch size
    try:
        batch_size = input("📦 Batch boyutu (varsayılan: 32): ").strip()
        batch_size = int(batch_size) if batch_size else 32
    except:
        batch_size = 32
    
    # Teknik göstergeler
    use_technical = input("📈 Teknik göstergeler kullanılsın mı? (y/n, varsayılan: y): ").strip().lower()
    use_technical = use_technical != 'n'
    
    # Haber analizi
    use_news = input("📰 Haber sentiment analizi kullanılsın mı? (y/n, varsayılan: y): ").strip().lower()
    use_news = use_news != 'n'
    
    # NewsAPI anahtarı (opsiyonel)
    newsapi_key = None
    if use_news:
        newsapi_input = input("🔑 NewsAPI anahtarı (opsiyonel, Enter = geç): ").strip()
        if newsapi_input:
            newsapi_key = newsapi_input
    
    # Whale analizi
    use_whale = input("🐋 Whale (büyük cüzdan) analizi kullanılsın mı? (y/n, varsayılan: y): ").strip().lower()
    use_whale = use_whale != 'n'
    
    # Whale Alert API anahtarı (opsiyonel)
    whale_api_key = None
    if use_whale:
        whale_api_input = input("🔑 Whale Alert API anahtarı (opsiyonel, Enter = geç): ").strip()
        if whale_api_input:
            whale_api_key = whale_api_input
    
    params = {
        'sequence_length': seq_length,
        'epochs': epochs,
        'batch_size': batch_size,
        'use_technical_indicators': use_technical,
        'use_news_analysis': use_news,
        'newsapi_key': newsapi_key,
        'use_whale_analysis': use_whale,
        'whale_api_key': whale_api_key
    }
    
    print(f"\n✅ Parametreler ayarlandı:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    return params

def main():
    """
    Ana uygulama fonksiyonu
    """
    try:
        # Banner yazdır
        print_banner()
        
        # Başlangıç mesajı
        print("🎯 Sistem başlatılıyor...\n")
        time.sleep(1)
        
        # 1. Kullanıcı girişi
        coin_symbol = get_user_input()
        model_params = configure_model_parameters()
        
        print(f"\n🚀 {coin_symbol} için analiz başlıyor...\n")
        
        # 2. Veri çekme
        print("="*60)
        print("                  1. VERİ ÇEKME")
        print("="*60)
        
        fetcher = CryptoDataFetcher()
        
        # Symbol doğrulama
        print(f"🔍 {coin_symbol} sembolü doğrulanıyor...")
        if not fetcher.validate_symbol(coin_symbol):
            print(f"❌ {coin_symbol} geçerli bir sembol değil!")
            print("💡 Mevcut semboller kontrol ediliyor...")
            symbols = fetcher.get_available_symbols()
            matching = [s for s in symbols if coin_symbol in s]
            if matching:
                print(f"📋 Benzer semboller: {matching[:10]}")
            return
        
        print(f"✅ {coin_symbol} sembolü doğrulandı!")
        
        # Veri çekme
        df = fetcher.fetch_ohlcv_data(coin_symbol)
        
        if df is None:
            print("❌ Veri çekme başarısız!")
            return
        
        print(f"✅ Veri çekme tamamlandı! {len(df)} veri noktası alındı.")
        
        # 2. Haber analizi (eğer isteniyorsa)
        sentiment_df = None
        news_analyzer = None
        news_analysis = None
        
        if model_params['use_news_analysis']:
            print("\n" + "="*60)
            print("                2. HABER ANALİZİ")
            print("="*60)
            
            try:
                news_analyzer = CryptoNewsAnalyzer(model_params['newsapi_key'])
                print("✅ News analyzer başlatıldı")
                
                # Haberleri çek (detaylı debug ile)
                print("\n📡 Haber kaynakları taranıyor...")
                all_news = news_analyzer.fetch_all_news(coin_symbol, days=7)
                
                if all_news:
                    print(f"\n🧠 {len(all_news)} haberin sentiment analizi yapılıyor...")
                    
                    # Sentiment analizi
                    news_sentiment_df = news_analyzer.analyze_news_sentiment_batch(all_news)
                    
                    if not news_sentiment_df.empty:
                        print("✅ News sentiment analizi tamamlandı")
                        
                        # Günlük sentiment özelliklerini oluştur
                        sentiment_df = news_analyzer.create_daily_sentiment_features(news_sentiment_df, df)
                        
                        # Haber-fiyat korelasyonunu hesapla
                        correlation_results = news_analyzer.calculate_news_price_correlation(sentiment_df, df)
                        
                        # News analysis sonuçlarını hazırla (database için)
                        news_analysis = {
                            'news_sentiment': correlation_results.get('correlation', 0),
                            'news_count': len(all_news),
                            'avg_sentiment': news_sentiment_df['overall_sentiment'].mean() if not news_sentiment_df.empty else 0,
                            'sentiment_confidence': news_sentiment_df['confidence'].mean() if not news_sentiment_df.empty else 0
                        }
                        
                        print(f"📊 Haber Analizi Özeti:")
                        print(f"   📰 Analiz edilen haber: {news_analysis['news_count']}")
                        print(f"   😊 Ortalama sentiment: {news_analysis['avg_sentiment']:+.3f}")
                        print(f"   🎯 Sentiment güveni: {news_analysis['sentiment_confidence']:.1%}")
                        print(f"   📈 Haber-fiyat korelasyonu: {news_analysis['news_sentiment']:+.3f}")
                        
                    else:
                        print("❌ Haber sentiment analizi başarısız")
                        print("💡 Çekilen haberler analiz edilemedi")
                else:
                    print("❌ Hiçbir kaynaktan haber çekilemedi")
                    print("💡 Olası nedenler:")
                    print("   • İnternet bağlantı problemi")
                    print("   • API anahtarları eksik/geçersiz")
                    print("   • Kaynak web siteleri erişilemez")
                    print("   • Coin sembolü için haber bulunamadı")
                    
            except Exception as news_error:
                print(f"❌ Haber analizi hatası: {str(news_error)}")
                print("⚠️ Haber analizi atlanıyor, sadece fiyat verileri kullanılacak")
                news_analysis = None
        else:
            print("\n⏭️ Haber analizi kullanıcı tarafından devre dışı bırakıldı")
        
        # 2.5 Whale analizi (eğer isteniyorsa)
        whale_features = None
        whale_tracker = None
        whale_analysis = None
        
        if model_params['use_whale_analysis']:
            print("\n" + "="*60)
            print("               2.5 WHALE ANALİZİ")
            print("="*60)
            
            whale_tracker = CryptoWhaleTracker(model_params['whale_api_key'])
            
            # Whale transferlerini çek (son 48 saat)
            whale_transactions = whale_tracker.fetch_whale_alert_transactions(coin_symbol, hours=48)
            
            if whale_transactions:
                # Whale transferlerini analiz et
                whale_analysis = whale_tracker.analyze_whale_transactions(whale_transactions)
                
                # Whale özelliklerini oluştur
                whale_features = whale_tracker.create_whale_features(whale_analysis, 48)
                
                # Whale-fiyat korelasyonunu hesapla
                correlation_analysis = whale_tracker.analyze_whale_price_correlation(whale_analysis, df, coin_symbol)
                
                print(f"✅ Whale analizi tamamlandı! Aktivite skoru: {whale_analysis['whale_activity_score']:.1f}/100")
                print(f"🐋 Toplam whale hacmi: ${whale_analysis['total_volume']:,.0f}")
                print(f"📊 İşlem sayısı: {whale_analysis['transaction_count']}")
            else:
                print("⚠️ Whale verisi çekilemedi, varsayılan değerler kullanılacak")
        
        # 3. Veri ön işleme
        print("\n" + "="*60)
        print("                3. VERİ ÖN İŞLEME")
        print("="*60)
        
        preprocessor = CryptoDataPreprocessor()
        
        # Veriyi hazırla (sentiment ve whale verileri ile birlikte)
        processed_df = preprocessor.prepare_data(df, model_params['use_technical_indicators'], sentiment_df, whale_features)
        
        # Veri analizini görselleştir
        print("📊 Veri analizi grafikleri oluşturuluyor...")
        preprocessor.plot_data_analysis(processed_df)
        
        # Veriyi ölçeklendir
        scaled_data = preprocessor.scale_data(processed_df)
        
        # Sekansları oluştur
        X, y = preprocessor.create_sequences(scaled_data, model_params['sequence_length'])
        
        # Veriyi böl
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        print("✅ Veri ön işleme tamamlandı!")
        
        # 4. Model eğitimi (Cache Sistemi ile)
        print("\n" + "="*60)
        print("                 4. MODEL EĞİTİMİ")
        print("="*60)
        
        # Model konfigürasyonu
        n_features = X_train.shape[2]
        model_config = {
            'sequence_length': model_params['sequence_length'],
            'lstm_units': [50, 50, 50],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': model_params['epochs'],
            'batch_size': model_params['batch_size'],
            'use_technical_indicators': model_params['use_technical_indicators'],
            'use_news_analysis': model_params['use_news_analysis'],
            'use_whale_analysis': model_params['use_whale_analysis']
        }
        
        # Cache kullanılabilirse cache'den yükle, yoksa eğit
        if CACHE_AVAILABLE:
            print("🔄 Model cache sistemi kullanılıyor...")
            cache_manager = CachedModelManager()
            
            # Cache'den model al veya eğit
            model, preprocessor_cached, training_info = cache_manager.get_or_train_model(
                coin_symbol, processed_df, model_config, preprocessor
            )
            
            print(f"✅ Model hazır! Tip: {training_info.get('training_type', 'unknown')}")
            
            if training_info.get('training_type') == 'incremental':
                print(f"🔄 Incremental training yapıldı - Zaman tasarrufu sağlandı!")
            elif training_info.get('training_type') == 'new':
                print(f"🆕 Yeni model eğitildi ve cache'lendi")
            else:
                print(f"📂 Cache'den mevcut model yüklendi")
            
            # Cache istatistikleri
            cache_info = cache_manager.get_cache_info()
            print(f"📊 Cache durumu: {cache_info['valid_models']} geçerli model")
            
        else:
            print("🏋️ Geleneksel model eğitimi başlıyor...")
            
            # Model oluştur
            model = CryptoLSTMModel(model_params['sequence_length'], n_features)
            
            # Model mimarisini oluştur
            model.build_model(
                lstm_units=[50, 50, 50],
                dropout_rate=0.2,
                learning_rate=0.001
            )
            
            # Modeli eğit
            history = model.train_model(
                X_train, y_train, X_val, y_val,
                epochs=model_params['epochs'],
                batch_size=model_params['batch_size']
            )
            
            # Eğitim geçmişini görselleştir
            print("📈 Eğitim grafikları oluşturuluyor...")
            model.plot_training_history()
        
        print("✅ Model eğitimi/yükleme tamamlandı!")
        
        # 5. Model değerlendirme
        print("\n" + "="*60)
        print("              5. MODEL DEĞERLENDİRME")
        print("="*60)
        
        # Test verisinde değerlendirme
        metrics, test_predictions = model.evaluate_model(X_test, y_test)
        
        # Tahminleri görselleştir
        model.plot_predictions(y_test, test_predictions.flatten())
        
        print("✅ Model değerlendirme tamamlandı!")
        
        # 6. Fiyat tahmini
        print("\n" + "="*60)
        print("                6. FİYAT TAHMİNİ")
        print("="*60)
        
        # Predictor oluştur (haber analizi ve whale tracker ile birlikte)
        predictor = CryptoPricePredictor(model, preprocessor, news_analyzer, whale_tracker)
        
        # Bir sonraki fiyatı tahmin et
        print("🔮 Bir sonraki kapanış fiyatı tahmin ediliyor...")
        prediction_result = predictor.predict_next_price(processed_df, model_params['sequence_length'])
        
        if prediction_result is None:
            print("❌ Tahmin başarısız!")
            return
        
        # Çoklu dönem tahmini
        print("📅 24 saatlik tahmin yapılıyor...")
        multiple_predictions = predictor.predict_multiple_periods(processed_df, periods=6)
        
        # Haber tabanlı analiz (eğer haber analizi aktifse)
        if news_analysis:
            print("📰 Son günlerin haber analizi yapılıyor...")
            news_analysis = predictor.analyze_recent_news_impact(coin_symbol, days=7)
        
        # Whale analizi (eğer whale tracker aktifse)
        whale_impact = None
        whale_strategy = None
        if whale_tracker:
            print("🐋 Güncel whale analizi yapılıyor...")
            whale_impact = predictor.analyze_whale_impact(coin_symbol, hours=24)
            
            if whale_impact.get('has_whale_data', False):
                whale_strategy = predictor.get_whale_strategy_recommendation(whale_impact, processed_df, coin_symbol)
                print(f"✅ Whale analizi tamamlandı! Strateji: {whale_strategy['strategy']}")
        
        # Yigit ATR Trailing Stop analizi
        print("📊 Yigit ATR Trailing Stop analizi yapılıyor...")
        yigit_analysis = predictor.analyze_yigit_signals(processed_df)
        
        # Tahmin görselleştirmesi
        predictor.plot_prediction_analysis(processed_df, prediction_result)
        
        print("✅ Fiyat tahmini tamamlandı!")
        
        # 7. Rapor oluşturma
        print("\n" + "="*60)
        print("                 7. RAPOR OLUŞTURMA")
        print("="*60)
        
        # Kapsamlı rapor oluştur (haber analizi, whale analizi ve Yigit analizi dahil)
        report = predictor.generate_report(coin_symbol, prediction_result, multiple_predictions, news_analysis, yigit_analysis)
        
        # Whale raporu ekle (eğer varsa)
        if whale_impact and whale_impact.get('has_whale_data', False):
            whale_summary = whale_tracker.generate_whale_summary(
                coin_symbol, whale_impact['whale_analysis'], 
                {'analysis': 'Whale-fiyat korelasyonu analiz edildi'}, whale_strategy
            )
            report += "\n" + whale_summary
        
        # Raporu ekrana yazdır
        print(report)
        
        # Raporu dosyaya kaydet
        save_report = input("\n💾 Raporu dosyaya kaydetmek ister misiniz? (y/n): ").strip().lower()
        if save_report == 'y':
            predictor.save_prediction_to_file(report)
        
            # Cache bilgisi
        if CACHE_AVAILABLE:
            print("\n💡 Model Cache Sistemi Bilgisi:")
            cache_info = cache_manager.get_cache_info()
            print(f"   📦 Toplam cache'li model: {cache_info['valid_models']}")
            print(f"   💾 Cache boyutu: {cache_info['cache_stats']['total_size_mb']} MB")
            print(f"   🚀 Sonraki çalıştırmalarda bu model otomatik yüklenecek!")
            
            # Cache temizlik seçeneği
            if cache_info['expired_models'] > 0:
                cleanup = input(f"🧹 {cache_info['expired_models']} eski model temizlensin mi? (y/n): ").strip().lower()
                if cleanup == 'y':
                    cleanup_result = cache_manager.cleanup_cache()
                    print(f"✅ {cleanup_result['deleted_models']} model temizlendi")
        
        # Manuel model kaydetme (opsiyonel)
        save_model = input("🗂️  Modeli ayrıca manuel kaydetmek ister misiniz? (y/n): ").strip().lower()
        if save_model == 'y':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"crypto_lstm_model_{coin_symbol}_{timestamp}.h5"
            model.save_model(model_filename)
        
        print("\n" + "="*60)
        print("                   🎉 TAMAMLANDI! 🎉")
        print("="*60)
        print(f"""
✨ {coin_symbol} için analiz başarıyla tamamlandı!

📊 İşlem Özeti:
   • Veri noktası: {len(df)}
   • Model tipi: LSTM
   • Tahmin güvenilirlik: {prediction_result['confidence']:.1f}%
   • Sonraki fiyat tahmini: ${prediction_result['predicted_price']:.6f}
   • Beklenen değişim: {prediction_result['price_change_percent']:+.2f}%

⚠️  Risk Uyarısı: Bu tahminler yatırım tavsiyesi değildir!
""")
        
    except KeyboardInterrupt:
        print("\n\n❌ İşlem kullanıcı tarafından iptal edildi.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {str(e)}")
        print("💡 Lütfen internet bağlantınızı kontrol edin ve tekrar deneyin.")
        sys.exit(1)

def quick_demo():
    """
    Hızlı demo fonksiyonu (BTC ile)
    """
    print("🚀 Hızlı Demo Modu - BTC Analizi")
    print("="*50)
    
    try:
        # BTC için otomatik analiz
        fetcher = CryptoDataFetcher()
        df = fetcher.fetch_ohlcv_data('BTC')
        
        if df is None:
            print("❌ Demo verisi çekilemedi!")
            return
        
        preprocessor = CryptoDataPreprocessor()
        processed_df = preprocessor.prepare_data(df, True)
        scaled_data = preprocessor.scale_data(processed_df)
        X, y = preprocessor.create_sequences(scaled_data, 30)  # Kısa sekans
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        # Cache sistemi varsa kullan
        if CACHE_AVAILABLE:
            cache_manager = CachedModelManager()
            demo_config = {
                'sequence_length': 30,
                'lstm_units': [32, 32],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'epochs': 20,
                'batch_size': 16
            }
            model, preprocessor, _ = cache_manager.get_or_train_model(
                'BTC', processed_df, demo_config, preprocessor
            )
        else:
            model = CryptoLSTMModel(30, X_train.shape[2])
            model.build_model([32, 32], 0.2, 0.001)
            model.train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=16)
        
        predictor = CryptoPricePredictor(model, preprocessor)
        prediction = predictor.predict_next_price(processed_df, 30)
        
        if prediction:
            print(f"""
✅ Demo Tamamlandı!

BTC Tahmin Sonucu:
• Mevcut Fiyat: ${prediction['current_price']:.2f}
• Tahmini Fiyat: ${prediction['predicted_price']:.2f}
• Değişim: {prediction['price_change_percent']:+.2f}%
• Güvenilirlik: {prediction['confidence']:.1f}%
""")
        
    except Exception as e:
        print(f"❌ Demo hatası: {str(e)}")

if __name__ == "__main__":
    # Komut satırı argümanlarını kontrol et
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        quick_demo()
    else:
        main() 