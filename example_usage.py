#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kripto Para LSTM Fiyat Tahmini - Kapsamlı Örnek Kullanım
Whale Tracker, Haber Analizi ve Yigit ATR özellikleri dahil

Bu dosya, sistemin tüm özelliklerinin nasıl kullanılacağına dair örnekler içerir.
"""

import warnings
warnings.filterwarnings('ignore')

from data_fetcher import CryptoDataFetcher
from data_preprocessor import CryptoDataPreprocessor
from lstm_model import CryptoLSTMModel
from predictor import CryptoPricePredictor
from news_analyzer import CryptoNewsAnalyzer
from whale_tracker import CryptoWhaleTracker

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def example_basic_prediction():
    """
    Temel fiyat tahmini örneği (sadece teknik analiz)
    """
    print("🚀 ÖRNEK 1: Temel Fiyat Tahmini")
    print("="*50)
    
    # 1. Veri çekme
    fetcher = CryptoDataFetcher()
    df = fetcher.fetch_ohlcv_data('BTC')
    
    if df is None:
        print("❌ Veri çekme başarısız!")
        return
    
    # 2. Veri ön işleme
    preprocessor = CryptoDataPreprocessor()
    processed_df = preprocessor.prepare_data(df, use_technical_indicators=True)
    scaled_data = preprocessor.scale_data(processed_df)
    X, y = preprocessor.create_sequences(scaled_data, 60)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # 3. Model eğitimi
    model = CryptoLSTMModel(60, X_train.shape[2])
    model.build_model([50, 50], 0.2, 0.001)
    model.train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # 4. Tahmin
    predictor = CryptoPricePredictor(model, preprocessor)
    prediction = predictor.predict_next_price(processed_df, 60)
    
    if prediction:
        print(f"""
✅ Temel Tahmin Tamamlandı!

BTC Sonuçları:
• Mevcut Fiyat: ${prediction['current_price']:.2f}
• Tahmini Fiyat: ${prediction['predicted_price']:.2f}
• Değişim: {prediction['price_change_percent']:+.2f}%
• Güvenilirlik: {prediction['confidence']:.1f}%
""")

def example_news_enhanced_prediction():
    """
    Haber analizi ile geliştirilmiş tahmin örneği
    """
    print("📰 ÖRNEK 2: Haber Analizi ile Geliştirilmiş Tahmin")
    print("="*50)
    
    # 1. Veri çekme
    fetcher = CryptoDataFetcher()
    df = fetcher.fetch_ohlcv_data('ETH')
    
    if df is None:
        print("❌ Veri çekme başarısız!")
        return
    
    # 2. Haber analizi
    news_analyzer = CryptoNewsAnalyzer()  # API anahtarı olmadan demo modu
    all_news = news_analyzer.fetch_all_news('ETH', days=30)
    
    sentiment_df = None
    if all_news:
        news_sentiment_df = news_analyzer.analyze_news_sentiment_batch(all_news)
        if not news_sentiment_df.empty:
            sentiment_df = news_analyzer.create_daily_sentiment_features(news_sentiment_df, df)
            print(f"✅ {len(all_news)} haber analiz edildi")
    
    # 3. Veri ön işleme (sentiment ile)
    preprocessor = CryptoDataPreprocessor()
    processed_df = preprocessor.prepare_data(df, True, sentiment_df)
    scaled_data = preprocessor.scale_data(processed_df)
    X, y = preprocessor.create_sequences(scaled_data, 60)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # 4. Model eğitimi
    model = CryptoLSTMModel(60, X_train.shape[2])
    model.build_model([50, 50, 50], 0.2, 0.001)
    model.train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # 5. Tahmin (haber analizi ile)
    predictor = CryptoPricePredictor(model, preprocessor, news_analyzer)
    prediction = predictor.predict_next_price(processed_df, 60)
    
    # 6. Son haberlerin etkisi
    news_impact = predictor.analyze_recent_news_impact('ETH', days=3)
    
    if prediction:
        print(f"""
✅ Haber Destekli Tahmin Tamamlandı!

ETH Sonuçları:
• Mevcut Fiyat: ${prediction['current_price']:.2f}
• Tahmini Fiyat: ${prediction['predicted_price']:.2f}
• Değişim: {prediction['price_change_percent']:+.2f}%
• Güvenilirlik: {prediction['confidence']:.1f}%

📰 Haber Etkisi:
• Sentiment Skoru: {news_impact.get('sentiment_score', 0):.2f}
• Haber Sayısı: {news_impact.get('news_count', 0)}
• Etki Düzeyi: {news_impact.get('impact_level', 'Bilinmiyor')}
""")

def example_whale_tracking():
    """
    Whale (büyük cüzdan) takibi örneği
    """
    print("🐋 ÖRNEK 3: Whale Tracking Analizi")
    print("="*50)
    
    # 1. Whale tracker oluştur
    whale_tracker = CryptoWhaleTracker()  # API anahtarı olmadan demo modu
    
    # 2. Whale transferlerini çek
    symbol = 'BTC'
    whale_transactions = whale_tracker.fetch_whale_alert_transactions(symbol, hours=48)
    
    if whale_transactions:
        # 3. Whale analizi
        whale_analysis = whale_tracker.analyze_whale_transactions(whale_transactions)
        
        # 4. Whale özelliklerini oluştur
        whale_features = whale_tracker.create_whale_features(whale_analysis, 48)
        
        # 5. Fiyat verilerini çek (korelasyon için)
        fetcher = CryptoDataFetcher()
        df = fetcher.fetch_ohlcv_data(symbol)
        
        if df is not None:
            # 6. Whale-fiyat korelasyonu
            correlation_analysis = whale_tracker.analyze_whale_price_correlation(whale_analysis, df, symbol)
            
            # 7. Strateji önerisi
            strategy = whale_tracker.get_whale_strategy_recommendation(whale_analysis, correlation_analysis)
            
            # 8. Rapor oluştur
            whale_summary = whale_tracker.generate_whale_summary(symbol, whale_analysis, correlation_analysis, strategy)
            
            print(f"""
✅ Whale Analizi Tamamlandı!

{whale_summary}

📊 Detaylı Veriler:
• Whale Özellikleri: {len(whale_features)} feature
• Net Flow: ${whale_analysis['net_flow']:,.0f}
• Exchange Giriş: ${whale_analysis['exchange_inflow']:,.0f}
• Exchange Çıkış: ${whale_analysis['exchange_outflow']:,.0f}
""")

def example_complete_analysis():
    """
    Tüm özelliklerin bir arada kullanıldığı kapsamlı analiz örneği
    """
    print("🌟 ÖRNEK 4: Kapsamlı Hibrit Analiz (LSTM + Haber + Whale + Yigit)")
    print("="*70)
    
    symbol = 'BTC'
    
    # 1. Veri çekme
    print("📊 Veri çekiliyor...")
    fetcher = CryptoDataFetcher()
    df = fetcher.fetch_ohlcv_data(symbol)
    
    if df is None:
        print("❌ Veri çekme başarısız!")
        return
    
    # 2. Haber analizi
    print("📰 Haber analizi yapılıyor...")
    news_analyzer = CryptoNewsAnalyzer()
    all_news = news_analyzer.fetch_all_news(symbol, days=30)
    
    sentiment_df = None
    if all_news:
        news_sentiment_df = news_analyzer.analyze_news_sentiment_batch(all_news)
        if not news_sentiment_df.empty:
            sentiment_df = news_analyzer.create_daily_sentiment_features(news_sentiment_df, df)
    
    # 3. Whale analizi
    print("🐋 Whale analizi yapılıyor...")
    whale_tracker = CryptoWhaleTracker()
    whale_transactions = whale_tracker.fetch_whale_alert_transactions(symbol, hours=48)
    
    whale_features = None
    whale_analysis = None
    if whale_transactions:
        whale_analysis = whale_tracker.analyze_whale_transactions(whale_transactions)
        whale_features = whale_tracker.create_whale_features(whale_analysis, 48)
    
    # 4. Veri ön işleme (tüm özellikler ile)
    print("⚙️ Veri preprocessing...")
    preprocessor = CryptoDataPreprocessor()
    processed_df = preprocessor.prepare_data(df, True, sentiment_df, whale_features)
    
    # 5. Model hazırlığı
    scaled_data = preprocessor.scale_data(processed_df)
    X, y = preprocessor.create_sequences(scaled_data, 60)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # 6. LSTM model eğitimi
    print("🧠 LSTM model eğitiliyor...")
    model = CryptoLSTMModel(60, X_train.shape[2])
    model.build_model([64, 64, 32], 0.3, 0.001)
    model.train_model(X_train, y_train, X_val, y_val, epochs=25, batch_size=32)
    
    # 7. Kapsamlı tahmin
    print("🔮 Kapsamlı tahmin yapılıyor...")
    predictor = CryptoPricePredictor(model, preprocessor, news_analyzer, whale_tracker)
    
    # Ana tahmin
    prediction = predictor.predict_next_price(processed_df, 60)
    
    # Çoklu dönem tahmini
    multiple_predictions = predictor.predict_multiple_periods(processed_df, periods=6)
    
    # Haber etkisi
    news_impact = predictor.analyze_recent_news_impact(symbol, days=7)
    
    # Whale etkisi
    whale_impact = predictor.analyze_whale_impact(symbol, hours=24)
    whale_strategy = None
    if whale_impact.get('has_whale_data', False):
        whale_strategy = predictor.get_whale_strategy_recommendation(whale_impact, processed_df, symbol)
    
    # Yigit ATR analizi
    yigit_analysis = predictor.analyze_yigit_signals(processed_df)
    
    # 8. Kapsamlı rapor
    print("📋 Rapor oluşturuluyor...")
    report = predictor.generate_report(symbol, prediction, multiple_predictions, news_impact, yigit_analysis)
    
    # Whale raporu ekle
    if whale_impact and whale_impact.get('has_whale_data', False):
        correlation_analysis = whale_tracker.analyze_whale_price_correlation(whale_analysis, df, symbol)
        whale_summary = whale_tracker.generate_whale_summary(symbol, whale_analysis, correlation_analysis, whale_strategy)
        report += "\n" + whale_summary
    
    # Görselleştirme
    predictor.plot_prediction_analysis(processed_df, prediction)
    
    print("="*70)
    print("🎉 KAPSAMLI ANALİZ TAMAMLANDI!")
    print("="*70)
    print(report)

def example_whale_features_analysis():
    """
    Whale özelliklerinin detaylı analizi örneği
    """
    print("🔬 ÖRNEK 5: Whale Özelliklerinin Detaylı Analizi")
    print("="*50)
    
    # Whale tracker oluştur
    whale_tracker = CryptoWhaleTracker()
    
    # Farklı coinler için whale analizi
    symbols = ['BTC', 'ETH', 'ADA']
    
    for symbol in symbols:
        print(f"\n📊 {symbol} Whale Analizi:")
        print("-" * 30)
        
        # Whale transferlerini çek
        transactions = whale_tracker.fetch_whale_alert_transactions(symbol, hours=24)
        
        if transactions:
            # Analiz
            analysis = whale_tracker.analyze_whale_transactions(transactions)
            features = whale_tracker.create_whale_features(analysis, 24)
            
            print(f"• İşlem sayısı: {analysis['transaction_count']}")
            print(f"• Toplam hacim: ${analysis['total_volume']:,.0f}")
            print(f"• Aktivite skoru: {analysis['whale_activity_score']:.1f}/100")
            print(f"• Net flow: ${analysis['net_flow']:,.0f}")
            print(f"• Whale sentiment: {features['whale_sentiment']:.3f}")
            
            # Aktivite seviyesi değerlendirmesi
            if analysis['whale_activity_score'] > 70:
                print("🔥 YÜKSEK WHALE AKTİVİTESİ!")
            elif analysis['whale_activity_score'] > 40:
                print("📈 Orta seviye whale aktivitesi")
            else:
                print("😴 Düşük whale aktivitesi")
        else:
            print("⚠️ Whale verisi bulunamadı")

def example_model_comparison():
    """
    Farklı model konfigürasyonlarının karşılaştırılması
    """
    print("⚖️ ÖRNEK 6: Model Konfigürasyonu Karşılaştırması")
    print("="*50)
    
    # Veri hazırla
    fetcher = CryptoDataFetcher()
    df = fetcher.fetch_ohlcv_data('ETH')
    
    if df is None:
        print("❌ Veri çekme başarısız!")
        return
    
    # Farklı konfigürasyonları test et
    configs = [
        {
            'name': 'Sadece Fiyat',
            'use_technical': False,
            'use_news': False,
            'use_whale': False
        },
        {
            'name': 'Fiyat + Teknik',
            'use_technical': True,
            'use_news': False,
            'use_whale': False
        },
        {
            'name': 'Fiyat + Teknik + Haber',
            'use_technical': True,
            'use_news': True,
            'use_whale': False
        },
        {
            'name': 'Tam Hibrit (Tümü)',
            'use_technical': True,
            'use_news': True,
            'use_whale': True
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n🧪 Test ediliyor: {config['name']}")
        
        try:
            # Veri hazırlama
            preprocessor = CryptoDataPreprocessor()
            
            # Özellikleri hazırla
            sentiment_df = None
            whale_features = None
            
            if config['use_news']:
                news_analyzer = CryptoNewsAnalyzer()
                all_news = news_analyzer.fetch_all_news('ETH', days=10)
                if all_news:
                    news_sentiment_df = news_analyzer.analyze_news_sentiment_batch(all_news)
                    if not news_sentiment_df.empty:
                        sentiment_df = news_analyzer.create_daily_sentiment_features(news_sentiment_df, df)
            
            if config['use_whale']:
                whale_tracker = CryptoWhaleTracker()
                whale_transactions = whale_tracker.fetch_whale_alert_transactions('ETH', hours=24)
                if whale_transactions:
                    whale_analysis = whale_tracker.analyze_whale_transactions(whale_transactions)
                    whale_features = whale_tracker.create_whale_features(whale_analysis, 24)
            
            # Veri işleme
            processed_df = preprocessor.prepare_data(df, config['use_technical'], sentiment_df, whale_features)
            scaled_data = preprocessor.scale_data(processed_df)
            X, y = preprocessor.create_sequences(scaled_data, 30)  # Kısa sekans
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
            
            # Model eğitimi (hızlı)
            model = CryptoLSTMModel(30, X_train.shape[2])
            model.build_model([32, 32], 0.2, 0.001)
            model.train_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=16)
            
            # Değerlendirme
            metrics, _ = model.evaluate_model(X_test, y_test)
            
            results.append({
                'config': config['name'],
                'features': X_train.shape[2],
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'directional_accuracy': metrics['directional_accuracy']
            })
            
            print(f"✅ Tamamlandı - Özellik sayısı: {X_train.shape[2]}, MSE: {metrics['mse']:.6f}")
            
        except Exception as e:
            print(f"❌ Hata: {str(e)}")
    
    # Sonuçları karşılaştır
    print("\n📊 KARŞILAŞTIRMA SONUÇLARI:")
    print("="*50)
    for result in results:
        print(f"""
{result['config']}:
• Özellik sayısı: {result['features']}
• MSE: {result['mse']:.6f}
• MAE: {result['mae']:.6f}
• Yön doğruluğu: {result['directional_accuracy']:.2f}%
""")

def main():
    """
    Tüm örnekleri çalıştır
    """
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║        🚀 KRİPTO LSTM TAHMİN SİSTEMİ - ÖRNEK KULLANIM 🚀         ║
║                                                                    ║
║  Bu dosya, sistemin tüm özelliklerinin nasıl kullanılacağına      ║
║  dair kapsamlı örnekler içerir.                                   ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    examples = [
        ("1", "Temel Fiyat Tahmini", example_basic_prediction),
        ("2", "Haber Analizi ile Tahmin", example_news_enhanced_prediction),
        ("3", "Whale Tracking", example_whale_tracking),
        ("4", "Kapsamlı Hibrit Analiz", example_complete_analysis),
        ("5", "Whale Özellik Analizi", example_whale_features_analysis),
        ("6", "Model Karşılaştırması", example_model_comparison),
        ("all", "Tüm Örnekleri Çalıştır", lambda: [func() for _, _, func in examples[:-1]])
    ]
    
    print("\n🎯 Hangi örneği çalıştırmak istiyorsunuz?")
    for num, desc, _ in examples:
        print(f"   {num}. {desc}")
    
    choice = input("\nSeçiminiz (1-6 veya 'all'): ").strip()
    
    for num, desc, func in examples:
        if choice == num:
            print(f"\n🚀 {desc} çalıştırılıyor...\n")
            func()
            return
    
    print("❌ Geçersiz seçim!")

if __name__ == "__main__":
    main() 