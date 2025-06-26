#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News API Test Script

Bu script News API'nin çalışıp çalışmadığını test eder.
"""

import os
from dotenv import load_dotenv
from news_analyzer import CryptoNewsAnalyzer

def test_news_api():
    """News API'yi test eder"""
    print("📰 News API Test Başlıyor...")
    
    # Environment variables yükle
    load_dotenv()
    
    # API key kontrol et
    api_key = os.getenv('NEWSAPI_KEY')
    if not api_key:
        print("❌ NEWSAPI_KEY bulunamadı!")
        print("💡 .env dosyasında NEWSAPI_KEY ayarlayın")
        return False
    
    print(f"✅ API Key bulundu: {api_key[:8]}...")
    
    try:
        # News analyzer oluştur
        analyzer = CryptoNewsAnalyzer(api_key)
        
        # Bitcoin haberleri çek
        print("\n🔍 Bitcoin haberleri çekiliyor...")
        news_data = analyzer.fetch_all_news("bitcoin", days=1)
        
        if news_data and len(news_data) > 0:
            print(f"✅ {len(news_data)} haber bulundu!")
            
            # İlk 3 haberi göster
            print("\n📄 Son Haberler:")
            for i, news in enumerate(news_data[:3]):
                print(f"   {i+1}. {news.get('title', 'Başlık yok')[:50]}...")
            
            # Sentiment analizi test et
            print("\n🧠 Sentiment analizi test ediliyor...")
            sentiment_df = analyzer.analyze_news_sentiment_batch(news_data)
            
            if not sentiment_df.empty:
                avg_sentiment = sentiment_df['overall_sentiment'].mean()
                print(f"✅ Ortalama sentiment: {avg_sentiment:.3f}")
                return True
            else:
                print("❌ Sentiment analizi başarısız!")
                return False
        else:
            print("❌ Hiç haber bulunamadı!")
            return False
            
    except Exception as e:
        print(f"❌ News API hatası: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_news_api()
    if success:
        print("\n🎉 News API başarıyla çalışıyor!")
    else:
        print("\n💥 News API problemi var!") 