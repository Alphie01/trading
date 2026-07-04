import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Sentiment analysis libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Web scraping
from bs4 import BeautifulSoup

# **NEW: Environment variable support**
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
    DOTENV_AVAILABLE = True
    print("✅ .env file loaded for NewsAPI configuration")
except ImportError:
    DOTENV_AVAILABLE = False

class CryptoNewsAnalyzer:
    """
    Kripto para haberleri çeken ve sentiment analizi yapan sınıf
    """
    
    def __init__(self, newsapi_key=None):
        """
        News analyzer'ı başlatır
        
        Args:
            newsapi_key (str): NewsAPI anahtarı (opsiyonel, .env'den otomatik çekilir)
        """
        # **IMPROVED: Automatic .env file reading for NewsAPI key**
        if newsapi_key is None:
            # Try to get from environment variable (.env file)
            newsapi_key = os.getenv('NEWSAPI_KEY')
            if newsapi_key:
                print(f"✅ NewsAPI key loaded from .env file")
            else:
                print("⚠️ NewsAPI key not found in .env file (NEWSAPI_KEY)")
                print("💡 Add NEWSAPI_KEY=your_api_key to .env file or pass as parameter")
        else:
            print("✅ NewsAPI key provided as parameter")
            
        self.newsapi_key = newsapi_key
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Display API status
        if self.newsapi_key:
            print("🔑 NewsAPI configured and ready")  # güvenlik: key değeri loglanmaz
        else:
            print("⚠️ NewsAPI not configured - will use free sources only")
        
        # FinBERT — LAZY yüklenir (ilk sentiment çağrısında). __init__'te YÜKLENMEZ:
        # ~440MB HuggingFace indirmesi, ağ yavaş/engelliyse startup'ı SONSUZ bloklardı (web 50dk açılmama sebebi).
        # use_finbert: None = henüz denenmedi, True = yüklü, False = kullanılamıyor (VADER'a düş).
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.finbert_pipeline = None
        self.use_finbert = None
        # HF indirmesi asla süresiz asılmasın (saniyeler) — başarısızsa VADER'a hızlı düşer.
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", os.getenv("HF_HUB_DOWNLOAD_TIMEOUT", "15"))
        
        # Kripto-specific keywords
        self.positive_keywords = [
            'bull', 'bullish', 'pump', 'moon', 'rocket', 'adoption', 'breakthrough',
            'partnership', 'upgrade', 'surge', 'rally', 'boom', 'growth', 'positive',
            'milestone', 'success', 'victory', 'gains', 'profit', 'institutional'
        ]
        
        self.negative_keywords = [
            'bear', 'bearish', 'dump', 'crash', 'fall', 'drop', 'decline', 'ban',
            'regulation', 'hack', 'scam', 'fraud', 'fear', 'panic', 'loss', 'risk',
            'concern', 'warning', 'problem', 'issue', 'negative', 'dead', 'bubble'
        ]
        
    def fetch_newsapi_headlines(self, coin_symbol, days=100):
        """
        NewsAPI'den kripto haberleri çeker
        
        Args:
            coin_symbol (str): Coin sembolü (örn: 'BTC')
            days (int): Kaç günlük haber çekileceği
        
        Returns:
            list: Haber listesi
        """
        if not self.newsapi_key:
            print("⚠️ NewsAPI anahtarı bulunamadı, bu kaynak atlanıyor")
            return []
        
        try:
            # Coin ismine göre arama terimleri
            coin_names = {
                'BTC': 'Bitcoin',
                'ETH': 'Ethereum',
                'BNB': 'Binance',
                'ADA': 'Cardano',
                'SOL': 'Solana',
                'XRP': 'Ripple',
                'DOT': 'Polkadot',
                'MATIC': 'Polygon',
                'AVAX': 'Avalanche',
                'LINK': 'Chainlink'
            }
            
            coin_name = coin_names.get(coin_symbol, coin_symbol)
            
            # Son 100 günlük tarih aralığı
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # NewsAPI parametreleri
            params = {
                'q': f'{coin_name} OR {coin_symbol}',
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.newsapi_key
            }
            
            url = 'https://newsapi.org/v2/everything'
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                print(f"📰 NewsAPI'den {len(articles)} haber çekildi")
                return articles
            else:
                print(f"❌ NewsAPI hatası: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ NewsAPI hata: {str(e)}")
            return []
    
    def fetch_coindesk_news(self, coin_symbol, days=100):
        """
        CoinDesk'ten kripto haberleri çeker (web scraping)
        
        Args:
            coin_symbol (str): Coin sembolü
            days (int): Kaç günlük haber
        
        Returns:
            list: Haber listesi
        """
        try:
            news_list = []
            
            # CoinDesk Bitcoin sayfası
            url = "https://www.coindesk.com/tag/bitcoin/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Haber başlıklarını bul
                articles = soup.find_all('div', class_='articleTextSection')
                
                for article in articles[:20]:  # İlk 20 haberi al
                    try:
                        title_elem = article.find('h4') or article.find('h3')
                        if title_elem:
                            title = title_elem.get_text().strip()
                            
                            news_list.append({
                                'title': title,
                                'description': title,  # Özet yok, başlığı kullan
                                'publishedAt': datetime.now().replace(tzinfo=None).isoformat(),
                                'source': {'name': 'CoinDesk'},
                                'url': 'https://www.coindesk.com'
                            })
                    except:
                        continue
                        
                print(f"📰 CoinDesk'ten {len(news_list)} haber çekildi")
                
            return news_list
            
        except Exception as e:
            print(f"❌ CoinDesk scraping hatası: {str(e)}")
            return []
    
    def fetch_reddit_crypto_posts(self, coin_symbol, days=30):
        """
        Reddit crypto topluluğundan gönderileri çeker
        
        Args:
            coin_symbol (str): Coin sembolü
            days (int): Kaç günlük veri
        
        Returns:
            list: Post listesi
        """
        try:
            news_list = []
            
            # Reddit JSON API (public)
            subreddits = ['cryptocurrency', 'bitcoin', 'ethereum']
            
            for subreddit in subreddits:
                try:
                    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
                    headers = {'User-Agent': 'CryptoAnalyzer/1.0'}
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        posts = data.get('data', {}).get('children', [])
                        
                        for post in posts:
                            post_data = post.get('data', {})
                            title = post_data.get('title', '')
                            selftext = post_data.get('selftext', '')
                            
                            # Coin ile ilgili postları filtrele
                            if coin_symbol.lower() in title.lower() or coin_symbol.lower() in selftext.lower():
                                news_list.append({
                                    'title': title,
                                    'description': selftext[:200] if selftext else title,
                                    'publishedAt': datetime.fromtimestamp(post_data.get('created_utc', 0)).replace(tzinfo=None).isoformat(),
                                    'source': {'name': f'Reddit r/{subreddit}'},
                                    'score': post_data.get('score', 0)
                                })
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    print(f"⚠️ Reddit {subreddit} hatası: {str(e)}")
                    continue
            
            print(f"📰 Reddit'ten {len(news_list)} post çekildi")
            return news_list
            
        except Exception as e:
            print(f"❌ Reddit API hatası: {str(e)}")
            return []
    
    def analyze_sentiment_vader(self, text):
        """
        VADER ile sentiment analizi
        
        Args:
            text (str): Analiz edilecek metin
        
        Returns:
            dict: Sentiment skorları
        """
        scores = self.vader_analyzer.polarity_scores(text)
        return {
            'vader_compound': scores['compound'],
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu']
        }
    
    def analyze_sentiment_textblob(self, text):
        """
        TextBlob ile sentiment analizi
        
        Args:
            text (str): Analiz edilecek metin
        
        Returns:
            dict: Sentiment skorları
        """
        blob = TextBlob(text)
        return {
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        }
    
    def _ensure_finbert(self):
        """FinBERT'i ilk ihtiyaçta BİR KEZ yükler (lazy). Başarısız/erişilemezse use_finbert=False → VADER'a düşülür.
        __init__'i bloklamadığı için web startup asılmaz; ilk haber analizi bir kez model yükler (worker'a taşınana dek)."""
        if self.use_finbert is not None:
            return self.use_finbert
        try:
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_pipeline = pipeline("sentiment-analysis",
                                             model=self.finbert_model,
                                             tokenizer=self.finbert_tokenizer)
            self.use_finbert = True
            print("✅ FinBERT modeli yüklendi (lazy, finansal sentiment analizi)")
        except Exception as e:
            print(f"⚠️ FinBERT yüklenemedi, VADER'a düşülüyor: {str(e)}")
            self.use_finbert = False
        return self.use_finbert

    def analyze_sentiment_finbert(self, text):
        """
        FinBERT ile finansal sentiment analizi

        Args:
            text (str): Analiz edilecek metin

        Returns:
            dict: FinBERT sentiment skorları
        """
        if not self._ensure_finbert():
            return {'finbert_label': 'neutral', 'finbert_score': 0.0}
        
        try:
            # Metni kısalt (FinBERT'in token limiti var)
            text = text[:512]
            
            result = self.finbert_pipeline(text)[0]
            
            # Label'ları normalize et
            label_map = {'positive': 1, 'negative': -1, 'neutral': 0}
            
            return {
                'finbert_label': result['label'].lower(),
                'finbert_score': result['score'],
                'finbert_sentiment': label_map.get(result['label'].lower(), 0)
            }
            
        except Exception as e:
            print(f"⚠️ FinBERT analiz hatası: {str(e)}")
            return {'finbert_label': 'neutral', 'finbert_score': 0.0, 'finbert_sentiment': 0}
    
    def analyze_crypto_keywords(self, text):
        """
        Kripto-specific keyword analizi
        
        Args:
            text (str): Analiz edilecek metin
        
        Returns:
            dict: Keyword sentiment skorları
        """
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        total_keywords = positive_count + negative_count
        
        if total_keywords > 0:
            keyword_sentiment = (positive_count - negative_count) / total_keywords
        else:
            keyword_sentiment = 0
        
        return {
            'keyword_positive_count': positive_count,
            'keyword_negative_count': negative_count,
            'keyword_sentiment': keyword_sentiment
        }
    
    def comprehensive_sentiment_analysis(self, text):
        """
        Kapsamlı sentiment analizi (tüm yöntemleri birleştirir)
        
        Args:
            text (str): Analiz edilecek metin
        
        Returns:
            dict: Tüm sentiment skorları
        """
        if not text or len(text.strip()) < 5:
            return {
                'overall_sentiment': 0,
                'confidence': 0,
                'vader_compound': 0,
                'textblob_polarity': 0,
                'finbert_sentiment': 0,
                'keyword_sentiment': 0
            }
        
        # Farklı yöntemlerle analiz
        vader_results = self.analyze_sentiment_vader(text)
        textblob_results = self.analyze_sentiment_textblob(text)
        finbert_results = self.analyze_sentiment_finbert(text)
        keyword_results = self.analyze_crypto_keywords(text)
        
        # Sonuçları birleştir
        results = {**vader_results, **textblob_results, **finbert_results, **keyword_results}
        
        # Genel sentiment skoru hesapla (ağırlıklı ortalama)
        sentiments = []
        weights = []
        
        if results['vader_compound'] != 0:
            sentiments.append(results['vader_compound'])
            weights.append(0.3)
        
        if results['textblob_polarity'] != 0:
            sentiments.append(results['textblob_polarity'])
            weights.append(0.2)
        
        if results.get('finbert_sentiment', 0) != 0:
            sentiments.append(results['finbert_sentiment'])
            weights.append(0.4)
        
        if results['keyword_sentiment'] != 0:
            sentiments.append(results['keyword_sentiment'])
            weights.append(0.1)
        
        if sentiments and weights:
            overall_sentiment = np.average(sentiments, weights=weights)
            confidence = min(1.0, len(sentiments) / 4.0)  # 4 yöntem varsa %100 güven
        else:
            overall_sentiment = 0
            confidence = 0
        
        results['overall_sentiment'] = overall_sentiment
        results['confidence'] = confidence
        
        return results
    
    def fetch_all_news(self, coin_symbol, days=None):
        """
        Tüm kaynaklardan haberleri çeker
        
        Args:
            coin_symbol (str): Coin sembolü
            days (int): Kaç günlük haber
        
        Returns:
            list: Tüm haberler
        """
        # Days parametresi verilmemişse environment'tan al
        if days is None:
            days = int(os.getenv('NEWS_ANALYSIS_DAYS', 7))  # Varsayılan: 7 gün
            
        print(f"🔍 {coin_symbol} için son {days} günün haberleri çekiliyor...")
        
        all_news = []
        sources_attempted = []
        sources_successful = []
        
        # NewsAPI
        sources_attempted.append("NewsAPI")
        if self.newsapi_key:
            print("   📡 NewsAPI'den haber çekiliyor...")
            try:
                newsapi_news = self.fetch_newsapi_headlines(coin_symbol, days)
                all_news.extend(newsapi_news)
                if newsapi_news:
                    sources_successful.append("NewsAPI")
                    print(f"   ✅ NewsAPI: {len(newsapi_news)} haber")
                else:
                    print("   ⚠️ NewsAPI: Haber bulunamadı")
            except Exception as e:
                print(f"   ❌ NewsAPI hatası: {str(e)}")
        else:
            print("   ⚠️ NewsAPI anahtarı bulunamadı, bu kaynak atlanıyor")
        
        # CoinDesk
        sources_attempted.append("CoinDesk")
        print("   📡 CoinDesk'ten haber çekiliyor...")
        try:
            coindesk_news = self.fetch_coindesk_news(coin_symbol, days)
            all_news.extend(coindesk_news)
            if coindesk_news:
                sources_successful.append("CoinDesk")
                print(f"   ✅ CoinDesk: {len(coindesk_news)} haber")
            else:
                print("   ⚠️ CoinDesk: Haber bulunamadı")
        except Exception as e:
            print(f"   ❌ CoinDesk hatası: {str(e)}")
        
        # Reddit
        sources_attempted.append("Reddit")
        print("   📡 Reddit'ten haber çekiliyor...")
        try:
            reddit_news = self.fetch_reddit_crypto_posts(coin_symbol, min(days, 30))
            all_news.extend(reddit_news)
            if reddit_news:
                sources_successful.append("Reddit")
                print(f"   ✅ Reddit: {len(reddit_news)} post")
            else:
                print("   ⚠️ Reddit: Post bulunamadı")
        except Exception as e:
            print(f"   ❌ Reddit hatası: {str(e)}")
        
        # Özet bilgi
        print(f"\n📊 Haber Çekme Özeti:")
        print(f"   🎯 Denenen kaynaklar: {len(sources_attempted)} ({', '.join(sources_attempted)})")
        print(f"   ✅ Başarılı kaynaklar: {len(sources_successful)} ({', '.join(sources_successful)})")
        print(f"   📰 Toplam haber/post: {len(all_news)}")
        
        if len(all_news) == 0:
            print("   ⚠️ Hiçbir kaynaktan haber çekilemedi!")
            # **VERİ DOĞRULUĞU: Mock veri YALNIZ DEMO_MODE=true iken üretilir.**
            # Production'da (DEMO_MODE kapalı) sahte haber üretme → adımı atla + uyar.
            try:
                from api_cache import demo_mode
                _demo = demo_mode()
            except Exception:
                _demo = False

            if _demo:
                print("   🧪 DEMO_MODE aktif — test amaçlı mock haber verileri oluşturuluyor...")
                try:
                    mock_news = self.get_mock_news_data(coin_symbol, days)
                    for _n in mock_news:
                        _n['is_mock'] = True
                        _n['data_source'] = 'demo_mock'
                    all_news.extend(mock_news)
                    print(f"   ✅ Mock veri (DEMO): {len(mock_news)} test haberi — GERÇEK DEĞİL!")
                except Exception as mock_error:
                    print(f"   ❌ Mock veri oluşturma hatası: {mock_error}")
            else:
                print("   ⚠️ NEWSAPI_KEY/kaynaklar yok. News sentiment skipped (DEMO_MODE kapalı, sahte veri üretilmedi).")

        return all_news
    
    def analyze_news_sentiment_batch(self, news_list):
        """
        Haber listesinin toplu sentiment analizi
        
        Args:
            news_list (list): Haber listesi
        
        Returns:
            pd.DataFrame: Sentiment analizi sonuçları
        """
        if not news_list:
            return pd.DataFrame()
        
        print("🧠 Haber sentiment analizi yapılıyor...")
        
        results = []
        
        for i, news in enumerate(news_list):
            try:
                # Metin hazırlama
                title = news.get('title', '')
                description = news.get('description', '')
                full_text = f"{title}. {description}"
                
                # Tarih parse etme (timezone uyumlu)
                try:
                    pub_date_str = news.get('publishedAt', None)
                    if pub_date_str:
                        pub_date = pd.to_datetime(pub_date_str, utc=True).tz_localize(None)
                    else:
                        pub_date = datetime.now()
                except:
                    pub_date = datetime.now()
                
                # Sentiment analizi
                sentiment_results = self.comprehensive_sentiment_analysis(full_text)
                
                # Sonuçları birleştir
                result = {
                    'date': pub_date,
                    'title': title,
                    'description': description,
                    'source': news.get('source', {}).get('name', 'Unknown'),
                    'url': news.get('url', ''),
                    **sentiment_results
                }
                
                results.append(result)
                
                # İlerleme göstergesi
                if (i + 1) % 10 == 0:
                    print(f"   📊 {i + 1}/{len(news_list)} haber işlendi")
                
            except Exception as e:
                print(f"⚠️ Haber {i} analiz hatası: {str(e)}")
                continue
        
        if results:
            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            print(f"✅ {len(df)} haberin sentiment analizi tamamlandı")
            return df
        else:
            print("❌ Hiçbir haber analiz edilemedi")
            return pd.DataFrame()
    
    def create_daily_sentiment_features(self, sentiment_df, price_df):
        """
        Günlük sentiment özelliklerini oluşturur
        
        Args:
            sentiment_df (pd.DataFrame): Sentiment analizi sonuçları
            price_df (pd.DataFrame): Fiyat verileri
        
        Returns:
            pd.DataFrame: Günlük sentiment özellikleri
        """
        if sentiment_df.empty:
            # Boş sentiment verileri için varsayılan değerler
            daily_features = pd.DataFrame({
                'date': price_df.index,
                'news_count': 0,
                'avg_sentiment': 0,
                'sentiment_volatility': 0,
                'positive_news_ratio': 0,
                'negative_news_ratio': 0,
                'news_confidence': 0
            })
            return daily_features.set_index('date')
        
        # Tarih sütununu indeks yap (timezone safe)
        sentiment_df = sentiment_df.copy()
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], utc=True).dt.tz_localize(None).dt.date
        
        # Günlük gruplandırma
        daily_sentiment = sentiment_df.groupby('date').agg({
            'overall_sentiment': ['count', 'mean', 'std'],
            'confidence': 'mean',
            'vader_compound': 'mean',
            'textblob_polarity': 'mean',
            'finbert_sentiment': 'mean',
            'keyword_sentiment': 'mean'
        }).round(4)
        
        # Sütun isimlerini düzelt
        daily_sentiment.columns = [
            'news_count', 'avg_sentiment', 'sentiment_volatility',
            'news_confidence', 'avg_vader', 'avg_textblob', 
            'avg_finbert', 'avg_keyword'
        ]
        
        # NaN değerleri doldur
        daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_volatility'].fillna(0)
        
        # Pozitif/negatif haber oranları (index-safe)
        sentiment_df['is_positive'] = sentiment_df['overall_sentiment'] > 0.1
        sentiment_df['is_negative'] = sentiment_df['overall_sentiment'] < -0.1
        
        positive_ratio = sentiment_df.groupby('date')['is_positive'].mean()
        negative_ratio = sentiment_df.groupby('date')['is_negative'].mean()
        
        # Index uyumlu olacak şekilde merge et
        for date, ratio in positive_ratio.items():
            if date in daily_sentiment.index:
                daily_sentiment.loc[date, 'positive_news_ratio'] = ratio
            
        for date, ratio in negative_ratio.items():
            if date in daily_sentiment.index:
                daily_sentiment.loc[date, 'negative_news_ratio'] = ratio
        
        # Eksik değerleri 0 ile doldur
        daily_sentiment['positive_news_ratio'] = daily_sentiment['positive_news_ratio'].fillna(0)
        daily_sentiment['negative_news_ratio'] = daily_sentiment['negative_news_ratio'].fillna(0)
        
        # Eksik günleri doldur (timezone safe, manual merge)
        try:
            price_dates = pd.to_datetime(price_df.index, utc=True).tz_localize(None).date
            
            # Yeni DataFrame oluştur - tüm price dates ile
            complete_data = pd.DataFrame(index=price_dates)
            
            # Daily sentiment verilerini merge et
            for col in daily_sentiment.columns:
                complete_data[col] = 0  # Varsayılan değer
                
                for date in daily_sentiment.index:
                    if date in price_dates:
                        complete_data.loc[date, col] = daily_sentiment.loc[date, col]
            
            daily_sentiment = complete_data
        except Exception as e:
            print(f"⚠️ Reindex hatası: {str(e)}, basit yöntem kullanılıyor")
            # Fallback: minimum ortak veri kullan
            pass
        
        print(f"📊 {len(daily_sentiment)} günlük sentiment feature oluşturuldu")
        
        return daily_sentiment
    
    def calculate_news_price_correlation(self, sentiment_df, price_df):
        """
        Haber sentiment'i ile fiyat değişimlerinin korelasyonunu hesaplar
        
        Args:
            sentiment_df (pd.DataFrame): Günlük sentiment verileri
            price_df (pd.DataFrame): Fiyat verileri
        
        Returns:
            dict: Korelasyon analizi sonuçları
        """
        try:
            # Index'leri normalize et
            sentiment_dates = pd.to_datetime(sentiment_df.index, utc=True).tz_localize(None)
            price_dates = pd.to_datetime(price_df.index, utc=True).tz_localize(None)
            
            # Ortak tarihleri bul (date level'da)
            sentiment_date_only = sentiment_dates.date
            price_date_only = price_dates.date
            
            common_dates = list(set(sentiment_date_only) & set(price_date_only))
            
            if len(common_dates) < 10:
                print("⚠️ Yeterli ortak tarih bulunamadı")
                return {'correlation': 0, 'significance': 0}
            
            # Ortak tarihler için verileri al (manuel olarak)
            sentiment_values = []
            price_changes = []
            
            for date in common_dates:
                # Sentiment data
                sentiment_mask = sentiment_date_only == date
                if sentiment_mask.any():
                    sent_val = sentiment_df[sentiment_mask]['avg_sentiment'].mean()
                    sentiment_values.append(sent_val)
                    
                    # Price data 
                    price_mask = price_date_only == date
                    if price_mask.any():
                        price_idx = price_mask.argmax()
                        if price_idx > 0:
                            current_price = price_df.iloc[price_idx]['close']
                            prev_price = price_df.iloc[price_idx-1]['close']
                            price_change = (current_price - prev_price) / prev_price
                            price_changes.append(price_change)
                        else:
                            price_changes.append(0)
                    else:
                        price_changes.append(0)
                else:
                    sentiment_values.append(0)
                    price_changes.append(0)
            
            # Korelasyonları hesapla (numpy kullanarak)
            correlations = {}
            
            if len(sentiment_values) > 1 and len(price_changes) > 1:
                sentiment_array = np.array(sentiment_values)
                price_array = np.array(price_changes)
                
                # Ana korelasyon
                if sentiment_array.std() > 0 and price_array.std() > 0:
                    main_correlation = np.corrcoef(sentiment_array, price_array)[0, 1]
                    main_correlation = main_correlation if not np.isnan(main_correlation) else 0
                else:
                    main_correlation = 0
                    
                correlations['avg_sentiment'] = main_correlation
            else:
                main_correlation = 0
                correlations['avg_sentiment'] = 0
            
            print(f"📈 Haber-Fiyat Korelasyonu: {main_correlation:.3f}")
            
            return {
                'correlation': main_correlation,
                'detailed_correlations': correlations,
                'sample_size': len(common_dates)
            }
            
        except Exception as e:
            print(f"❌ Korelasyon hesaplama hatası: {str(e)}")
            return {'correlation': 0, 'significance': 0}
    
    def get_mock_news_data(self, coin_symbol, days=7):
        """
        Haber kaynakları erişilemez olduğunda test için mock news data sağlar
        
        Args:
            coin_symbol (str): Coin sembolü
            days (int): Kaç günlük mock news
        
        Returns:
            list: Mock haber listesi
        """
        from datetime import datetime, timedelta
        import random
        
        # Kripto ile ilgili sample haberler (pozitif, negatif, nötr)
        sample_news_templates = [
            # Pozitif haberler
            {"title": "{coin} Sees Strong Institutional Adoption", "sentiment": "positive"},
            {"title": "{coin} Price Rally Continues as Volume Surges", "sentiment": "positive"}, 
            {"title": "Major Exchange Announces {coin} Futures Trading", "sentiment": "positive"},
            {"title": "{coin} Network Upgrade Shows Promising Results", "sentiment": "positive"},
            {"title": "Crypto Whale Accumulates Large {coin} Position", "sentiment": "positive"},
            
            # Negatif haberler
            {"title": "{coin} Faces Regulatory Pressure in Major Market", "sentiment": "negative"},
            {"title": "Technical Analysis Shows {coin} Bearish Signals", "sentiment": "negative"},
            {"title": "{coin} Price Drops Amid Market Uncertainty", "sentiment": "negative"},
            {"title": "Concerns Rise Over {coin} Network Congestion", "sentiment": "negative"},
            {"title": "Large {coin} Sell-off Triggers Market Volatility", "sentiment": "negative"},
            
            # Nötr haberler
            {"title": "{coin} Trading Volume Remains Stable", "sentiment": "neutral"},
            {"title": "{coin} Market Analysis: Mixed Signals Continue", "sentiment": "neutral"},
            {"title": "{coin} Price Consolidates in Current Range", "sentiment": "neutral"},
            {"title": "Analysts Debate {coin} Future Market Direction", "sentiment": "neutral"},
            {"title": "{coin} Network Statistics Show Steady Growth", "sentiment": "neutral"}
        ]
        
        # Mock news oluştur
        mock_news = []
        base_date = datetime.now()
        
        # Her gün için 2-4 haber oluştur
        for day in range(days):
            news_count = random.randint(2, 4)
            day_date = base_date - timedelta(days=day)
            
            for _ in range(news_count):
                template = random.choice(sample_news_templates)
                
                news_item = {
                    'title': template['title'].format(coin=coin_symbol),
                    'description': f"Mock news article about {coin_symbol} market developments. " + 
                                 template['title'].format(coin=coin_symbol),
                    'publishedAt': (day_date - timedelta(
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59)
                    )).isoformat(),
                    'source': {'name': f'Mock{random.choice(["News", "Crypto", "Finance"])}'},
                    'url': f'https://mocknews.com/{coin_symbol.lower()}-{random.randint(1000, 9999)}',
                    'mock_sentiment': template['sentiment']  # Test için sentiment ipucu
                }
                
                mock_news.append(news_item)
        
        print(f"📰 {len(mock_news)} mock haber oluşturuldu (test amaçlı)")
        return mock_news 