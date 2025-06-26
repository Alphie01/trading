import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import warnings
warnings.filterwarnings('ignore')

class CryptoWhaleTracker:
    """
    Büyük kripto cüzdanları (whale'ler) ve transferlerini takip eden sınıf
    """
    
    def __init__(self, whale_alert_api_key=None):
        """
        Whale tracker'ı başlatır
        
        Args:
            whale_alert_api_key (str): Whale Alert API anahtarı (opsiyonel)
        """
        self.whale_alert_api_key = whale_alert_api_key
        self.session = requests.Session()
        
        # Bilinen büyük cüzdan adresleri (örnek listesi)
        self.known_whale_addresses = {
            'bitcoin': [
                '1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ',  # Satoshi'nin cüzdanı
                '3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r',  # Büyük holder
                '34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo',  # Silk Road FBI
            ],
            'ethereum': [
                '0x00000000219ab540356cBB839Cbe05303d7705Fa',  # ETH 2.0 Contract
                '0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8',  # Binance Hot
                '0x8315177aB297bA92A06054cE80a67Ed4DBd7ed3a',  # Bitfinex Cold
            ]
        }
        
        # Exchange adresleri
        self.exchange_addresses = {
            'binance': [
                '0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8',
                '0x28C6c06298d514Db089934071355E5743bf21d60'
            ],
            'coinbase': [
                '0x71660c4005BA85c37ccec55d0C4493E66Fe775d3',
                '0x503828976D22510aad0201ac7EC88293211D23Da'
            ],
            'kraken': [
                '0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2',
                '0x267be1C1D684F78cb4F6a176C4911b741E4Ffdc0'
            ]
        }
        
        # Whale transfer eşikleri (USD)
        self.whale_thresholds = {
            'BTC': 1000000,  # 1M USD
            'ETH': 500000,   # 500K USD  
            'ADA': 100000,   # 100K USD
            'SOL': 100000,   # 100K USD
            'MATIC': 50000,  # 50K USD
        }
    
    def fetch_whale_alert_transactions(self, symbol, hours=24):
        """
        Whale Alert API'den büyük transferleri çeker
        
        Args:
            symbol (str): Coin sembolü (BTC, ETH, etc.)
            hours (int): Kaç saatlik veri çekileceği
        
        Returns:
            list: Whale transfer listesi
        """
        if not self.whale_alert_api_key:
            print("⚠️ Whale Alert API anahtarı bulunamadı, demo veriler kullanılıyor")
            return self._generate_demo_whale_data(symbol, hours)
        
        try:
            # Whale Alert API endpoint
            url = "https://api.whale-alert.io/v1/transactions"
            
            # Son X saatlik veri için timestamp
            start_time = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            params = {
                'api_key': self.whale_alert_api_key,
                'start': start_time,
                'currency': symbol.lower(),
                'min_value': self.whale_thresholds.get(symbol.upper(), 100000)
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                transactions = data.get('transactions', [])
                print(f"📋 Whale Alert'ten {len(transactions)} büyük transfer çekildi")
                return transactions
            else:
                print(f"❌ Whale Alert API hatası: {response.status_code}")
                return self._generate_demo_whale_data(symbol, hours)
        
        except Exception as e:
            print(f"❌ Whale Alert hata: {str(e)}")
            return self._generate_demo_whale_data(symbol, hours)
    
    def _generate_demo_whale_data(self, symbol, hours):
        """
        Demo whale verisi oluşturur (API anahtarı yoksa)
        
        Args:
            symbol (str): Coin sembolü
            hours (int): Saat sayısı
        
        Returns:
            list: Demo whale transfer listesi
        """
        demo_transactions = []
        num_transactions = max(1, hours // 4)  # Her 4 saatte bir transfer
        
        for i in range(num_transactions):
            # Random transfer büyüklüğü
            amount_usd = np.random.uniform(
                self.whale_thresholds.get(symbol.upper(), 100000),
                self.whale_thresholds.get(symbol.upper(), 100000) * 10
            )
            
            # Random transfer tipi
            transfer_types = ['exchange_to_wallet', 'wallet_to_exchange', 'whale_to_whale']
            transfer_type = np.random.choice(transfer_types)
            
            transaction = {
                'id': f"demo_{symbol}_{i}",
                'timestamp': datetime.now() - timedelta(hours=i*4),
                'symbol': symbol.upper(),
                'amount_usd': amount_usd,
                'from': 'unknown' if 'wallet_to' in transfer_type else 'exchange',
                'to': 'exchange' if 'to_exchange' in transfer_type else 'unknown',
                'transaction_type': transfer_type,
                'blockchain': 'bitcoin' if symbol.upper() == 'BTC' else 'ethereum'
            }
            
            demo_transactions.append(transaction)
        
        print(f"📊 {len(demo_transactions)} demo whale transfer oluşturuldu")
        return demo_transactions
    
    def analyze_whale_transactions(self, transactions):
        """
        Whale transferlerini analiz eder
        
        Args:
            transactions (list): Whale transfer listesi
        
        Returns:
            dict: Whale analiz sonuçları
        """
        if not transactions:
            return {
                'total_volume': 0,
                'exchange_inflow': 0,
                'exchange_outflow': 0,
                'whale_activity_score': 0,
                'net_flow': 0,
                'transaction_count': 0,
                'avg_transaction_size': 0,
                'analysis': 'Whale aktivitesi bulunamadı'
            }
        
        # DataFrame'e çevir
        df = pd.DataFrame(transactions)
        
        # Temel istatistikler
        total_volume = df['amount_usd'].sum() if 'amount_usd' in df.columns else 0
        transaction_count = len(df)
        avg_transaction_size = total_volume / transaction_count if transaction_count > 0 else 0
        
        # Exchange flow analizi
        exchange_inflow = 0
        exchange_outflow = 0
        
        for _, tx in df.iterrows():
            tx_amount = tx.get('amount_usd', 0)
            from_addr = str(tx.get('from', '')).lower()
            to_addr = str(tx.get('to', '')).lower()
            
            # Exchange'e giriş (satış baskısı)
            if 'exchange' in to_addr or any(exchange in to_addr for exchange in self.exchange_addresses.keys()):
                exchange_inflow += tx_amount
            
            # Exchange'den çıkış (HODLing)
            elif 'exchange' in from_addr or any(exchange in from_addr for exchange in self.exchange_addresses.keys()):
                exchange_outflow += tx_amount
        
        # Net flow (pozitif = çıkış/HODL, negatif = giriş/satış)
        net_flow = exchange_outflow - exchange_inflow
        
        # Whale aktivite skoru hesaplama
        activity_score = self._calculate_whale_activity_score(
            total_volume, transaction_count, net_flow, avg_transaction_size
        )
        
        # Analiz
        analysis = self._generate_whale_analysis(
            transaction_count, total_volume, net_flow, activity_score
        )
        
        return {
            'total_volume': total_volume,
            'exchange_inflow': exchange_inflow,
            'exchange_outflow': exchange_outflow,
            'whale_activity_score': activity_score,
            'net_flow': net_flow,
            'transaction_count': transaction_count,
            'avg_transaction_size': avg_transaction_size,
            'analysis': analysis
        }
    
    def _calculate_whale_activity_score(self, volume, count, net_flow, avg_size):
        """
        Whale aktivite skorunu hesaplar (0-100)
        
        Args:
            volume: Toplam hacim
            count: İşlem sayısı
            net_flow: Net akış
            avg_size: Ortalama işlem büyüklüğü
        
        Returns:
            float: Aktivite skoru
        """
        # Hacim skoru (0-40 puan)
        volume_score = min(40, (volume / 10000000) * 40)  # 10M USD = 40 puan
        
        # İşlem sayısı skoru (0-20 puan)
        count_score = min(20, count * 2)  # Her işlem 2 puan
        
        # Net flow skoru (0-25 puan)
        net_flow_score = min(25, abs(net_flow) / 1000000 * 25)  # 1M USD = 25 puan
        
        # Ortalama işlem büyüklüğü skoru (0-15 puan)
        avg_size_score = min(15, (avg_size / 1000000) * 15)  # 1M USD = 15 puan
        
        total_score = volume_score + count_score + net_flow_score + avg_size_score
        return min(100, total_score)
    
    def _generate_whale_analysis(self, count, volume, net_flow, score):
        """
        Whale analizini metin olarak oluşturur
        
        Returns:
            str: Analiz metni
        """
        if count == 0:
            return "Önemli whale aktivitesi tespit edilmedi"
        
        # Aktivite seviyesi
        if score > 80:
            activity_level = "ÇOK YÜKSEK"
        elif score > 60:
            activity_level = "YÜKSEK"
        elif score > 40:
            activity_level = "ORTA"
        elif score > 20:
            activity_level = "DÜŞÜK"
        else:
            activity_level = "ÇOK DÜŞÜK"
        
        # Flow analizi
        if net_flow > 1000000:
            flow_analysis = "Büyük çıkış akışı - HODLing artışı 📈"
            market_impact = "YÜKSELIŞ BASKısı"
        elif net_flow < -1000000:
            flow_analysis = "Büyük giriş akışı - Satış baskısı 📉"
            market_impact = "DÜŞÜŞ BASKISI"
        else:
            flow_analysis = "Dengeli akış"
            market_impact = "NÖTR ETKİ"
        
        analysis = f"""
Whale Aktivite Seviyesi: {activity_level} ({score:.1f}/100)
İşlem Sayısı: {count}
Toplam Hacim: ${volume:,.0f}
Net Flow: ${net_flow:,.0f}

{flow_analysis}
Piyasa Etkisi: {market_impact}
"""
        
        return analysis
    
    def create_whale_features(self, whale_analysis, timeframe_hours=24):
        """
        Whale analizinden LSTM için özellikler oluşturur
        
        Args:
            whale_analysis (dict): Whale analiz sonuçları
            timeframe_hours (int): Zaman dilimi
        
        Returns:
            dict: Whale özellikleri
        """
        # Normalizasyon için referans değerler
        max_volume = 100000000  # 100M USD
        max_transactions = 50
        max_avg_size = 10000000  # 10M USD
        
        # Normalize edilmiş özellikler
        volume_normalized = min(1.0, whale_analysis['total_volume'] / max_volume)
        count_normalized = min(1.0, whale_analysis['transaction_count'] / max_transactions)
        avg_size_normalized = min(1.0, whale_analysis['avg_transaction_size'] / max_avg_size)
        
        # Net flow normalize et (-1 ile 1 arası)
        max_net_flow = 50000000  # 50M USD
        net_flow_normalized = max(-1.0, min(1.0, whale_analysis['net_flow'] / max_net_flow))
        
        # Aktivite skoru (0-1 arası)
        activity_score_normalized = whale_analysis['whale_activity_score'] / 100
        
        # Exchange flow oranları
        total_flow = whale_analysis['exchange_inflow'] + whale_analysis['exchange_outflow']
        inflow_ratio = whale_analysis['exchange_inflow'] / total_flow if total_flow > 0 else 0
        outflow_ratio = whale_analysis['exchange_outflow'] / total_flow if total_flow > 0 else 0
        
        # Whale sentiment skoru
        whale_sentiment = self._calculate_whale_sentiment(whale_analysis)
        
        return {
            'whale_volume_norm': volume_normalized,
            'whale_count_norm': count_normalized,
            'whale_avg_size_norm': avg_size_normalized,
            'whale_net_flow_norm': net_flow_normalized,
            'whale_activity_score': activity_score_normalized,
            'whale_inflow_ratio': inflow_ratio,
            'whale_outflow_ratio': outflow_ratio,
            'whale_sentiment': whale_sentiment
        }
    
    def _calculate_whale_sentiment(self, whale_analysis):
        """
        Whale hareketlerinden sentiment hesaplar
        
        Args:
            whale_analysis (dict): Whale analiz sonuçları
        
        Returns:
            float: Whale sentiment (-1 ile 1 arası)
        """
        net_flow = whale_analysis['net_flow']
        activity_score = whale_analysis['whale_activity_score']
        
        # Temel sentiment net flow'dan
        base_sentiment = np.tanh(net_flow / 10000000)  # -1 ile 1 arası
        
        # Aktivite seviyesi ile ağırlıklandır
        activity_weight = activity_score / 100
        
        # Final sentiment
        whale_sentiment = base_sentiment * (0.5 + activity_weight * 0.5)
        
        return whale_sentiment
    
    def analyze_whale_price_correlation(self, whale_data, price_data, symbol):
        """
        Whale hareketleri ile fiyat değişimlerinin korelasyonunu analiz eder
        
        Args:
            whale_data (dict): Whale analiz verileri
            price_data (pd.DataFrame): Fiyat verileri
            symbol (str): Coin sembolü
        
        Returns:
            dict: Korelasyon analizi
        """
        try:
            if whale_data['transaction_count'] == 0:
                return {
                    'correlation': 0,
                    'significance': 'low',
                    'analysis': 'Yeterli whale verisi yok'
                }
            
            # Son fiyat değişimi
            recent_price_change = price_data['close'].pct_change().tail(24).mean()
            
            # Whale sentiment
            whale_sentiment = whale_data.get('whale_sentiment', 0)
            
            # Basit korelasyon hesabı
            if abs(whale_sentiment) > 0.3 and abs(recent_price_change) > 0.02:
                if (whale_sentiment > 0 and recent_price_change > 0) or \
                   (whale_sentiment < 0 and recent_price_change < 0):
                    correlation = 0.7  # Pozitif korelasyon
                    significance = 'high'
                    analysis = 'Whale hareketleri fiyat ile uyumlu'
                else:
                    correlation = -0.3  # Negatif korelasyon
                    significance = 'medium'
                    analysis = 'Whale hareketleri fiyat ile ters'
            else:
                correlation = 0.1
                significance = 'low'
                analysis = 'Whale etkisi belirsiz'
            
            return {
                'correlation': correlation,
                'significance': significance,
                'analysis': analysis,
                'whale_sentiment': whale_sentiment,
                'price_change': recent_price_change
            }
            
        except Exception as e:
            return {
                'correlation': 0,
                'significance': 'low',
                'analysis': f'Korelasyon analizi hatası: {str(e)}'
            }
    
    def get_whale_strategy_recommendation(self, whale_analysis, correlation_analysis):
        """
        Whale verilerine göre strateji önerisi
        
        Args:
            whale_analysis (dict): Whale analiz sonuçları
            correlation_analysis (dict): Korelasyon analizi
        
        Returns:
            dict: Strateji önerisi
        """
        activity_score = whale_analysis['whale_activity_score']
        net_flow = whale_analysis['net_flow']
        correlation = correlation_analysis['correlation']
        
        # Strateji belirleme
        if activity_score > 70:
            if net_flow > 2000000:  # Büyük çıkış
                if correlation > 0.5:
                    strategy = "GÜÇLÜ ALIM - Whale'ler HODL yapıyor"
                    confidence = "YÜKSEK"
                else:
                    strategy = "DİKKATLİ ALIM - Whale çıkışı var ama etki belirsiz"
                    confidence = "ORTA"
            elif net_flow < -2000000:  # Büyük giriş
                if correlation > 0.5:
                    strategy = "SATIM HAZIRLIĞI - Whale'ler satış yapıyor"
                    confidence = "YÜKSEK"
                else:
                    strategy = "BEKLE - Whale girişi var ama etki belirsiz"
                    confidence = "DÜŞÜK"
            else:
                strategy = "BEKLE VE İZLE - Yüksek aktivite ama yön belirsiz"
                confidence = "ORTA"
        
        elif activity_score > 40:
            strategy = "NORMAL TAKİP - Orta seviye whale aktivitesi"
            confidence = "ORTA"
        
        else:
            strategy = "TEKNIK ANALİZ ODAKLI - Düşük whale aktivitesi"
            confidence = "DÜŞÜK"
        
        return {
            'strategy': strategy,
            'confidence': confidence,
            'reasoning': f"Aktivite: {activity_score:.1f}/100, Net Flow: ${net_flow:,.0f}, Korelasyon: {correlation:.2f}"
        }
    
    def generate_whale_summary(self, symbol, whale_analysis, correlation_analysis, strategy):
        """
        Whale analizi özeti oluşturur
        
        Returns:
            str: Whale özet raporu
        """
        activity_score = whale_analysis['whale_activity_score']
        volume = whale_analysis['total_volume']
        net_flow = whale_analysis['net_flow']
        tx_count = whale_analysis['transaction_count']
        
        summary = f"""
🐋 WHALE ANALİZİ ÖZETI - {symbol.upper()}

📊 Aktivite Skoru: {activity_score:.1f}/100
💰 Toplam Hacim: ${volume:,.0f}
🔄 İşlem Sayısı: {tx_count}
💸 Net Flow: ${net_flow:,.0f}

📈 Piyasa Etkisi: {correlation_analysis['analysis']}
🎯 Strateji: {strategy['strategy']}
🎲 Güven: {strategy['confidence']}

{whale_analysis['analysis']}
"""
        
        return summary 