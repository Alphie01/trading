import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class CryptoDataPreprocessor:
    """
    Kripto para verilerini LSTM modeli için hazırlayan sınıf
    """
    
    def __init__(self):
        """
        Preprocessor'ı başlatır
        """
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = None
        self.original_data = None
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume']
        self.sentiment_features = []
        self.whale_features = []
        self.use_news_data = False
        self.use_whale_data = False
    
    @property
    def n_features(self):
        """
        Returns the number of features currently configured
        """
        return len(self.feature_columns)
    
    def add_technical_indicators(self, df):
        """
        Teknik analiz göstergelerini ekler
        
        Args:
            df (pd.DataFrame): OHLCV verileri
        
        Returns:
            pd.DataFrame: Teknik göstergeler eklenmiş veriler
        """
        df = df.copy()
        
        # Simple Moving Averages
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_25'] = df['close'].rolling(window=25).mean()
        
        # Exponential Moving Average
        df['ema_12'] = df['close'].ewm(span=12).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Price change percentage
        df['price_change'] = df['close'].pct_change()
        
        # Volume change percentage
        df['volume_change'] = df['volume'].pct_change()
        
        # Yigit ATR Trailing Stop Indicator (Pine Script ported)
        atr_data = self.add_yigit_atr_indicator(df)
        for col, values in atr_data.items():
            df[col] = values
        
        return df
    
    def add_yigit_atr_indicator(self, df, key_value=1.0, atr_period=10):
        """
        Yigit ATR Trailing Stop indikatörünü ekler (Pine Script'ten porte edildi)
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            key_value (float): Hassasiyet değeri (varsayılan: 1.0)
            atr_period (int): ATR periyodu (varsayılan: 10)
        
        Returns:
            dict: Yigit indikatör verileri
        """
        # True Range hesaplama
        df = df.copy()
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # ATR (Average True Range) hesaplama
        atr = true_range.rolling(window=atr_period).mean()
        
        # nLoss hesaplama
        n_loss = key_value * atr
        
        # ATR Trailing Stop hesaplama
        close = df['close']
        xATRTrailingStop = np.zeros(len(df))
        
        for i in range(1, len(df)):
            prev_stop = xATRTrailingStop[i-1] if i > 0 else 0
            current_close = close.iloc[i]
            prev_close = close.iloc[i-1] if i > 0 else current_close
            current_loss = n_loss.iloc[i] if not pd.isna(n_loss.iloc[i]) else 0
            
            # Pine Script iff conditions ported to Python
            if current_close > prev_stop and prev_close > prev_stop:
                xATRTrailingStop[i] = max(prev_stop, current_close - current_loss)
            elif current_close < prev_stop and prev_close < prev_stop:
                xATRTrailingStop[i] = min(prev_stop, current_close + current_loss)
            elif current_close > prev_stop:
                xATRTrailingStop[i] = current_close - current_loss
            else:
                xATRTrailingStop[i] = current_close + current_loss
        
        # Position hesaplama
        pos = np.zeros(len(df))
        for i in range(1, len(df)):
            prev_close = close.iloc[i-1]
            current_close = close.iloc[i]
            prev_stop = xATRTrailingStop[i-1]
            current_stop = xATRTrailingStop[i]
            
            if prev_close < prev_stop and current_close > current_stop:
                pos[i] = 1  # Long position
            elif prev_close > prev_stop and current_close < current_stop:
                pos[i] = -1  # Short position
            else:
                pos[i] = pos[i-1] if i > 0 else 0
        
        # EMA (1 period = close price)
        ema = close  # EMA(1) = close price
        
        # xATRTrailingStop'u aynı index'e sahip Series'e çevir
        atr_stop_series = pd.Series(xATRTrailingStop, index=df.index)
        
        # Crossover detection (index-safe)
        above = self.crossover(ema, atr_stop_series)
        below = self.crossover(atr_stop_series, ema)
        
        # Buy/Sell signals (index-safe)
        buy_signal = (close > atr_stop_series) & above
        sell_signal = (close < atr_stop_series) & below
        
        # Bar color signals (index-safe)
        bar_buy = close > atr_stop_series
        bar_sell = close < atr_stop_series
        
        # Trend strength indicator (index-safe)
        trend_strength = np.abs(close - atr_stop_series) / atr
        trend_strength = trend_strength.fillna(0)
        
        # Volume-Price relationship
        volume_price_ratio = df['volume'] / close
        
        return {
            'yigit_atr_stop': xATRTrailingStop,
            'yigit_position': pos,
            'yigit_buy_signal': buy_signal.astype(int),
            'yigit_sell_signal': sell_signal.astype(int),
            'yigit_bar_buy': bar_buy.astype(int),
            'yigit_bar_sell': bar_sell.astype(int),
            'yigit_trend_strength': trend_strength,
            'yigit_volume_price_ratio': volume_price_ratio,
            'yigit_atr': atr
        }
    
    def crossover(self, series1, series2):
        """
        Crossover detection (series1 crosses above series2) - Index safe
        
        Args:
            series1, series2: Pandas Series
        
        Returns:
            pd.Series: Boolean series indicating crossover points
        """
        try:
            # Her iki series'i de aynı index'e align et
            if hasattr(series1, 'index') and hasattr(series2, 'index'):
                # Ortak index bul
                common_index = series1.index.intersection(series2.index)
                if len(common_index) == 0:
                    # Ortak index yoksa, minimum uzunlukta ortak index oluştur
                    min_len = min(len(series1), len(series2))
                    common_index = range(min_len)
                
                # Değerleri ortak index'e göre align et
                s1_values = series1.reindex(common_index, method='ffill').fillna(0)
                s2_values = series2.reindex(common_index, method='ffill').fillna(0)
                
                # Crossover hesapla
                current_cross = s1_values > s2_values
                prev_cross = s1_values.shift(1).fillna(False) <= s2_values.shift(1).fillna(False)
                result = current_cross & prev_cross
                
                return result
            else:
                # Eğer pandas Series değillerse, numpy array olarak işle
                val1 = np.array(series1) if not isinstance(series1, np.ndarray) else series1
                val2 = np.array(series2) if not isinstance(series2, np.ndarray) else series2
                
                # Uzunlukları eşitle
                min_len = min(len(val1), len(val2))
                val1 = val1[:min_len]
                val2 = val2[:min_len]
                
                current_cross = val1 > val2
                prev_cross = np.concatenate([[False], val1[:-1] <= val2[:-1]])
                result = current_cross & prev_cross
                
                return pd.Series(result, index=range(len(result)))
                
        except Exception as e:
            print(f"⚠️ Crossover hesaplama hatası: {str(e)}, fallback kullanılıyor")
            # Son çare: basit boolean array döndür
            length = min(len(series1), len(series2)) if hasattr(series1, '__len__') and hasattr(series2, '__len__') else 1
            return pd.Series([False] * length, index=range(length))
    
    def add_sentiment_features(self, df, sentiment_df):
        """
        Sentiment özelliklerini fiyat verilerine ekler
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            sentiment_df (pd.DataFrame): Günlük sentiment verileri
        
        Returns:
            pd.DataFrame: Sentiment özellikleri eklenmiş veriler
        """
        if sentiment_df.empty:
            print("⚠️ Sentiment verileri boş, varsayılan değerler kullanılıyor")
            # Varsayılan sentiment değerleri ekle
            df['news_count'] = 0
            df['avg_sentiment'] = 0
            df['sentiment_volatility'] = 0
            df['positive_news_ratio'] = 0
            df['negative_news_ratio'] = 0
            df['news_confidence'] = 0
            return df
        
        print("📊 Sentiment özellikleri fiyat verilerine entegre ediliyor...")
        
        df = df.copy()
        
        # Tarihleri align et (timezone safe)
        df_dates = pd.to_datetime(df.index, utc=True).tz_localize(None).date
        sentiment_dates = pd.to_datetime(sentiment_df.index, utc=True).tz_localize(None).date
        
        # Sentiment verilerini fiyat verilerine merge et
        for col in sentiment_df.columns:
            df[col] = 0  # Varsayılan değer
            
            for i, date in enumerate(df_dates):
                if date in sentiment_dates:
                    sentiment_idx = list(sentiment_dates).index(date)
                    df.iloc[i, df.columns.get_loc(col)] = sentiment_df.iloc[sentiment_idx][col]
        
        # Sentiment özelliklerini feature listesine ekle
        self.sentiment_features = list(sentiment_df.columns)
        self.use_news_data = True
        
        print(f"✅ {len(self.sentiment_features)} sentiment özelliği eklendi")
        
        return df
    
    def add_whale_features(self, df, whale_features_dict):
        """
        Whale özelliklerini fiyat verilerine ekler
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            whale_features_dict (dict): Whale özellikleri
        
        Returns:
            pd.DataFrame: Whale özellikleri eklenmiş veriler
        """
        if not whale_features_dict:
            print("⚠️ Whale verileri boş, varsayılan değerler kullanılıyor")
            # Varsayılan whale değerleri ekle
            whale_defaults = {
                'whale_volume_norm': 0,
                'whale_count_norm': 0,
                'whale_avg_size_norm': 0,
                'whale_net_flow_norm': 0,
                'whale_activity_score': 0,
                'whale_inflow_ratio': 0,
                'whale_outflow_ratio': 0,
                'whale_sentiment': 0
            }
            for feature, value in whale_defaults.items():
                df[feature] = value
            self.whale_features = list(whale_defaults.keys())
            return df
        
        print("🐋 Whale özellikleri fiyat verilerine entegre ediliyor...")
        
        df = df.copy()
        
        # Whale özelliklerini tüm satırlara ekle (whale verileri genellikle günlük/saatlik)
        for feature, value in whale_features_dict.items():
            df[feature] = value
        
        # Whale özelliklerini feature listesine ekle
        self.whale_features = list(whale_features_dict.keys())
        self.use_whale_data = True
        
        print(f"✅ {len(self.whale_features)} whale özelliği eklendi")
        
        return df
    
    def prepare_data(self, df, use_technical_indicators=True, sentiment_df=None, whale_features=None):
        """
        Veriyi LSTM için hazırlar
        
        Args:
            df (pd.DataFrame): Ham OHLCV verileri
            use_technical_indicators (bool): Teknik göstergelerin kullanılıp kullanılmayacağı
            sentiment_df (pd.DataFrame): Sentiment verileri (opsiyonel)
            whale_features (dict): Whale analiz özellikleri (opsiyonel)
        
        Returns:
            pd.DataFrame: Hazırlanmış veriler veya None (yetersiz veri durumunda)
        """
        self.original_data = df.copy()
        
        if use_technical_indicators:
            df = self.add_technical_indicators(df)
            # Teknik göstergeleri feature listesine ekle
            self.feature_columns.extend([
                'sma_7', 'sma_25', 'ema_12', 'rsi', 'macd', 'macd_signal',
                'bb_middle', 'bb_upper', 'bb_lower', 'price_change', 'volume_change',
                'yigit_atr_stop', 'yigit_position', 'yigit_buy_signal', 'yigit_sell_signal',
                'yigit_bar_buy', 'yigit_bar_sell', 'yigit_trend_strength', 
                'yigit_volume_price_ratio', 'yigit_atr'
            ])
        
        # Sentiment özelliklerini ekle (eğer varsa)
        if sentiment_df is not None and not sentiment_df.empty:
            df = self.add_sentiment_features(df, sentiment_df)
            # Sentiment özelliklerini feature listesine ekle
            self.feature_columns.extend(self.sentiment_features)
        
        # Whale özelliklerini ekle (eğer varsa)
        if whale_features is not None:
            df = self.add_whale_features(df, whale_features)
            # Whale özelliklerini feature listesine ekle
            self.feature_columns.extend(self.whale_features)
        
        # NaN değerleri temizle
        df = df.dropna()
        
        if len(df) < 50:
            # ValueError yerine None döndür ve log spam'i engelle
            return None
        
        print(f"Veri hazırlama tamamlandı. Toplam {len(df)} veri noktası.")
        print(f"Toplam özellik sayısı: {len(self.feature_columns)}")
        if self.use_news_data:
            print(f"📰 Haber sentiment özellikleri dahil: {len(self.sentiment_features)} özellik")
        if self.use_whale_data:
            print(f"🐋 Whale analiz özellikleri dahil: {len(self.whale_features)} özellik")
        
        return df
    
    def scale_data(self, df, fit_scaler=True):
        """
        Veriyi normalize eder
        
        Args:
            df (pd.DataFrame): Normalize edilecek veriler
            fit_scaler (bool): Scaler'ı fit etmek için True, sadece transform için False
        
        Returns:
            np.array: Normalize edilmiş veriler
        """
        # Sadece sayısal sütunları al
        numeric_data = df[self.feature_columns].values
        
        # Scaler'ın zaten fit edilip edilmediğini kontrol et
        try:
            if fit_scaler:
                # Yeni eğitim: fit ve transform
                self.scaled_data = self.scaler.fit_transform(numeric_data)
                print("🔧 Scaler fit edildi ve transform uygulandı")
            else:
                # Cache'den yüklenen model: sadece transform
                self.scaled_data = self.scaler.transform(numeric_data)
                print("🔄 Mevcut scaler ile transform uygulandı")
        except Exception as e:
            # Eğer scaler fit edilmemişse (hata durumu), fit et
            print(f"⚠️ Scaler hatası: {str(e)}, yeniden fit ediliyor...")
            self.scaled_data = self.scaler.fit_transform(numeric_data)
            print("🔧 Scaler zorla fit edildi ve transform uygulandı")
        
        print(f"Veri ölçeklendirme tamamlandı. Shape: {self.scaled_data.shape}")
        return self.scaled_data
    
    def create_sequences(self, data, sequence_length=60, target_column_index=3):
        """
        LSTM için zaman serisi sekansları oluşturur
        
        Args:
            data (np.array): Ölçeklendirilmiş veriler
            sequence_length (int): Sekans uzunluğu (varsayılan: 60)
            target_column_index (int): Hedef sütun indeksi (close price için 3)
        
        Returns:
            tuple: (X, y) - Features ve targets
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            # Geçmiş sequence_length kadar veriyi al
            X.append(data[i-sequence_length:i])
            # Bir sonraki kapanış fiyatını hedef olarak al
            y.append(data[i, target_column_index])
        
        X, y = np.array(X), np.array(y)
        
        print(f"Sekanslar oluşturuldu. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def split_data(self, X, y, test_size=0.2, validation_size=0.1):
        """
        Veriyi eğitim, doğrulama ve test setlerine böler
        
        Args:
            X (np.array): Features
            y (np.array): Targets
            test_size (float): Test seti oranı
            validation_size (float): Doğrulama seti oranı
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # İlk olarak train ve test'i ayır
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Train'i train ve validation'a ayır
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, shuffle=False
        )
        
        print(f"Veri bölme tamamlandı:")
        print(f"  Eğitim seti: {X_train.shape[0]} örneklem")
        print(f"  Doğrulama seti: {X_val.shape[0]} örneklem")
        print(f"  Test seti: {X_test.shape[0]} örneklem")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def inverse_transform_prediction(self, prediction):
        """
        Normalize edilmiş tahmini orijinal ölçeğe çevirir
        
        Args:
            prediction (float): Normalize edilmiş tahmin
        
        Returns:
            float: Orijinal ölçekteki tahmin
        """
        # Dummy array oluştur (sadece close price sütunu için)
        dummy = np.zeros((1, len(self.feature_columns)))
        dummy[0, 3] = prediction  # close price indeksi
        
        # Inverse transform uygula
        dummy_inversed = self.scaler.inverse_transform(dummy)
        
        return dummy_inversed[0, 3]
    
    def plot_data_analysis(self, df):
        """
        Veri analizi grafikleri çizer
        
        Args:
            df (pd.DataFrame): Analiz edilecek veriler
        """
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Fiyat grafiği
        axes[0, 0].plot(df.index, df['close'], label='Close Price', color='blue')
        if 'sma_7' in df.columns:
            axes[0, 0].plot(df.index, df['sma_7'], label='SMA 7', alpha=0.7)
            axes[0, 0].plot(df.index, df['sma_25'], label='SMA 25', alpha=0.7)
        axes[0, 0].set_title('Price Movement')
        axes[0, 0].set_ylabel('Price (USDT)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Volume grafiği
        axes[0, 1].bar(df.index, df['volume'], alpha=0.7, color='orange')
        axes[0, 1].set_title('Trading Volume')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True)
        
        # RSI grafiği (eğer varsa)
        if 'rsi' in df.columns:
            axes[1, 0].plot(df.index, df['rsi'], color='purple')
            axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7)
            axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('RSI')
            axes[1, 0].set_ylabel('RSI')
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].grid(True)
        
        # MACD grafiği (eğer varsa)
        if 'macd' in df.columns:
            axes[1, 1].plot(df.index, df['macd'], label='MACD', color='blue')
            axes[1, 1].plot(df.index, df['macd_signal'], label='Signal', color='red')
            axes[1, 1].set_title('MACD')
            axes[1, 1].set_ylabel('MACD')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Yigit ATR Trailing Stop grafiği
        if 'yigit_atr_stop' in df.columns:
            axes[2, 0].plot(df.index, df['close'], label='Close Price', color='blue', linewidth=2)
            axes[2, 0].plot(df.index, df['yigit_atr_stop'], label='ATR Trailing Stop', color='orange', linewidth=1.5)
            
            # Buy/Sell sinyallerini göster
            buy_signals = df[df['yigit_buy_signal'] == 1]
            sell_signals = df[df['yigit_sell_signal'] == 1]
            
            if not buy_signals.empty:
                axes[2, 0].scatter(buy_signals.index, buy_signals['close'], 
                                 color='green', marker='^', s=100, label='Al Sinyali', zorder=5)
            
            if not sell_signals.empty:
                axes[2, 0].scatter(sell_signals.index, sell_signals['close'], 
                                 color='red', marker='v', s=100, label='Sat Sinyali', zorder=5)
            
            axes[2, 0].set_title('Yigit ATR Trailing Stop')
            axes[2, 0].set_ylabel('Fiyat (USDT)')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
        
        # Yigit Trend Strength ve Volume-Price Ratio
        if 'yigit_trend_strength' in df.columns and 'yigit_volume_price_ratio' in df.columns:
            # Trend strength
            ax2_1_twin = axes[2, 1].twinx()
            axes[2, 1].plot(df.index, df['yigit_trend_strength'], color='purple', linewidth=1, label='Trend Gücü')
            axes[2, 1].set_ylabel('Trend Gücü', color='purple')
            axes[2, 1].tick_params(axis='y', labelcolor='purple')
            
            # Volume-Price ratio
            ax2_1_twin.plot(df.index, df['yigit_volume_price_ratio'], color='brown', alpha=0.7, label='Volume/Price Oranı')
            ax2_1_twin.set_ylabel('Volume/Price Oranı', color='brown')
            ax2_1_twin.tick_params(axis='y', labelcolor='brown')
            
            axes[2, 1].set_title('Yigit Trend Gücü & Volume/Price Oranı')
            axes[2, 1].grid(True)
            
            # Position renk kodlaması için bar color
            if 'yigit_position' in df.columns:
                position_colors = ['red' if pos == -1 else 'green' if pos == 1 else 'gray' 
                                 for pos in df['yigit_position']]
                axes[2, 1].scatter(df.index[::10], [0.5] * len(df.index[::10]), 
                                 c=[position_colors[i] for i in range(0, len(position_colors), 10)], 
                                 s=20, alpha=0.6, label='Pozisyon')
        
        plt.tight_layout()
        plt.show()
        
    def get_latest_sequence(self, df, sequence_length=60):
        """
        En son sequence'ı tahmin için hazırlar
        
        Args:
            df (pd.DataFrame): Veriler
            sequence_length (int): Sekans uzunluğu
        
        Returns:
            np.array: Tahmin için hazır sekans
        """
        # Son sequence_length kadar veriyi al
        latest_data = df[self.feature_columns].tail(sequence_length).values
        
        # Ölçeklendir
        latest_scaled = self.scaler.transform(latest_data)
        
        # LSTM için şekillendirme (1, sequence_length, features)
        latest_sequence = latest_scaled.reshape(1, sequence_length, -1)
        
        return latest_sequence 