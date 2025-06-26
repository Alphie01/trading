import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

# **CRITICAL: Use centralized TensorFlow configuration**
from tf_config import get_tensorflow, is_tensorflow_available

tf = get_tensorflow()
TF_AVAILABLE = is_tensorflow_available()

print(f"🧠 LSTM Model - TensorFlow Available: {TF_AVAILABLE}")

# Safe TensorFlow imports with fallbacks
if TF_AVAILABLE and tf:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras import backend as K
    try:
        from tensorflow.keras.optimizers.legacy import Adam
    except ImportError:
        from tensorflow.keras.optimizers import Adam
else:
    # Mock classes for when TensorFlow is not available
    Sequential = None
    LSTM = Dense = Dropout = BatchNormalization = None
    EarlyStopping = ReduceLROnPlateau = ModelCheckpoint = None
    K = None
    Adam = None
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Environment variables yükle
load_dotenv()

def directional_accuracy(y_true, y_pred):
    """
    Fiyat yönü tahmin doğruluğunu hesaplar (custom metric)
    Regression problemleri için accuracy metriği
    """
    # Fiyat değişim yönlerini hesapla
    true_direction = K.sign(y_true[1:] - y_true[:-1])
    pred_direction = K.sign(y_pred[1:] - y_pred[:-1])
    
    # Yönlerin ne kadar uyuştuğunu hesapla
    correct_directions = K.equal(true_direction, pred_direction)
    accuracy = K.mean(K.cast(correct_directions, K.floatx()))
    
    return accuracy

class CryptoLSTMModel:
    """
    Kripto para fiyat tahmini için LSTM modeli sınıfı
    """
    
    def __init__(self, sequence_length=60, n_features=16):
        """
        LSTM modelini başlatır
        
        Args:
            sequence_length (int): Giriş sekans uzunluğu
            n_features (int): Özellik sayısı
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
        # GPU configuration handled by centralized tf_config
        print(f"🧠 LSTM Model initialized (TF Available: {TF_AVAILABLE})")
    
    def build_model(self, lstm_units=[50, 50, 50], dropout_rate=0.2, learning_rate=0.001):
        """
        LSTM modelini oluşturur
        
        Args:
            lstm_units (list): Her LSTM katmanının nöron sayısı
            dropout_rate (float): Dropout oranı
            learning_rate (float): Öğrenme oranı
        """
        self.model = Sequential()
        
        # İlk LSTM katmanı
        self.model.add(LSTM(
            lstm_units[0], 
            return_sequences=True, 
            input_shape=(self.sequence_length, self.n_features)
        ))
        self.model.add(Dropout(dropout_rate))
        self.model.add(BatchNormalization())
        
        # Orta LSTM katmanları
        for units in lstm_units[1:-1]:
            self.model.add(LSTM(units, return_sequences=True))
            self.model.add(Dropout(dropout_rate))
            self.model.add(BatchNormalization())
        
        # Son LSTM katmanı
        self.model.add(LSTM(lstm_units[-1], return_sequences=False))
        self.model.add(Dropout(dropout_rate))
        self.model.add(BatchNormalization())
        
        # Dense katmanlar
        self.model.add(Dense(25, activation='relu'))
        self.model.add(Dropout(dropout_rate/2))
        
        # Çıkış katmanı
        self.model.add(Dense(1, activation='linear'))
        
        # Modeli derle
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape', directional_accuracy]
        )
        
        # Model özetini yazdır
        print("Model Mimarisi:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs=None, batch_size=32, patience=15, verbose=1, 
                   use_early_stopping=True):
        """
        Modeli eğitir
        
        Args:
            X_train (np.array): Eğitim özellikleri
            y_train (np.array): Eğitim hedefleri
            X_val (np.array): Doğrulama özellikleri
            y_val (np.array): Doğrulama hedefleri
            epochs (int): Epoch sayısı (None ise environment'tan alır)
            batch_size (int): Batch boyutu
            patience (int): Early stopping için sabır
            verbose (int): Eğitim çıktı seviyesi (0=sessiz, 1=progress bar, 2=bir satır per epoch)
            use_early_stopping (bool): Early stopping kullanılsın mı (İlk eğitim için False önerilir)
        
        Returns:
            History: Eğitim geçmişi
        """
        # Epochs parametresi verilmemişse environment'tan al
        if epochs is None:
            epochs = int(os.getenv('LSTM_EPOCHS', 1000))  # Varsayılan: 30 epoch
        if self.model is None:
            raise ValueError("Model henüz oluşturulmadı. Önce build_model() çalıştırın.")
        
        # Callbacks listesi
        callbacks = []
        
        # **YENİ: Early stopping isteğe bağlı**
        if use_early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1 if verbose > 0 else 0
            ))
            if verbose > 0:
                print(f"📊 Early stopping aktif (patience={patience})")
        else:
            if verbose > 0:
                print("🔄 Early stopping devre dışı - tüm epoch'lar çalışacak")
        
        # Diğer callbacks (her zaman aktif)
        callbacks.extend([
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(5, patience//3),  # Early stopping'den daha düşük patience
                min_lr=1e-7,
                verbose=1 if verbose > 0 else 0
            ),
            ModelCheckpoint(
                'best_crypto_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1 if verbose > 0 else 0
            )
        ])
        
        if verbose > 0:
            print("Model eğitimi başlıyor...")
            print(f"Eğitim verisi: {X_train.shape}")
            print(f"Doğrulama verisi: {X_val.shape}")
            print(f"Toplam epoch: {epochs}")
        
        # Modeli eğit
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        if verbose > 0:
            print("Model eğitimi tamamlandı!")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Modeli test verisiyle değerlendirir
        
        Args:
            X_test (np.array): Test özellikleri
            y_test (np.array): Test hedefleri
        
        Returns:
            dict: Değerlendirme metrikleri
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi.")
        
        # Test verisiyle değerlendirme
        evaluation_results = self.model.evaluate(X_test, y_test, verbose=0)
        test_loss, test_mae, test_mape, test_directional_accuracy = evaluation_results
        
        # Tahminleri al
        predictions = self.model.predict(X_test)
        
        # Metrikler
        mse = np.mean((y_test - predictions.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - predictions.flatten()))
        mape = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100
        
        # Manuel directional accuracy hesapla (doğrulama için)
        y_true_diff = np.diff(y_test)
        y_pred_diff = np.diff(predictions.flatten())
        true_direction = np.sign(y_true_diff)
        pred_direction = np.sign(y_pred_diff)
        manual_dir_accuracy = np.mean(true_direction == pred_direction)
        
        metrics = {
            'test_loss': test_loss,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': test_directional_accuracy,
            'manual_dir_accuracy': manual_dir_accuracy
        }
        
        print("Model Değerlendirme Sonuçları:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.6f}")
        
        return metrics, predictions
    
    def predict(self, X):
        """
        Tahmin yapar
        
        Args:
            X (np.array): Tahmin için giriş verisi
        
        Returns:
            np.array: Tahminler
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi.")
        
        return self.model.predict(X)
    
    def plot_training_history(self):
        """
        Eğitim geçmişini görselleştirir
        """
        if self.history is None:
            print("Henüz eğitim geçmişi yok.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss grafiği
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE grafiği
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MAPE grafiği
        axes[1, 0].plot(self.history.history['mape'], label='Training MAPE')
        axes[1, 0].plot(self.history.history['val_mape'], label='Validation MAPE')
        axes[1, 0].set_title('Model MAPE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Directional Accuracy grafiği
        if 'directional_accuracy' in self.history.history:
            axes[0, 2].plot(self.history.history['directional_accuracy'], label='Training Dir. Accuracy')
            axes[0, 2].plot(self.history.history['val_directional_accuracy'], label='Validation Dir. Accuracy')
            axes[0, 2].set_title('Directional Accuracy')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        else:
            axes[0, 2].text(0.5, 0.5, 'Directional Accuracy\nhistory not available', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
        
        # Learning rate grafiği (eğer varsa)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nhistory not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Boş alan için placeholder
        axes[1, 2].text(0.5, 0.5, 'Additional Metrics\n(Reserved)', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, title="Tahmin vs Gerçek Değerler"):
        """
        Tahminleri gerçek değerlerle karşılaştırır
        
        Args:
            y_true (np.array): Gerçek değerler
            y_pred (np.array): Tahmin edilen değerler
            title (str): Grafik başlığı
        """
        plt.figure(figsize=(12, 6))
        
        # Zaman serisi olarak göster
        plt.plot(y_true, label='Gerçek Değerler', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Tahminler', color='red', alpha=0.7)
        
        plt.title(title)
        plt.xlabel('Zaman')
        plt.ylabel('Normalize Edilmiş Fiyat')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahmin Edilen Değerler')
        plt.title('Tahmin vs Gerçek (Scatter Plot)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Modeli kaydet
        
        Args:
            filepath (str): Kaydedilecek dosya yolu
        """
        if self.model is None:
            raise ValueError("Kaydedilecek model yok.")
        
        self.model.save(filepath)
        print(f"Model şuraya kaydedildi: {filepath}")
    
    def load_model(self, filepath):
        """
        Modeli yükle
        
        Args:
            filepath (str): Yüklenecek model dosyası
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")
        
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model şuradan yüklendi: {filepath}")
        
        return self.model 