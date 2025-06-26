#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Model Cache Sistemi

Bu modül eğitilmiş LSTM modellerini cache'leyerek:
- Daha önce eğitilmiş modelleri yükler
- Incremental training (artırımlı eğitim) yapar
- Model versiyonlarını takip eder
- Eğitim süresini ve işlem gücünü optimize eder
"""

import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from lstm_model import CryptoLSTMModel
    from data_preprocessor import CryptoDataPreprocessor
except ImportError:
    print("⚠️ LSTM modül ve preprocessor gerekli")

class ModelCache:
    """
    LSTM model cache ve versiyonlama sistemi
    """
    
    def __init__(self, cache_dir: str = "model_cache"):
        """
        Model cache'ini başlatır
        
        Args:
            cache_dir (str): Cache dizini
        """
        self.cache_dir = cache_dir
        self.models_dir = os.path.join(cache_dir, "models")
        self.metadata_dir = os.path.join(cache_dir, "metadata")
        self.preprocessors_dir = os.path.join(cache_dir, "preprocessors")
        
        # Dizinleri oluştur
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.preprocessors_dir, exist_ok=True)
        
        # Cache ayarları
        self.max_model_age_days = 7  # Model maksimum 7 gün geçerli
        self.incremental_training_threshold = 0.85  # %85 accuracy altında yeniden eğit
        self.max_cache_size = 50  # Maksimum cache'lenebilecek model sayısı
        
        print(f"📦 Model Cache başlatıldı: {cache_dir}")
    
    def _generate_model_id(self, symbol: str, config: Dict) -> str:
        """
        Model için benzersiz ID oluşturur
        
        Args:
            symbol (str): Trading çifti
            config (Dict): Model konfigürasyonu
        
        Returns:
            str: Model ID
        """
        # Config'i string'e çevir ve hash oluştur
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{symbol.replace('/', '_')}_{config_hash}"
    
    def _get_model_paths(self, model_id: str) -> Dict[str, str]:
        """
        Model dosya yollarını döndürür
        
        Args:
            model_id (str): Model ID
        
        Returns:
            Dict[str, str]: Dosya yolları
        """
        return {
            'model': os.path.join(self.models_dir, f"{model_id}.h5"),
            'weights': os.path.join(self.models_dir, f"{model_id}_weights.h5"),
            'metadata': os.path.join(self.metadata_dir, f"{model_id}.json"),
            'preprocessor': os.path.join(self.preprocessors_dir, f"{model_id}.pkl"),
            'scaler': os.path.join(self.preprocessors_dir, f"{model_id}_scaler.pkl")
        }
    
    def save_model(self, model: 'CryptoLSTMModel', preprocessor: 'CryptoDataPreprocessor',
                   symbol: str, config: Dict, training_data_info: Dict,
                   performance_metrics: Dict) -> str:
        """
        Modeli cache'e kaydeder
        
        Args:
            model: Eğitilmiş LSTM modeli
            preprocessor: Veri ön işleme sınıfı
            symbol: Trading çifti
            config: Model konfigürasyonu
            training_data_info: Eğitim verisi bilgileri
            performance_metrics: Performans metrikleri
        
        Returns:
            str: Model ID
        """
        try:
            model_id = self._generate_model_id(symbol, config)
            paths = self._get_model_paths(model_id)
            
            # Model'i kaydet
            model.save_model(paths['model'])
            
            # Preprocessor'ı kaydet
            with open(paths['preprocessor'], 'wb') as f:
                pickle.dump(preprocessor, f)
            
            # Scaler'ı ayrı kaydet
            with open(paths['scaler'], 'wb') as f:
                pickle.dump(preprocessor.scaler, f)
            
            # Metadata oluştur
            metadata = {
                'model_id': model_id,
                'symbol': symbol,
                'config': config,
                'training_data_info': training_data_info,
                'performance_metrics': performance_metrics,
                'created_at': datetime.now().isoformat(),
                'last_trained': datetime.now().isoformat(),
                'training_count': 1,
                'feature_count': training_data_info.get('feature_count', 0),
                'data_hash': training_data_info.get('data_hash', ''),
                'version': '1.0'
            }
            
            # Metadata kaydet
            with open(paths['metadata'], 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Model cache'lendi: {model_id}")
            return model_id
            
        except Exception as e:
            print(f"❌ Model cache kaydetme hatası: {str(e)}")
            return None
    
    def load_model(self, model_id: str) -> Optional[Tuple['CryptoLSTMModel', 'CryptoDataPreprocessor', Dict]]:
        """
        Cache'den model yükler
        
        Args:
            model_id (str): Model ID
        
        Returns:
            Optional[Tuple]: (model, preprocessor, metadata) veya None
        """
        try:
            paths = self._get_model_paths(model_id)
            
            # Dosyaların varlığını kontrol et
            if not all(os.path.exists(path) for path in [paths['model'], paths['metadata'], paths['preprocessor']]):
                return None
            
            # Metadata yükle
            with open(paths['metadata'], 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Model yaşını kontrol et
            last_trained = datetime.fromisoformat(metadata['last_trained'])
            if datetime.now() - last_trained > timedelta(days=self.max_model_age_days):
                print(f"⚠️ Model çok eski: {model_id} ({(datetime.now() - last_trained).days} gün)")
                return None
            
            # Preprocessor yükle
            with open(paths['preprocessor'], 'rb') as f:
                preprocessor = pickle.load(f)
            
            # Scaler yükle (varsa)
            if os.path.exists(paths['scaler']):
                with open(paths['scaler'], 'rb') as f:
                    preprocessor.scaler = pickle.load(f)
            
            # Model yükle
            config = metadata['config']
            sequence_length = config.get('sequence_length', 60)
            feature_count = metadata.get('feature_count', 10)
            
            model = CryptoLSTMModel(sequence_length, feature_count)
            model.load_model(paths['model'])
            
            print(f"📂 Model yüklendi: {model_id}")
            return model, preprocessor, metadata
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {str(e)}")
            return None
    
    def find_cached_model(self, symbol: str, config: Dict) -> Optional[str]:
        """
        Belirli symbol ve config için cache'lenmiş model arar
        
        Args:
            symbol (str): Trading çifti
            config (Dict): Model konfigürasyonu
        
        Returns:
            Optional[str]: Model ID veya None
        """
        try:
            model_id = self._generate_model_id(symbol, config)
            paths = self._get_model_paths(model_id)
            
            if os.path.exists(paths['metadata']):
                with open(paths['metadata'], 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Model yaşını kontrol et
                last_trained = datetime.fromisoformat(metadata['last_trained'])
                if datetime.now() - last_trained <= timedelta(days=self.max_model_age_days):
                    return model_id
            
            return None
            
        except Exception as e:
            print(f"❌ Model arama hatası: {str(e)}")
            return None
    
    def update_model(self, model_id: str, model: 'CryptoLSTMModel', 
                     new_performance: Dict, additional_data_info: Dict) -> bool:
        """
        Mevcut modeli incremental training ile günceller
        
        Args:
            model_id (str): Model ID
            model: Güncellenmiş model
            new_performance: Yeni performans metrikleri
            additional_data_info: Ek veri bilgileri
        
        Returns:
            bool: Güncelleme başarısı
        """
        try:
            paths = self._get_model_paths(model_id)
            
            # Mevcut metadata'yı yükle
            with open(paths['metadata'], 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Model'i güncelle
            model.save_model(paths['model'])
            
            # Metadata güncelle
            metadata['last_trained'] = datetime.now().isoformat()
            metadata['training_count'] += 1
            metadata['performance_metrics'] = new_performance
            metadata['version'] = f"{metadata['training_count']}.0"
            
            # Ek veri bilgilerini ekle
            if 'training_history' not in metadata:
                metadata['training_history'] = []
            
            metadata['training_history'].append({
                'timestamp': datetime.now().isoformat(),
                'performance': new_performance,
                'data_info': additional_data_info
            })
            
            # Metadata kaydet
            with open(paths['metadata'], 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"🔄 Model güncellendi: {model_id} (v{metadata['version']})")
            return True
            
        except Exception as e:
            print(f"❌ Model güncelleme hatası: {str(e)}")
            return False
    
    def should_retrain_model(self, metadata: Dict, current_performance: Optional[Dict] = None) -> bool:
        """
        Modelin yeniden eğitilip eğitilmeyeceğini belirler
        
        Args:
            metadata: Model metadata'sı
            current_performance: Mevcut performans (varsa)
        
        Returns:
            bool: Yeniden eğitim gerekli mi?
        """
        try:
            # Yaş kontrolü
            last_trained = datetime.fromisoformat(metadata['last_trained'])
            age_days = (datetime.now() - last_trained).days
            
            if age_days > self.max_model_age_days:
                print(f"🔄 Model çok eski, yeniden eğitim gerekli: {age_days} gün")
                return True
            
            # Performans kontrolü
            if current_performance:
                cached_accuracy = metadata.get('performance_metrics', {}).get('directional_accuracy', 0)
                current_accuracy = current_performance.get('directional_accuracy', 0)
                
                if current_accuracy < self.incremental_training_threshold:
                    print(f"📉 Düşük performans, yeniden eğitim gerekli: {current_accuracy:.1f}% < {self.incremental_training_threshold*100:.1f}%")
                    return True
                
                # Performans düştüyse
                if current_accuracy < cached_accuracy * 0.9:  # %10 düşüş
                    print(f"📉 Performans düştü, yeniden eğitim gerekli: {current_accuracy:.1f}% -> {cached_accuracy:.1f}%")
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Yeniden eğitim kontrolü hatası: {str(e)}")
            return True  # Hata durumunda güvenli tarafta kal
    
    def get_cache_stats(self) -> Dict:
        """
        Cache istatistiklerini döndürür
        
        Returns:
            Dict: Cache istatistikleri
        """
        try:
            stats = {
                'total_models': 0,
                'total_size_mb': 0,
                'models_by_symbol': {},
                'oldest_model': None,
                'newest_model': None,
                'cache_dir': self.cache_dir
            }
            
            model_dates = []
            
            for filename in os.listdir(self.metadata_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.metadata_dir, filename)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    stats['total_models'] += 1
                    
                    symbol = metadata['symbol']
                    if symbol not in stats['models_by_symbol']:
                        stats['models_by_symbol'][symbol] = 0
                    stats['models_by_symbol'][symbol] += 1
                    
                    model_dates.append(datetime.fromisoformat(metadata['last_trained']))
            
            # Tarih istatistikleri
            if model_dates:
                stats['oldest_model'] = min(model_dates).isoformat()
                stats['newest_model'] = max(model_dates).isoformat()
            
            # Boyut hesapla
            total_size = 0
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    total_size += os.path.getsize(filepath)
            
            stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            print(f"❌ Cache istatistik hatası: {str(e)}")
            return {}
    
    def cleanup_old_models(self, max_age_days: Optional[int] = None) -> int:
        """
        Eski modelleri temizler
        
        Args:
            max_age_days: Maksimum yaş (varsayılan: self.max_model_age_days)
        
        Returns:
            int: Temizlenen model sayısı
        """
        if max_age_days is None:
            max_age_days = self.max_model_age_days
        
        deleted_count = 0
        
        try:
            for filename in os.listdir(self.metadata_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.metadata_dir, filename)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    last_trained = datetime.fromisoformat(metadata['last_trained'])
                    age_days = (datetime.now() - last_trained).days
                    
                    if age_days > max_age_days:
                        model_id = metadata['model_id']
                        
                        # İlgili dosyaları sil
                        paths = self._get_model_paths(model_id)
                        for path in paths.values():
                            if os.path.exists(path):
                                os.remove(path)
                        
                        deleted_count += 1
                        print(f"🗑️ Eski model silindi: {model_id} ({age_days} gün)")
            
            print(f"✅ {deleted_count} eski model temizlendi")
            return deleted_count
            
        except Exception as e:
            print(f"❌ Model temizleme hatası: {str(e)}")
            return 0
    
    def list_cached_models(self) -> List[Dict]:
        """
        Cache'deki tüm modelleri listeler
        
        Returns:
            List[Dict]: Model listesi
        """
        models = []
        
        try:
            for filename in os.listdir(self.metadata_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.metadata_dir, filename)
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Yaş hesapla
                    last_trained = datetime.fromisoformat(metadata['last_trained'])
                    age_days = (datetime.now() - last_trained).days
                    
                    model_info = {
                        'model_id': metadata['model_id'],
                        'symbol': metadata['symbol'],
                        'version': metadata.get('version', '1.0'),
                        'last_trained': metadata['last_trained'],
                        'age_days': age_days,
                        'training_count': metadata.get('training_count', 1),
                        'performance': metadata.get('performance_metrics', {}),
                        'feature_count': metadata.get('feature_count', 0),
                        'valid': age_days <= self.max_model_age_days
                    }
                    
                    models.append(model_info)
            
            # Son eğitim tarihine göre sırala
            models.sort(key=lambda x: x['last_trained'], reverse=True)
            
            return models
            
        except Exception as e:
            print(f"❌ Model listeleme hatası: {str(e)}")
            return []

class CachedModelManager:
    """
    Model cache'i kullanarak model eğitimi ve yönetimi yapar
    """
    
    def __init__(self, cache_dir: str = "model_cache"):
        """
        CachedModelManager'ı başlatır
        
        Args:
            cache_dir (str): Cache dizini
        """
        self.cache = ModelCache(cache_dir)
        print("🧠 Cached Model Manager başlatıldı!")
    
    def get_or_train_model(self, symbol: str, data: pd.DataFrame, 
                          config: Dict, preprocessor: 'CryptoDataPreprocessor',
                          force_retrain: bool = False) -> Tuple['CryptoLSTMModel', 'CryptoDataPreprocessor', Dict]:
        """
        Cache'den model yükler veya yeni eğitir
        
        Args:
            symbol (str): Trading çifti
            data (pd.DataFrame): Eğitim verisi
            config (Dict): Model konfigürasyonu
            preprocessor (CryptoDataPreprocessor): Veri ön işleme
            force_retrain (bool): Zorla yeniden eğitim
        
        Returns:
            Tuple: (model, preprocessor, training_info)
        """
        try:
            # Cache'de model ara
            model_id = self.cache.find_cached_model(symbol, config)
            
            if model_id and not force_retrain:
                print(f"📂 Cache'den model yükleniyor: {symbol}")
                
                cached_result = self.cache.load_model(model_id)
                if cached_result:
                    model, cached_preprocessor, metadata = cached_result
                    
                    # Hızlı performans testi
                    quick_test_result = self._quick_performance_test(model, cached_preprocessor, data)
                    
                    # Yeniden eğitim gerekli mi?
                    if not self.cache.should_retrain_model(metadata, quick_test_result):
                        print(f"✅ Cache'den model kullanılıyor: {symbol}")
                        return model, cached_preprocessor, metadata
                    else:
                        print(f"🔄 Incremental training yapılıyor: {symbol}")
                        return self._incremental_training(model_id, model, cached_preprocessor, data, config)
            
            # Yeni model eğit
            print(f"🧠 Yeni model eğitiliyor: {symbol}")
            return self._train_new_model(symbol, data, config, preprocessor)
            
        except Exception as e:
            print(f"❌ Model alma/eğitme hatası: {str(e)}")
            # Fallback: Yeni model eğit
            return self._train_new_model(symbol, data, config, preprocessor)
    
    def _train_new_model(self, symbol: str, data: pd.DataFrame, 
                        config: Dict, preprocessor: 'CryptoDataPreprocessor') -> Tuple['CryptoLSTMModel', 'CryptoDataPreprocessor', Dict]:
        """
        Yeni model eğitir ve cache'ler
        """
        try:
            # Veri hazırla
            scaled_data = preprocessor.scale_data(data)
            X, y = preprocessor.create_sequences(scaled_data, config.get('sequence_length', 60))
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
            
            # Model oluştur ve eğit
            model = CryptoLSTMModel(config.get('sequence_length', 60), X_train.shape[2])
            model.build_model(
                config.get('lstm_units', [64, 64, 32]),
                config.get('dropout_rate', 0.3),
                config.get('learning_rate', 0.001)
            )
            
            # Eğitim
            history = model.train_model(
                X_train, y_train, X_val, y_val,
                epochs=config.get('epochs', 30),
                batch_size=config.get('batch_size', 32)
            )
            
            # Performans değerlendirme
            metrics, predictions = model.evaluate_model(X_test, y_test)
            
            # Directional accuracy hesapla
            if len(y_test) > 1:
                y_test_direction = np.diff(y_test) > 0
                pred_direction = np.diff(predictions.flatten()) > 0
                directional_accuracy = np.mean(y_test_direction == pred_direction)
                metrics['directional_accuracy'] = directional_accuracy
            
            # Eğitim bilgileri
            training_data_info = {
                'data_shape': data.shape,
                'feature_count': X_train.shape[2],
                'training_samples': len(X_train),
                'data_hash': hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()[:16]
            }
            
            # Cache'e kaydet
            model_id = self.cache.save_model(
                model, preprocessor, symbol, config, 
                training_data_info, metrics
            )
            
            return model, preprocessor, {
                'model_id': model_id,
                'training_type': 'new',
                'performance_metrics': metrics,
                'training_data_info': training_data_info
            }
            
        except Exception as e:
            print(f"❌ Yeni model eğitim hatası: {str(e)}")
            raise
    
    def _incremental_training(self, model_id: str, model: 'CryptoLSTMModel',
                            preprocessor: 'CryptoDataPreprocessor', new_data: pd.DataFrame,
                            config: Dict) -> Tuple['CryptoLSTMModel', 'CryptoDataPreprocessor', Dict]:
        """
        Mevcut model üzerinde incremental training yapar
        """
        try:
            print("🔄 Incremental training başlıyor...")
            
            # Yeni veri hazırla (cache'den yüklenen preprocessor için fit_scaler=False)
            scaled_data = preprocessor.scale_data(new_data, fit_scaler=False)
            X, y = preprocessor.create_sequences(scaled_data, config.get('sequence_length', 60))
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
            
            # Düşük learning rate ile eğitim
            incremental_lr = config.get('learning_rate', 0.001) * 0.1  # %10'u kadar
            
            # Model'in learning rate'ini güncelle
            import tensorflow as tf
            tf.keras.backend.set_value(model.model.optimizer.learning_rate, incremental_lr)
            
            # Kısa eğitim (5-10 epoch)
            incremental_epochs = min(10, config.get('epochs', 30) // 3)
            
            # Incremental training için minimal callbacks
            from tensorflow.keras.callbacks import EarlyStopping
            incremental_callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True,
                    verbose=0
                )
            ]
            
            history = model.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=incremental_epochs,
                batch_size=config.get('batch_size', 32),
                verbose=0,
                callbacks=incremental_callbacks
            )
            
            # Yeni performans değerlendirme
            metrics, predictions = model.evaluate_model(X_test, y_test)
            
            # Directional accuracy hesapla
            if len(y_test) > 1:
                y_test_direction = np.diff(y_test) > 0
                pred_direction = np.diff(predictions.flatten()) > 0
                directional_accuracy = np.mean(y_test_direction == pred_direction)
                metrics['directional_accuracy'] = directional_accuracy
            
            # Ek veri bilgileri
            additional_data_info = {
                'incremental_training': True,
                'new_data_shape': new_data.shape,
                'incremental_epochs': incremental_epochs,
                'incremental_lr': incremental_lr
            }
            
            # Cache'i güncelle
            self.cache.update_model(model_id, model, metrics, additional_data_info)
            
            print(f"✅ Incremental training tamamlandı: MSE {metrics['mse']:.6f}")
            
            return model, preprocessor, {
                'model_id': model_id,
                'training_type': 'incremental',
                'performance_metrics': metrics,
                'additional_data_info': additional_data_info
            }
            
        except Exception as e:
            print(f"❌ Incremental training hatası: {str(e)}")
            # Fallback: Yeni model eğit
            return self._train_new_model(symbol, new_data, config, preprocessor)
    
    def _quick_performance_test(self, model: 'CryptoLSTMModel', 
                               preprocessor: 'CryptoDataPreprocessor',
                               data: pd.DataFrame) -> Dict:
        """
        Modelin hızlı performans testi
        """
        try:
            # Son %20'lik veri ile test
            test_size = int(len(data) * 0.2)
            test_data = data.tail(test_size)
            
            scaled_data = preprocessor.scale_data(test_data, fit_scaler=False)
            X, y = preprocessor.create_sequences(scaled_data, 60)
            
            if len(X) > 0:
                metrics, predictions = model.evaluate_model(X, y)
                
                # Directional accuracy hesapla
                if len(y) > 1:
                    y_direction = np.diff(y) > 0
                    pred_direction = np.diff(predictions.flatten()) > 0
                    directional_accuracy = np.mean(y_direction == pred_direction)
                    metrics['directional_accuracy'] = directional_accuracy
                
                return metrics
            
            return {'directional_accuracy': 0.5}  # Varsayılan
            
        except Exception as e:
            print(f"⚠️ Hızlı test hatası: {str(e)}")
            return {'directional_accuracy': 0.5}
    
    def get_cache_info(self) -> Dict:
        """
        Cache bilgilerini döndürür
        """
        stats = self.cache.get_cache_stats()
        models = self.cache.list_cached_models()
        
        return {
            'cache_stats': stats,
            'cached_models': models,
            'valid_models': len([m for m in models if m['valid']]),
            'expired_models': len([m for m in models if not m['valid']])
        }
    
    def cleanup_cache(self) -> Dict:
        """
        Cache temizliği yapar
        """
        deleted_count = self.cache.cleanup_old_models()
        
        return {
            'deleted_models': deleted_count,
            'cleanup_completed': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup_old_models(self, days: int = 7):
        """
        Eski modelleri temizler (CachedModelManager için wrapper)
        
        Args:
            days (int): Kaç gün öncesine kadar temizlenecek
        """
        return self.cache.cleanup_old_models(days)

def main():
    """
    Model cache test fonksiyonu
    """
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                🧠 LSTM MODEL CACHE SİSTEMİ 🧠                     ║
║                                                                    ║
║  Bu sistem eğitilmiş modelleri cache'leyerek eğitim süresini      ║
║  ve işlem gücünü optimize eder.                                   ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    # Cache manager oluştur
    cache_manager = CachedModelManager()
    
    # Cache durumunu göster
    cache_info = cache_manager.get_cache_info()
    
    print(f"\n📊 CACHE DURUMU:")
    print(f"💾 Toplam Model: {cache_info['cache_stats']['total_models']}")
    print(f"✅ Geçerli Model: {cache_info['valid_models']}")
    print(f"⏰ Süresi Dolmuş: {cache_info['expired_models']}")
    print(f"📁 Cache Boyutu: {cache_info['cache_stats']['total_size_mb']} MB")
    
    if cache_info['cached_models']:
        print(f"\n📋 CACHE'DEKİ MODELLER:")
        for model in cache_info['cached_models'][:5]:  # İlk 5'i göster
            status = "✅" if model['valid'] else "❌"
            print(f"   {status} {model['symbol']} (v{model['version']}) - {model['age_days']} gün")
    
    # Seçenekler
    print(f"\n🔧 CACHE YÖNETİMİ:")
    print("1. 🗑️ Eski modelleri temizle")
    print("2. 📊 Detaylı cache istatistikleri")
    print("3. 📋 Tüm modelleri listele")
    print("4. 🚪 Çıkış")
    
    choice = input("\nSeçiminiz (1-4): ").strip()
    
    if choice == '1':
        result = cache_manager.cleanup_cache()
        print(f"✅ {result['deleted_models']} model temizlendi")
    
    elif choice == '2':
        stats = cache_info['cache_stats']
        print(f"\n📊 DETAYLI İSTATİSTİKLER:")
        print(f"Cache Dizini: {stats['cache_dir']}")
        print(f"Toplam Model: {stats['total_models']}")
        print(f"Cache Boyutu: {stats['total_size_mb']} MB")
        
        if stats['models_by_symbol']:
            print("\nSymbol Bazında Dağılım:")
            for symbol, count in stats['models_by_symbol'].items():
                print(f"  {symbol}: {count} model")
    
    elif choice == '3':
        models = cache_info['cached_models']
        print(f"\n📋 TÜM MODELLER ({len(models)}):")
        
        for model in models:
            status = "✅ Geçerli" if model['valid'] else "❌ Süresi Dolmuş"
            perf = model['performance'].get('directional_accuracy', 0)
            
            print(f"""
Model ID: {model['model_id']}
Symbol: {model['symbol']}
Versiyon: {model['version']}
Durum: {status}
Yaş: {model['age_days']} gün
Eğitim Sayısı: {model['training_count']}
Performans: {perf:.1f}% accuracy
Özellik Sayısı: {model['feature_count']}
""")

if __name__ == "__main__":
    main() 