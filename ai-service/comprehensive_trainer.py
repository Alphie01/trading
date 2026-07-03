#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Training System

Bu modül yeni coin eklendiğinde tüm algoritmaları (LSTM, DQN, Hybrid) aynı anda eğitir
ve 4 saatlik & 1 günlük tahminler yapar.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import concurrent.futures
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core imports
from data_fetcher import CryptoDataFetcher
from data_preprocessor import CryptoDataPreprocessor
from lstm_model import CryptoLSTMModel

try:
    from dqn_trading_model import DQNTradingModel
    from hybrid_trading_model import HybridTradingModel
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False

try:
    from model_cache import CachedModelManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from trading_db import TradingDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

class ComprehensiveTrainer:
    """
    Tüm modelleri comprehensive olarak eğiten ana sınıf
    """
    
    def __init__(self, cache_dir: str = "model_cache"):
        """
        Comprehensive Trainer'ı başlatır
        """
        self.data_fetcher = CryptoDataFetcher()
        
        if CACHE_AVAILABLE:
            self.cache_manager = CachedModelManager(cache_dir)
        else:
            self.cache_manager = None
            
        if DATABASE_AVAILABLE:
            self.db = TradingDatabase()
        else:
            self.db = None
            
        # Training configuration
        self.training_config = {
            'first_training': {
                'data_days': 1000,
                'lstm_epochs': 100,
                'dqn_episodes': 150,
                'hybrid_lstm_epochs': 80,
                'hybrid_dqn_episodes': 120
            },
            'fine_tune': {
                'data_days': 200,
                'lstm_epochs': 30,
                'dqn_episodes': 50,
                'hybrid_lstm_epochs': 25,
                'hybrid_dqn_episodes': 40
            }
        }
        
        print("🚀 Comprehensive Trainer başlatıldı!")
        print(f"🧠 DQN/Hybrid: {'✅' if DQN_AVAILABLE else '❌'}")
        print(f"📦 Cache: {'✅' if CACHE_AVAILABLE else '❌'}")
        print(f"💾 Database: {'✅' if DATABASE_AVAILABLE else '❌'}")
    
    async def train_all_models_for_coin(self, coin_symbol: str, is_fine_tune: bool = False) -> Dict:
        """
        Bir coin için tüm modelleri eğitir
        """
        try:
            print(f"\n🎯 {coin_symbol} için comprehensive training başlıyor...")
            print(f"📊 Mod: {'Fine-tune' if is_fine_tune else 'İlk Eğitim'}")
            
            config = self.training_config['fine_tune' if is_fine_tune else 'first_training']
            
            # 1. Veri hazırlama
            data_result = await self._prepare_training_data(coin_symbol, config['data_days'])
            if not data_result['success']:
                return {
                    'success': False,
                    'error': f"Veri hazırlama hatası: {data_result['error']}",
                    'coin_symbol': coin_symbol
                }
            
            processed_df = data_result['processed_data']
            
            # 2. Model eğitimi
            training_results = {}
            
            # LSTM eğitimi
            lstm_result = await self._train_lstm_model_async(coin_symbol, processed_df, config, is_fine_tune)
            training_results['LSTM'] = lstm_result
            
            # DQN eğitimi (eğer available ise)
            if DQN_AVAILABLE:
                dqn_result = await self._train_dqn_model_async(coin_symbol, processed_df, config, is_fine_tune)
                training_results['DQN'] = dqn_result
            
            # Hybrid eğitimi (eğer DQN available ise)
            if DQN_AVAILABLE:
                hybrid_result = await self._train_hybrid_model_async(coin_symbol, processed_df, config, is_fine_tune, training_results)
                training_results['Hybrid'] = hybrid_result
            
            # 3. Multi-timeframe predictions
            predictions = await self._generate_multi_timeframe_predictions(coin_symbol, processed_df, training_results)
            
            # 4. Sonuçları kaydet
            save_result = await self._save_training_results(coin_symbol, training_results, predictions, is_fine_tune)
            
            final_result = {
                'success': True,
                'coin_symbol': coin_symbol,
                'training_mode': 'fine_tune' if is_fine_tune else 'first_training',
                'models_trained': list(training_results.keys()),
                'successful_models': [k for k, v in training_results.items() if v.get('success', False)],
                'failed_models': [k for k, v in training_results.items() if not v.get('success', False)],
                'predictions': predictions,
                'training_results': training_results,
                'saved_to_db': save_result.get('success', False),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"\n🎉 {coin_symbol} comprehensive training tamamlandı!")
            print(f"✅ Başarılı modeller: {len(final_result['successful_models'])}/{len(training_results)}")
            
            return final_result
            
        except Exception as e:
            print(f"❌ Comprehensive training hatası: {e}")
            return {
                'success': False,
                'error': str(e),
                'coin_symbol': coin_symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _prepare_training_data(self, coin_symbol: str, days: int) -> Dict:
        """Eğitim verilerini hazırlar"""
        try:
            print(f"📊 {coin_symbol} için {days} günlük veri çekiliyor...")
            
            df = self.data_fetcher.fetch_ohlcv_data(coin_symbol, days=days)
            if df is None or len(df) < 100:
                return {
                    'success': False,
                    'error': f'Yetersiz veri: {len(df) if df is not None else 0} < 100'
                }
            
            print(f"✅ {len(df)} veri noktası çekildi")
            
            # Veriyi hazırla
            preprocessor = CryptoDataPreprocessor()
            processed_df = preprocessor.prepare_data(df, use_technical_indicators=True)
            
            if processed_df is None or len(processed_df) < 50:
                return {
                    'success': False,
                    'error': 'Veri ön işleme başarısız'
                }
            
            print(f"✅ Veri hazırlama tamamlandı: {len(processed_df)} işlenmiş veri")
            
            return {
                'success': True,
                'data': df,
                'processed_data': processed_df,
                'preprocessor': preprocessor
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _train_lstm_model_async(self, coin_symbol: str, processed_df: pd.DataFrame, 
                                    config: Dict, is_fine_tune: bool) -> Dict:
        """LSTM modelini async olarak eğitir"""
        def train_lstm():
            try:
                print(f"🧠 LSTM eğitimi başlıyor...")
                
                preprocessor = CryptoDataPreprocessor()
                scaled_data = preprocessor.scale_data(processed_df, fit_scaler=True)
                X, y = preprocessor.create_sequences(scaled_data, 60)
                
                if len(X) == 0:
                    return {'success': False, 'error': 'Sequence oluşturulamadı'}
                
                X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
                
                # Model oluştur
                feature_count = X_train.shape[2]
                model = CryptoLSTMModel(60, feature_count)
                model.build_model([50, 50, 50], 0.2, 0.001)
                
                epochs = config['lstm_epochs']
                history = model.train_model(
                    X_train, y_train, X_val, y_val,
                    epochs=epochs,
                    batch_size=32,
                    verbose=1,
                    use_early_stopping=False if not is_fine_tune else True
                )
                
                if not history:
                    return {'success': False, 'error': 'LSTM eğitimi başarısız'}
                
                # Performans testi
                metrics, predictions = model.evaluate_model(X_test, y_test)
                
                # Model kaydet
                os.makedirs("model_cache", exist_ok=True)
                model_file = f"model_cache/lstm_{coin_symbol.lower()}_comprehensive.h5"
                model.save_model(model_file)
                
                print(f"✅ LSTM eğitimi tamamlandı - Accuracy: {metrics.get('directional_accuracy', 0):.2%}")
                
                return {
                    'success': True,
                    'model': model,
                    'preprocessor': preprocessor,
                    'metrics': metrics,
                    'model_file': model_file,
                    'epochs_trained': epochs
                }
                
            except Exception as e:
                print(f"❌ LSTM eğitim hatası: {e}")
                return {'success': False, 'error': str(e)}
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, train_lstm)
    
    async def _train_dqn_model_async(self, coin_symbol: str, processed_df: pd.DataFrame,
                                   config: Dict, is_fine_tune: bool) -> Dict:
        """DQN modelini async olarak eğitir"""
        def train_dqn():
            try:
                print(f"🤖 DQN eğitimi başlıyor...")
                
                dqn_model = DQNTradingModel(lookback_window=60, initial_balance=10000)
                dqn_model.prepare_data(processed_df)
                
                episodes = config['dqn_episodes']
                success = dqn_model.train(processed_df, episodes=episodes, verbose=True)
                
                if not success:
                    return {'success': False, 'error': 'DQN eğitimi başarısız'}
                
                # Model kaydet
                model_file = f"model_cache/dqn_{coin_symbol.lower()}_comprehensive.h5"
                dqn_model.save_model(model_file)
                
                performance = dqn_model.get_performance_summary()
                
                print(f"✅ DQN eğitimi tamamlandı - Episodes: {episodes}")
                
                return {
                    'success': True,
                    'model': dqn_model,
                    'performance': performance,
                    'model_file': model_file,
                    'episodes_trained': episodes
                }
                
            except Exception as e:
                print(f"❌ DQN eğitim hatası: {e}")
                return {'success': False, 'error': str(e)}
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, train_dqn)
    
    async def _train_hybrid_model_async(self, coin_symbol: str, processed_df: pd.DataFrame,
                                      config: Dict, is_fine_tune: bool, 
                                      training_results: Dict) -> Dict:
        """Hybrid modelini async olarak eğitir"""
        def train_hybrid():
            try:
                print(f"🔗 Hybrid eğitimi başlıyor...")
                
                lstm_success = training_results.get('LSTM', {}).get('success', False)
                dqn_success = training_results.get('DQN', {}).get('success', False)
                
                if not lstm_success and not dqn_success:
                    return {'success': False, 'error': 'LSTM ve DQN eğitimi başarısız'}
                
                hybrid_model = HybridTradingModel(sequence_length=60, initial_balance=10000)
                hybrid_model.prepare_models(processed_df)
                
                lstm_epochs = config['hybrid_lstm_epochs']
                dqn_episodes = config['hybrid_dqn_episodes']
                
                success = hybrid_model.train_hybrid_model(
                    processed_df,
                    lstm_epochs=lstm_epochs,
                    dqn_episodes=dqn_episodes,
                    verbose=True
                )
                
                if not success:
                    return {'success': False, 'error': 'Hybrid eğitimi başarısız'}
                
                # Model kaydet
                model_file = f"model_cache/hybrid_{coin_symbol.lower()}_comprehensive"
                hybrid_model.save_hybrid_model(model_file)
                
                performance = hybrid_model.get_model_performance_summary()
                
                print(f"✅ Hybrid eğitimi tamamlandı")
                
                return {
                    'success': True,
                    'model': hybrid_model,
                    'performance': performance,
                    'model_file': model_file,
                    'lstm_epochs': lstm_epochs,
                    'dqn_episodes': dqn_episodes
                }
                
            except Exception as e:
                print(f"❌ Hybrid eğitim hatası: {e}")
                return {'success': False, 'error': str(e)}
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, train_hybrid)
    
    async def _generate_multi_timeframe_predictions(self, coin_symbol: str, 
                                                  processed_df: pd.DataFrame,
                                                  training_results: Dict) -> Dict:
        """4h ve 1d tahminler üretir"""
        try:
            print(f"🔮 Multi-timeframe tahminler oluşturuluyor...")
            
            predictions = {
                '4h': {},
                '1d': {},
                'current_price': processed_df['close'].iloc[-1] if len(processed_df) > 0 else 0
            }
            
            # LSTM tahminleri
            if training_results.get('LSTM', {}).get('success', False):
                lstm_model = training_results['LSTM']['model']
                preprocessor = training_results['LSTM']['preprocessor']
                
                # 4h tahmin
                pred_4h = self._predict_lstm_next_price(lstm_model, preprocessor, processed_df)
                if pred_4h:
                    predictions['4h']['LSTM'] = pred_4h
                
                # 1d tahmin (6 periods = 24h)
                pred_1d = self._predict_lstm_multiple_periods(lstm_model, preprocessor, processed_df, 6)
                if pred_1d:
                    predictions['1d']['LSTM'] = pred_1d
            
            print(f"✅ Tahminler oluşturuldu: 4h={len(predictions['4h'])}, 1d={len(predictions['1d'])}")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Multi-timeframe tahmin hatası: {e}")
            return {'4h': {}, '1d': {}, 'current_price': 0, 'error': str(e)}
    
    def _predict_lstm_next_price(self, model, preprocessor, df):
        """LSTM ile 4h tahmin"""
        try:
            if len(df) < 60:
                return None
                
            # Son 60 veriyi al
            recent_data = df.tail(60)
            scaled_data = preprocessor.scale_data(recent_data, fit_scaler=False)
            
            # Sequence oluştur
            sequence = scaled_data[-60:].reshape(1, 60, -1)
            
            # Tahmin yap
            prediction_normalized = model.predict(sequence)[0][0]
            predicted_price = preprocessor.inverse_transform_prediction(prediction_normalized)
            
            current_price = df['close'].iloc[-1]
            price_change_percent = ((predicted_price - current_price) / current_price) * 100
            
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_percent': price_change_percent,
                'timeframe': '4h',
                'model': 'LSTM',
                'prediction_time': datetime.now(),
                'next_candle_time': datetime.now() + timedelta(hours=4)
            }
            
        except Exception as e:
            print(f"❌ LSTM 4h tahmin hatası: {e}")
            return None
    
    def _predict_lstm_multiple_periods(self, model, preprocessor, df, periods):
        """LSTM ile multi-period tahmin"""
        try:
            predictions = []
            current_df = df.copy()
            
            for i in range(periods):
                pred = self._predict_lstm_next_price(model, preprocessor, current_df)
                if pred:
                    predictions.append(pred)
                    
                    # Sonraki tahmin için veriyi güncelle
                    next_row = current_df.iloc[-1].copy()
                    next_row['close'] = pred['predicted_price']
                    next_row.name = current_df.index[-1] + pd.Timedelta(hours=4)
                    
                    current_df = pd.concat([current_df, next_row.to_frame().T])
                else:
                    break
            
            if predictions:
                final_price = predictions[-1]['predicted_price']
                current_price = df['close'].iloc[-1]
                total_change = ((final_price - current_price) / current_price) * 100
                
                return {
                    'current_price': current_price,
                    'predicted_price': final_price,
                    'price_change_percent': total_change,
                    'timeframe': '1d',
                    'model': 'LSTM',
                    'periods': periods,
                    'individual_predictions': predictions,
                    'prediction_time': datetime.now(),
                    'target_time': datetime.now() + timedelta(hours=4*periods)
                }
            
            return None
            
        except Exception as e:
            print(f"❌ LSTM multi-period tahmin hatası: {e}")
            return None
    
    async def _save_training_results(self, coin_symbol: str, training_results: Dict,
                                   predictions: Dict, is_fine_tune: bool) -> Dict:
        """Eğitim sonuçlarını kaydeder"""
        try:
            # JSON dosyası olarak kaydet
            os.makedirs("training_results", exist_ok=True)
            filename = f"training_results/{coin_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            save_data = {
                'coin_symbol': coin_symbol,
                'training_mode': 'fine_tune' if is_fine_tune else 'first_training',
                'training_results': {k: {
                    'success': v.get('success', False),
                    'error': v.get('error'),
                    'model_file': v.get('model_file'),
                    'metrics': v.get('metrics'),
                    'performance': v.get('performance')
                } for k, v in training_results.items()},
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 Training sonuçları kaydedildi: {filename}")
            
            return {'success': True, 'file': filename}
            
        except Exception as e:
            print(f"❌ Kaydetme hatası: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_coin_sync(self, coin_symbol: str, is_fine_tune: bool = False) -> Dict:
        """
        Synchronous wrapper for comprehensive training
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.train_all_models_for_coin(coin_symbol, is_fine_tune)
            )
            return result
        finally:
            try:
                loop.close()
            except:
                pass

if __name__ == "__main__":
    # Test
    trainer = ComprehensiveTrainer()
    result = trainer.train_coin_sync('BTC', is_fine_tune=False)
    print(f"Test result: {result['success']}") 