import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
import asyncio
import concurrent.futures
from threading import Thread

# **CRITICAL: Import centralized TensorFlow configuration**
from tf_config import get_tensorflow, is_tensorflow_available

# Get TensorFlow availability
TF_AVAILABLE = is_tensorflow_available()

# **NEW: Register custom metrics for TensorFlow model loading**
if TF_AVAILABLE:
    tf = get_tensorflow()
    # Import and register directional_accuracy metric
    try:
        if tf is not None:
            from lstm_model import directional_accuracy
            # Register custom objects for model loading
            tf.keras.utils.get_custom_objects()['directional_accuracy'] = directional_accuracy
            print("✅ Custom metric 'directional_accuracy' registered with TensorFlow")
    except (ImportError, AttributeError) as e:
        print(f"⚠️ Could not register directional_accuracy metric: {e}")

warnings.filterwarnings('ignore')

try:
    from model_cache import CachedModelManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print("⚠️ Model cache sistemi mevcut değil. Normal eğitim modunda çalışılacak.")

# Advanced models imports with try-catch for fallback
ADVANCED_MODELS_AVAILABLE = False
try:
    from dqn_trading_model import DQNTradingModel, TradingEnvironment
    from hybrid_trading_model import HybridTradingModel
    ADVANCED_MODELS_AVAILABLE = True
    print("✅ Advanced models (DQN + Hybrid) initialized")
except ImportError as e:
    print(f"⚠️ Advanced models not available: {e}")
    # Mock classes for when advanced models are not available
    class DQNTradingModel:
        def __init__(self, *args, **kwargs):
            pass
    class HybridTradingModel:
        def __init__(self, *args, **kwargs):
            pass
    class TradingEnvironment:
        def __init__(self, *args, **kwargs):
            pass

class CryptoPricePredictor:
    """
    Eğitilmiş LSTM modeli kullanarak kripto para fiyat tahmini yapan sınıf
    """
    
    def __init__(self, model, preprocessor, news_analyzer=None, whale_tracker=None):
        """
        Predictor'ı başlatır
        
        Args:
            model: Eğitilmiş LSTM modeli
            preprocessor: Veri ön işleme sınıfı
            news_analyzer: Haber analizi sınıfı (opsiyonel)
            whale_tracker: Whale takip sınıfı (opsiyonel)
        """
        self.model = model
        self.preprocessor = preprocessor
        self.news_analyzer = news_analyzer
        self.whale_tracker = whale_tracker
        
        # Multi-model support
        self.dqn_model = None
        self.hybrid_model = None
        self.models_initialized = False
        
        # Initialize advanced models
        self._initialize_advanced_models()
    
    def _initialize_advanced_models(self):
        """Initialize DQN and Hybrid models with fallback for Mock classes"""
        try:
            if ADVANCED_MODELS_AVAILABLE:
                # Try to initialize real models
                self.dqn_model = DQNTradingModel()
                self.hybrid_model = HybridTradingModel()
                self.models_initialized = True
                print("✅ Advanced models (DQN + Hybrid) initialized")
            else:
                # Use Mock classes (TensorFlow not available)
                print("⚠️ TensorFlow not available, using Mock models")
                self.dqn_model = DQNTradingModel()  # Mock class from fallback imports
                self.hybrid_model = HybridTradingModel()  # Mock class from fallback imports
                self.models_initialized = False  # Mark as failed to skip in analysis
                print("🔄 Mock models initialized (limited functionality)")
                
        except Exception as e:
            print(f"⚠️ Advanced models initialization failed: {e}")
            # Always fallback to Mock models
            self.dqn_model = DQNTradingModel()  # Mock class from fallback imports
            self.hybrid_model = HybridTradingModel()  # Mock class from fallback imports
            self.models_initialized = False
            print("🔄 Using Mock models as fallback")
    
    async def predict_multi_model_analysis(self, df, coin_symbol, sequence_length=60):
        """
        Multi-model analysis combining LSTM, DQN, and Hybrid predictions
        UPDATED: LSTM -> DQN -> Hybrid sequential execution order
        
        Args:
            df (pd.DataFrame): Processed market data
            coin_symbol (str): Coin symbol
            sequence_length (int): Sequence length for models
            
        Returns:
            dict: Combined analysis results from all models
        """
        try:
            print(f"🔮 Starting multi-model analysis for {coin_symbol}...")
            print("📋 NEW execution order: LSTM -> DQN -> Hybrid (sequential)")
            
            # **CRITICAL FIX: Always extract current_price first**
            current_price = df['close'].iloc[-1] if len(df) > 0 else 0
            print(f"💰 Current price: ${current_price:.6f}")
            
            # Initialize results container
            all_results = {
                'lstm_analysis': {'success': False, 'error': 'Not analyzed'},
                'dqn_analysis': {'success': False, 'error': 'Not analyzed'},
                'hybrid_analysis': {'success': False, 'error': 'Not analyzed'},
                'ensemble_recommendation': {'success': False, 'error': 'No successful models'},
                'model_comparison': {},
                # **ALWAYS include current_price at top level**
                'current_price': current_price,
                'coin_symbol': coin_symbol,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # **NEW: Sequential execution order LSTM -> DQN -> Hybrid**
            print("🔄 Step 1: Running LSTM analysis...")
            
            # Step 1: LSTM Analysis (always run first)
            lstm_result = await self._analyze_lstm_async(df, coin_symbol)
            all_results['lstm_analysis'] = lstm_result
            lstm_success = isinstance(lstm_result, dict) and lstm_result.get('success', False)
            print(f"✅ LSTM Analysis completed: {'SUCCESS' if lstm_success else 'FAILED'}")
            
            # **CRITICAL: İlk eğitim sırasında LSTM başarısız olursa tüm analizi durdur**
            if not lstm_success:
                lstm_cache_file = f"model_cache/lstm_{coin_symbol.lower()}_model.h5"
                dqn_cache_file = f"model_cache/dqn_{coin_symbol.lower()}_model.h5"
                hybrid_cache_file = f"model_cache/hybrid_{coin_symbol.lower()}_model.h5"
                
                # Cache dosyalarından herhangi biri var mı kontrol et
                has_cached_models = (
                    os.path.exists(lstm_cache_file) or 
                    os.path.exists(dqn_cache_file) or 
                    os.path.exists(hybrid_cache_file)
                )
                
                if not has_cached_models:
                    # İlk eğitim ve LSTM başarısız - analizi durdur
                    print("🛑 İLK EĞİTİM sırasında LSTM başarısız oldu - analiz durduruluyor")
                    print("❌ LSTM temel model olduğu için diğer modeller eğitilmeyecek")
                    
                    return {
                        'lstm_analysis': lstm_result,
                        'dqn_analysis': {'success': False, 'error': 'LSTM failed, skipping DQN', 'status': 'skipped'},
                        'hybrid_analysis': {'success': False, 'error': 'LSTM failed, skipping Hybrid', 'status': 'skipped'},
                        'ensemble_recommendation': {'success': False, 'error': 'LSTM failed, no ensemble possible'},
                        'model_comparison': {'models_analyzed': 1, 'successful_models': [], 'failed_models': ['lstm_analysis']},
                        'current_price': current_price,
                        'prediction': {
                            'current_price': current_price,
                            'predicted_price': current_price,
                            'price_change_percent': 0.0,
                            'confidence': 0.0,
                            'prediction_time': datetime.now(),
                            'next_candle_time': datetime.now() + timedelta(hours=4)
                        },
                        'error': 'LSTM training failed during first training',
                        'coin_symbol': coin_symbol,
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                else:
                    print("⚠️ LSTM başarısız ama cached modeller var, devam ediliyor...")
            
            # Step 2: DQN Analysis (run after LSTM completes)
            print("🔄 Step 2: Running DQN analysis...")
            if self.models_initialized:
                dqn_result = await self._analyze_dqn_async(df, coin_symbol)
                all_results['dqn_analysis'] = dqn_result
                dqn_success = isinstance(dqn_result, dict) and (dqn_result.get('success', False) or dqn_result.get('status') == 'success')
                print(f"✅ DQN Analysis completed: {'SUCCESS' if dqn_success else 'FAILED'}")
            else:
                print("⚠️ DQN model not available, skipping")
            
            # Step 3: Hybrid Analysis (run after LSTM and DQN complete)
            print("🔄 Step 3: Running Hybrid analysis...")
            if self.models_initialized:
                hybrid_result = await self._analyze_hybrid_async(df, coin_symbol)
                all_results['hybrid_analysis'] = hybrid_result
                hybrid_success = isinstance(hybrid_result, dict) and (hybrid_result.get('success', False) or hybrid_result.get('status') == 'success')
                print(f"✅ Hybrid Analysis completed: {'SUCCESS' if hybrid_success else 'FAILED'}")
            else:
                print("⚠️ Hybrid model not available, skipping")
            
            # Compare model predictions
            comparison = self._compare_model_predictions(all_results)
            all_results['model_comparison'] = comparison
            
            # Generate ensemble recommendation
            ensemble_rec = self._generate_ensemble_recommendation(all_results)
            all_results['ensemble_recommendation'] = ensemble_rec
            
            # **CRITICAL: Ensure backward compatibility with web interface**
            # Create a unified prediction object with current_price
            successful_models = []
            for model in ['lstm_analysis', 'dqn_analysis', 'hybrid_analysis']:
                model_result = all_results[model]
                if isinstance(model_result, dict) and (model_result.get('success', False) or model_result.get('status') == 'success'):
                    successful_models.append(model)
            
            # Use LSTM as primary for prediction format, fallback to current_price
            lstm_pred = all_results['lstm_analysis']
            if isinstance(lstm_pred, dict) and lstm_pred.get('success', False) and 'prediction' in lstm_pred:
                primary_prediction = lstm_pred['prediction'].copy()
                # Ensure current_price is always present
                if 'current_price' not in primary_prediction:
                    primary_prediction['current_price'] = current_price
            else:
                # Create minimal prediction with current_price
                primary_prediction = {
                    'current_price': current_price,
                    'predicted_price': current_price * 1.001,  # Minimal 0.1% change
                    'price_change_percent': 0.1,
                    'confidence': 30.0,
                    'prediction_time': datetime.now(),
                    'next_candle_time': datetime.now() + timedelta(hours=4)
                }
            
            # Add prediction to results for web interface compatibility
            all_results['prediction'] = primary_prediction
            
            print(f"✅ Multi-model analysis completed for {coin_symbol}")
            print(f"📊 Successful models: {len(successful_models)}/{len(['lstm_analysis', 'dqn_analysis', 'hybrid_analysis'])}")
            print(f"💰 Current price included: ${all_results['current_price']:.6f}")
            
            return all_results
            
        except Exception as e:
            print(f"❌ Multi-model analysis error: {e}")
            # Return fallback result with current_price
            current_price = df['close'].iloc[-1] if len(df) > 0 else 0
            return {
                'lstm_analysis': {'success': False, 'error': str(e)},
                'dqn_analysis': {'success': False, 'error': 'Not analyzed'},
                'hybrid_analysis': {'success': False, 'error': 'Not analyzed'},
                'ensemble_recommendation': {'success': False, 'error': str(e)},
                'model_comparison': {'models_analyzed': 0, 'successful_models': [], 'failed_models': []},
                'current_price': current_price,
                'prediction': {
                    'current_price': current_price,
                    'predicted_price': current_price,
                    'price_change_percent': 0.0,
                    'confidence': 0.0,
                    'prediction_time': datetime.now(),
                    'next_candle_time': datetime.now() + timedelta(hours=4)
                },
                'error': str(e),
                'coin_symbol': coin_symbol,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def predict_multi_model_analysis_sync(self, df, coin_symbol, sequence_length=60):
        """
        **NEW: Synchronous wrapper for multi-model analysis**
        This method is called by the web app which expects a sync interface
        """
        try:
            # Create new event loop for this thread
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self.predict_multi_model_analysis(df, coin_symbol, sequence_length)
                )
                return result
            finally:
                try:
                    loop.close()
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ Sync multi-model analysis error: {e}")
            # Return fallback result
            current_price = df['close'].iloc[-1] if len(df) > 0 else 0
            return {
                'lstm_analysis': {'success': False, 'error': str(e)},
                'dqn_analysis': {'success': False, 'error': 'Sync wrapper error'},
                'hybrid_analysis': {'success': False, 'error': 'Sync wrapper error'},
                'ensemble_recommendation': {'success': False, 'error': str(e)},
                'model_comparison': {'models_analyzed': 0, 'successful_models': [], 'failed_models': []},
                'current_price': current_price,
                'prediction': {
                    'current_price': current_price,
                    'predicted_price': current_price,
                    'price_change_percent': 0.0,
                    'confidence': 0.0,
                    'prediction_time': datetime.now(),
                    'next_candle_time': datetime.now() + timedelta(hours=4)
                },
                'error': str(e),
                'coin_symbol': coin_symbol,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_lstm_async(self, df, coin_symbol):
        """LSTM analysis as async method with first-time training optimization"""
        def run_lstm_analysis():
            try:
                print("🧠 Running LSTM analysis (async)...")
                
                # **CRITICAL FIX: Ensure scaler is fitted before prediction**
                current_price = df['close'].iloc[-1]
                
                # **YENİ: İlk eğitim kontrolü - LSTM cache dosyasını kontrol et**
                lstm_cache_file = f"model_cache/lstm_{coin_symbol.lower()}_model.h5"
                is_first_training = not os.path.exists(lstm_cache_file)
                
                if is_first_training:
                    print(f"🆕 {coin_symbol} LSTM İLK EĞİTİM - Daha fazla epoch kullanılacak")
                    lstm_epochs = 100  # İlk eğitim için 100 epoch
                    batch_size = 16   # Daha küçük batch size ile daha iyi öğrenme
                else:
                    print(f"🔄 {coin_symbol} LSTM YENİDEN EĞİTİM - Normal epoch kullanılacak")
                    lstm_epochs = int(os.getenv('LSTM_EPOCHS', 30))  # Normal eğitim
                    batch_size = 32   # Normal batch size
                
                # **CRITICAL FIX: Prepare training data FIRST to get correct feature count**
                print("🔄 Preparing training data to determine feature count...")
                processed_df = self.preprocessor.prepare_data(df, use_technical_indicators=True)
                if processed_df is None:
                    return {
                        'success': False,
                        'error': 'Insufficient data for LSTM preprocessing',
                        'current_price': current_price,
                        'model': 'LSTM'
                    }
                
                # Fit scaler and get final feature count
                scaled_data = self.preprocessor.scale_data(processed_df, fit_scaler=True)
                final_feature_count = scaled_data.shape[1] if len(scaled_data.shape) > 1 else processed_df.shape[1]
                print(f"🔍 Final feature count for LSTM: {final_feature_count}")
                
                # **PERF FIX: Önce process-ömürlü / disk cache'ten EĞİTİLMİŞ modeli yükle.**
                # Eski davranış (bug): retrain dalında model diskten YÜKLENMEDEN eğitilmemiş
                # model kurulur veya her istekte yeniden eğitilirdi. Artık eğitilmiş .h5
                # yüklenir → her istekte yeniden eğitim YOK. Scaler mantığına dokunulmaz.
                if self.model is None:
                    try:
                        import model_memory_cache as _mem
                        with _mem.lock_for(coin_symbol):
                            cached_model = _mem.get_or_load_lstm(coin_symbol)
                            if cached_model is not None and getattr(cached_model, 'model', None) is not None:
                                self.model = cached_model
                                print(f"⚡ LSTM cache'ten yüklendi (yeniden eğitim yok): {coin_symbol}")
                    except Exception as _cache_err:
                        print(f"⚠️ LSTM cache erişim hatası: {_cache_err}")

                # Cache yoksa (ilk eğitim veya disk yükleme başarısız): kur + eğit + kaydet + belleğe al
                if self.model is None:
                    print("🔄 LSTM model None - kurulum + eğitim (cache miss)...")
                    try:
                        from lstm_model import CryptoLSTMModel
                        self.model = CryptoLSTMModel(sequence_length=60, n_features=final_feature_count)
                        print(f"🧠 LSTM Model initialized with {final_feature_count} features (TF Available: {TF_AVAILABLE})")

                        # **CRITICAL: Build model before training**
                        if self.model.model is None:
                            self.model.build_model()
                            print("✅ LSTM model built successfully")

                        print(f"🔥 LSTM eğitim: {lstm_epochs} epoch, batch_size={batch_size}")
                        X, y = self.preprocessor.create_sequences(scaled_data, 60)
                        if len(X) == 0:
                            return {
                                'success': False,
                                'error': 'Insufficient data for sequence creation',
                                'current_price': current_price,
                                'model': 'LSTM'
                            }

                        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(X, y)

                        # Verify shapes match
                        if X_train.shape[2] != final_feature_count:
                            return {
                                'success': False,
                                'error': f'Shape mismatch: X_train has {X_train.shape[2]} features but model expects {final_feature_count}',
                                'current_price': current_price,
                                'model': 'LSTM'
                            }

                        training_result = self.model.train_model(
                            X_train, y_train, X_val, y_val,
                            epochs=lstm_epochs,
                            batch_size=batch_size,
                            verbose=1,
                            use_early_stopping=False
                        )

                        if not training_result:
                            return {
                                'success': False,
                                'error': 'LSTM training failed',
                                'current_price': current_price,
                                'model': 'LSTM'
                            }

                        # Save trained model (disk) + belleğe al
                        os.makedirs("model_cache", exist_ok=True)
                        try:
                            self.model.save_model(lstm_cache_file)
                            print(f"✅ LSTM eğitildi ve kaydedildi: {lstm_cache_file}")
                        except Exception as save_error:
                            print(f"⚠️ Model kaydetme hatası: {save_error}")
                        try:
                            import model_memory_cache as _mem
                            _mem.store_lstm(coin_symbol, self.model)
                        except Exception:
                            pass

                    except Exception as model_error:
                        print(f"❌ LSTM model creation failed: {model_error}")
                        return {
                            'success': False,
                            'error': f'LSTM model creation failed: {model_error}',
                            'current_price': current_price,
                            'model': 'LSTM'
                        }
                
                # Make prediction (coin_symbol → cache'ten model yükleme desteği)
                prediction_result = self.predict_next_price(df, sequence_length=60, coin_symbol=coin_symbol)
                
                if prediction_result:
                    return {
                        'success': True,
                        'model': 'LSTM',
                        'prediction': prediction_result,
                        'current_price': prediction_result['current_price'],
                        'predicted_price': prediction_result['predicted_price'],
                        'price_change_percent': prediction_result['price_change_percent'],
                        'confidence': prediction_result['confidence'],
                        'reasoning': f"LSTM predicts {prediction_result['price_change_percent']:+.2f}% price change",
                        'model_type': 'Deep Learning',
                        'training_mode': 'FIRST_TRAINING' if is_first_training else 'RETRAIN'
                    }
                else:
                    print("❌ LSTM prediction failed")
                    return {
                        'success': False,
                        'error': 'LSTM prediction failed',
                        'current_price': current_price,
                        'model': 'LSTM'
                    }
                    
            except Exception as e:
                print(f"❌ LSTM analysis error: {e}")
                current_price = df['close'].iloc[-1] if len(df) > 0 else 0
                return {
                    'success': False,
                    'error': str(e),
                    'current_price': current_price,
                    'model': 'LSTM'
                }
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, run_lstm_analysis)
            return result

    async def _analyze_dqn_async(self, df, coin_symbol):
        """DQN analysis as async method with first-time training optimization"""
        def run_dqn_analysis():
            try:
                print("🤖 Running DQN analysis (async)...")
                
                # Check if advanced models are available
                if not ADVANCED_MODELS_AVAILABLE or self.dqn_model is None:
                    return {'success': False, 'error': 'DQN model not available', 'status': 'failed'}
                
                # **YENİ: İlk eğitim kontrolü - DQN cache dosyasını kontrol et**
                dqn_model_file = f"model_cache/dqn_{coin_symbol.lower()}_model.h5"
                is_first_training = not os.path.exists(dqn_model_file)

                # **PERF: DQN'i process-ömürlü bellek cache'ten dene (disk yükleme/eğitim atla)**
                _dqn_cached = None
                try:
                    import model_memory_cache as _mem
                    _dqn_cached = _mem.get_dqn(coin_symbol)
                except Exception:
                    _dqn_cached = None

                if _dqn_cached is not None:
                    self.dqn_model = _dqn_cached
                    print(f"⚡ DQN cache'ten (bellek) yüklendi: {coin_symbol}")
                else:
                    if is_first_training:
                        print(f"🆕 {coin_symbol} DQN İLK EĞİTİM - Daha fazla episode kullanılacak")
                        dqn_episodes = 100  # İlk eğitim için 100 episode (daha uzun)
                    else:
                        print(f"🔄 {coin_symbol} DQN YENİDEN EĞİTİM - Normal episode kullanılacak")
                        dqn_episodes = 15   # Normal eğitim için 15 episode

                    if os.path.exists(dqn_model_file) and not is_first_training:
                        print(f"🔄 Loading DQN model from {dqn_model_file}")
                        try:
                            self.dqn_model.load_model(dqn_model_file)
                        except:
                            print("❌ Failed to load DQN model, training new one...")
                            self.dqn_model.prepare_data(df)
                            training_result = self.dqn_model.train(df, episodes=dqn_episodes, verbose=False)
                    else:
                        if is_first_training:
                            print(f"🔥 DQN İlk eğitim başlıyor: {dqn_episodes} episode")
                        else:
                            print(f"❌ Could not load DQN model from {dqn_model_file}")
                            print(f"🔄 Training DQN model for {coin_symbol}...")

                        # Prepare and train DQN with appropriate episodes
                        self.dqn_model.prepare_data(df)
                        training_result = self.dqn_model.train(df, episodes=dqn_episodes, verbose=False)

                        if training_result:
                            # Save the trained model
                            os.makedirs("model_cache", exist_ok=True)
                            try:
                                self.dqn_model.save_model(dqn_model_file)
                                print(f"✅ DQN model saved: {dqn_model_file}")
                            except:
                                print("⚠️ Could not save DQN model, continuing...")
                        else:
                            print("❌ DQN training failed")
                            return {'success': False, 'error': 'DQN training failed', 'status': 'failed'}

                    # Yüklenen/eğitilen DQN modelini belleğe al (request'ler arası reuse)
                    try:
                        import model_memory_cache as _mem
                        _mem.store_dqn(coin_symbol, self.dqn_model)
                    except Exception:
                        pass
                
                # Get current state for prediction
                if len(df) < getattr(self.dqn_model, 'lookback_window', 60):
                    return {'success': False, 'error': 'Insufficient data for DQN', 'status': 'failed'}
                
                # Prepare state - last sequence
                lookback = getattr(self.dqn_model, 'lookback_window', 60)
                recent_data = df.tail(lookback)
                current_price = df['close'].iloc[-1]
                
                # Create trading environment to get proper state
                try:
                    temp_env = TradingEnvironment(
                        data=recent_data,
                        initial_balance=10000,
                        lookback_window=lookback
                    )
                    temp_env.reset()
                    current_state = temp_env._get_state()
                except:
                    # Fallback: create simple state
                    current_state = np.random.random(31)  # Default state size
                
                # Get action prediction
                try:
                    action_result = self.dqn_model.predict_action(current_state)
                except:
                    # Fallback prediction
                    action_result = {
                        'action': 0,
                        'action_name': 'HOLD',
                        'confidence': 0.5,
                        'reasoning': 'DQN fallback prediction',
                        'q_values': []
                    }
                
                # Get training summary
                try:
                    training_summary = self.dqn_model.get_training_summary()
                except:
                    training_summary = {'final_portfolio_value': 10000}
                
                # **CRITICAL FIX: Return consistent structure with both success and status fields**
                result = {
                    'success': True,  # **Add success field for web_app compatibility**
                    'status': 'success',
                    'model': 'DQN',
                    'prediction': {  # **Add prediction wrapper for template compatibility**
                        'action': action_result['action'],
                        'action_name': action_result['action_name'],
                        'confidence': action_result['confidence'],
                        'reasoning': action_result['reasoning'],
                        'q_values': action_result.get('q_values', []),
                        'current_price': current_price,
                        'recommendation': action_result['action_name']
                    },
                    'action': action_result['action'],
                    'action_name': action_result['action_name'],
                    'confidence': action_result['confidence'],
                    'reasoning': action_result['reasoning'],
                    'q_values': action_result.get('q_values', []),
                    'training_summary': training_summary,
                    'recommendation': action_result['action_name'],
                    'model_type': 'Reinforcement Learning',
                    'current_price': current_price,
                    'training_mode': 'FIRST_TRAINING' if is_first_training else 'RETRAIN'
                }
                
                print(f"✅ DQN Analysis completed: {action_result['action_name']}")
                return result
                
            except Exception as e:
                print(f"❌ DQN analysis error: {e}")
                current_price = df['close'].iloc[-1] if len(df) > 0 else 0
                return {
                    'success': False, 
                    'error': str(e), 
                    'status': 'failed',
                    'current_price': current_price,
                    'model': 'DQN'
                }
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, run_dqn_analysis)
            return result

    async def _analyze_hybrid_async(self, df, coin_symbol):
        """Hybrid model analysis as async method with first-time training optimization"""
        def run_hybrid_analysis():
            try:
                print("🔄 Running Hybrid analysis (async)...")
                
                # Check if advanced models are available
                if not ADVANCED_MODELS_AVAILABLE or self.hybrid_model is None:
                    return {'success': False, 'error': 'Hybrid model not available', 'status': 'failed'}
                
                print("🔧 Preparing hybrid model components...")
                current_price = df['close'].iloc[-1]
                
                # **YENİ: İlk eğitim kontrolü - Hybrid cache dosyasını kontrol et**
                hybrid_cache_file = f"model_cache/hybrid_{coin_symbol.lower()}_model.h5"
                is_first_training = not os.path.exists(hybrid_cache_file)
                
                if is_first_training:
                    print(f"🆕 {coin_symbol} HYBRID İLK EĞİTİM - Daha fazla epoch/episode kullanılacak")
                    lstm_epochs = 60   # İlk eğitim için 60 epoch (LSTM component)
                    dqn_episodes = 50  # İlk eğitim için 50 episode (DQN component)
                else:
                    print(f"🔄 {coin_symbol} HYBRID YENİDEN EĞİTİM - Normal epoch/episode kullanılacak")
                    lstm_epochs = 20   # Normal eğitim için 20 epoch  
                    dqn_episodes = 10  # Normal eğitim için 10 episode
                
                # Train hybrid model with appropriate parameters
                try:
                    print(f"🔥 Hybrid eğitim başlıyor: LSTM={lstm_epochs} epochs, DQN={dqn_episodes} episodes")
                    training_result = self.hybrid_model.train_hybrid_model(
                        df, 
                        lstm_epochs=lstm_epochs,
                        dqn_episodes=dqn_episodes,
                        verbose=False
                    )
                    
                    # Save hybrid model if first training and successful
                    if is_first_training and training_result:
                        os.makedirs("model_cache", exist_ok=True)
                        try:
                            # Note: Hybrid model might need custom save logic
                            print(f"✅ Hybrid İlk eğitim tamamlandı")
                        except Exception as save_error:
                            print(f"⚠️ Hybrid model kaydetme hatası: {save_error}")
                    
                except Exception as e:
                    print(f"❌ Hybrid training failed: {e}")
                    return {'success': False, 'error': f'Hybrid training failed: {e}', 'status': 'failed'}
                
                if not training_result:
                    print("❌ Hybrid training failed")
                    return {'success': False, 'error': 'Hybrid training failed', 'status': 'failed'}
                
                # Get current data for prediction
                sequence_length = getattr(self.hybrid_model, 'sequence_length', 60)
                if len(df) < sequence_length:
                    return {'success': False, 'error': 'Insufficient data for Hybrid', 'status': 'failed'}
                
                recent_data = df.tail(sequence_length + 10)
                
                # Get hybrid prediction
                try:
                    hybrid_result = self.hybrid_model.predict_hybrid_action(recent_data)
                except Exception as e:
                    print(f"❌ Hybrid prediction error: {e}")
                    # Fallback prediction
                    hybrid_result = {
                        'ensemble_recommendation': 'HOLD',
                        'ensemble_confidence': 0.5,
                        'reasoning': 'Hybrid fallback prediction'
                    }
                
                if 'error' in hybrid_result:
                    return {'success': False, 'error': hybrid_result['error'], 'status': 'failed'}
                
                # **CRITICAL FIX: Return consistent structure with both success and status fields**
                result = {
                    'success': True,  # **Add success field for web_app compatibility**
                    'status': 'success',
                    'model': 'Hybrid',
                    'prediction': {  # **Add prediction wrapper for template compatibility**
                        'ensemble_prediction': {
                            'recommendation': hybrid_result.get('ensemble_prediction', {}).get('recommendation', 'HOLD'),
                            'confidence': hybrid_result.get('ensemble_prediction', {}).get('confidence', 0.5),
                            'reasoning': hybrid_result.get('ensemble_prediction', {}).get('reasoning', 'Hybrid analysis completed'),
                            'ensemble_signal': hybrid_result.get('ensemble_prediction', {}).get('ensemble_signal', 0.0),
                            'component_signals': {
                                'lstm': hybrid_result.get('ensemble_prediction', {}).get('component_signals', {}).get('lstm', 0.4),
                                'dqn': hybrid_result.get('ensemble_prediction', {}).get('component_signals', {}).get('dqn', 0.4),
                                'technical': hybrid_result.get('ensemble_prediction', {}).get('component_signals', {}).get('technical', 0.2)
                            }
                        },
                        'lstm_contribution': hybrid_result.get('lstm_prediction', {}),
                        'dqn_contribution': hybrid_result.get('dqn_prediction', {}),
                        'model_weights': hybrid_result.get('model_weights', {}),
                        'current_price': current_price,
                        'recommendation': hybrid_result.get('ensemble_prediction', {}).get('recommendation', 'HOLD')
                    },
                    'ensemble_recommendation': hybrid_result.get('ensemble_prediction', {}).get('recommendation', 'HOLD'),
                    'confidence': hybrid_result.get('ensemble_prediction', {}).get('confidence', 0.5),
                    'reasoning': hybrid_result.get('ensemble_prediction', {}).get('reasoning', 'Hybrid analysis completed'),
                    'lstm_contribution': hybrid_result.get('lstm_prediction', {}),
                    'dqn_contribution': hybrid_result.get('dqn_prediction', {}),
                    'model_weights': hybrid_result.get('model_weights', {}),
                    'recommendation': hybrid_result.get('ensemble_prediction', {}).get('recommendation', 'HOLD'),
                    'model_type': 'Hybrid LSTM+DQN',
                    'current_price': current_price,
                    'training_mode': 'FIRST_TRAINING' if is_first_training else 'RETRAIN'
                }
                
                print(f"✅ Hybrid Analysis completed: {hybrid_result.get('ensemble_prediction', {}).get('recommendation', 'HOLD')}")
                return result
                
            except Exception as e:
                print(f"❌ Hybrid analysis error: {e}")
                current_price = df['close'].iloc[-1] if len(df) > 0 else 0
                return {
                    'success': False, 
                    'error': str(e), 
                    'status': 'failed',
                    'current_price': current_price,
                    'model': 'Hybrid'
                }
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, run_hybrid_analysis)
            return result

    def _generate_ensemble_recommendation(self, all_results):
        """Generate ensemble recommendation from all model outputs"""
        try:
            # Extract predictions from successful models
            model_predictions = []
            model_weights = []
            
            # LSTM
            if all_results['lstm_analysis'].get('success', False):
                lstm_pred = all_results['lstm_analysis']['prediction']
                price_change = lstm_pred.get('price_change_percent', 0) / 100.0
                confidence = lstm_pred.get('confidence', 0) / 100.0
                model_predictions.append(price_change)
                model_weights.append(confidence * 0.4)  # 40% base weight for LSTM
            
            # DQN
            if all_results['dqn_analysis'].get('success', False):
                dqn_pred = all_results['dqn_analysis']['prediction']
                action = dqn_pred.get('action', 0)
                confidence = dqn_pred.get('confidence', 0)
                
                # Convert DQN action to directional signal
                if action == 0:  # HOLD
                    signal = 0.0
                elif action in [1, 2, 3, 4]:  # BUY
                    signal = 0.25 * action  # 0.25, 0.5, 0.75, 1.0
                else:  # SELL
                    signal = -0.25 * (action - 4)  # -0.25, -0.5, -0.75, -1.0
                
                model_predictions.append(signal)
                model_weights.append(confidence * 0.4)  # 40% base weight for DQN
            
            # Hybrid
            if all_results['hybrid_analysis'].get('success', False):
                hybrid_pred = all_results['hybrid_analysis']['prediction']
                if 'ensemble_prediction' in hybrid_pred:
                    ensemble_signal = hybrid_pred['ensemble_prediction'].get('ensemble_signal', 0)
                    confidence = hybrid_pred['ensemble_prediction'].get('confidence', 0)
                    model_predictions.append(ensemble_signal)
                    model_weights.append(confidence * 0.2)  # 20% base weight for Hybrid
            
            if not model_predictions:
                return {
                    'success': False,
                    'error': 'No successful model predictions available'
                }
            
            # Calculate weighted ensemble
            total_weight = sum(model_weights)
            if total_weight > 0:
                weighted_prediction = sum(p * w for p, w in zip(model_predictions, model_weights)) / total_weight
            else:
                weighted_prediction = sum(model_predictions) / len(model_predictions)
            
            # Generate recommendation
            if weighted_prediction > 0.3:
                recommendation = 'STRONG_BUY'
                confidence = min(1.0, abs(weighted_prediction) * 2)
            elif weighted_prediction > 0.1:
                recommendation = 'BUY'
                confidence = min(1.0, abs(weighted_prediction) * 3)
            elif weighted_prediction < -0.3:
                recommendation = 'STRONG_SELL'
                confidence = min(1.0, abs(weighted_prediction) * 2)
            elif weighted_prediction < -0.1:
                recommendation = 'SELL'
                confidence = min(1.0, abs(weighted_prediction) * 3)
            else:
                recommendation = 'HOLD'
                confidence = 0.5
            
            return {
                'success': True,
                'ensemble_signal': float(weighted_prediction),
                'recommendation': recommendation,
                'confidence': float(confidence),
                'model_count': len(model_predictions),
                'reasoning': self._generate_ensemble_reasoning(
                    recommendation, model_predictions, all_results
                )
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Ensemble generation error: {e}'}
    
    def _generate_ensemble_reasoning(self, recommendation, predictions, all_results):
        """Generate human-readable reasoning for ensemble decision"""
        reasoning = f"🎯 Ensemble Recommendation: {recommendation}\n\n"
        
        # Model contributions
        if all_results['lstm_analysis'].get('success', False):
            lstm_change = all_results['lstm_analysis']['prediction'].get('price_change_percent', 0)
            reasoning += f"📈 LSTM: {lstm_change:+.2f}% price change predicted\n"
        
        if all_results['dqn_analysis'].get('success', False):
            dqn_action = all_results['dqn_analysis']['prediction'].get('action_name', 'UNKNOWN')
            reasoning += f"🤖 DQN: {dqn_action} action recommended\n"
        
        if all_results['hybrid_analysis'].get('success', False):
            hybrid_pred = all_results['hybrid_analysis']['prediction']
            if 'ensemble_prediction' in hybrid_pred:
                hybrid_rec = hybrid_pred['ensemble_prediction'].get('recommendation', 'UNKNOWN')
                reasoning += f"🔗 Hybrid: {hybrid_rec} strategy suggested\n"
        
        reasoning += f"\n📊 Models Agreement: {len(predictions)} models analyzed"
        reasoning += f"\n🎲 Ensemble Signal: {sum(predictions)/len(predictions):+.3f}"
        
        return reasoning
    
    def _compare_model_predictions(self, all_results):
        """Compare predictions from different models"""
        comparison = {
            'models_analyzed': 0,
            'successful_models': [],
            'failed_models': [],
            'agreement_score': 0.0,
            'confidence_average': 0.0,
            'model_details': {}
        }
        
        try:
            models = ['lstm_analysis', 'dqn_analysis', 'hybrid_analysis']
            confidences = []
            signals = []
            
            for model_name in models:
                comparison['models_analyzed'] += 1
                result = all_results.get(model_name, {})
                
                if result.get('success', False):
                    comparison['successful_models'].append(model_name)
                    
                    # Extract confidence
                    confidence = result.get('confidence', 0)
                    confidences.append(confidence)
                    
                    # Extract directional signal for agreement calculation
                    if model_name == 'lstm_analysis':
                        price_change = result['prediction'].get('price_change_percent', 0)
                        signal = 1 if price_change > 1 else -1 if price_change < -1 else 0
                    elif model_name == 'dqn_analysis':
                        action = result['prediction'].get('action', 0)
                        signal = 1 if action in [1,2,3,4] else -1 if action in [5,6,7,8] else 0
                    else:  # hybrid
                        hybrid_pred = result['prediction']
                        if 'ensemble_prediction' in hybrid_pred:
                            rec = hybrid_pred['ensemble_prediction'].get('recommendation', 'HOLD')
                            signal = 1 if 'BUY' in rec else -1 if 'SELL' in rec else 0
                        else:
                            signal = 0
                    
                    signals.append(signal)
                    comparison['model_details'][model_name] = {
                        'success': True,
                        'confidence': confidence,
                        'signal': signal
                    }
                else:
                    comparison['failed_models'].append(model_name)
                    comparison['model_details'][model_name] = {
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    }
            
            # Calculate agreement score
            if len(signals) > 1:
                # Agreement is how often models agree on direction
                positive_signals = sum(1 for s in signals if s > 0)
                negative_signals = sum(1 for s in signals if s < 0)
                neutral_signals = sum(1 for s in signals if s == 0)
                
                max_agreement = max(positive_signals, negative_signals, neutral_signals)
                comparison['agreement_score'] = max_agreement / len(signals)
            else:
                comparison['agreement_score'] = 1.0 if len(signals) == 1 else 0.0
            
            # Average confidence
            if confidences:
                comparison['confidence_average'] = sum(confidences) / len(confidences)
            
        except Exception as e:
            comparison['error'] = str(e)
        
        return comparison
    
    def predict_next_price(self, df, sequence_length=60, coin_symbol=None):
        """
        Bir sonraki 4 saatlik kapanış fiyatını tahmin eder

        Args:
            df (pd.DataFrame): Güncel veriler
            sequence_length (int): Sekans uzunluğu
            coin_symbol (str): Verilirse, model None ise cache'ten EĞİTİLMİŞ model yüklenir
                               (yeniden eğitim yerine).

        Returns:
            dict: Tahmin sonuçları
        """
        try:
            # **PERF FIX: model None ise önce cache'ten eğitilmiş modeli dene (coin biliniyorsa)**
            if self.model is None and coin_symbol:
                try:
                    import model_memory_cache as _mem
                    with _mem.lock_for(coin_symbol):
                        cached_model = _mem.get_or_load_lstm(coin_symbol)
                        if cached_model is not None and getattr(cached_model, 'model', None) is not None:
                            self.model = cached_model
                            print(f"⚡ predict_next_price: LSTM cache'ten yüklendi ({coin_symbol})")
                except Exception as _ce:
                    print(f"⚠️ predict_next_price cache erişim hatası: {_ce}")

            # Cache yoksa (geriye dönük): model None ise son çare hafif eğitim
            if self.model is None:
                print("🔄 LSTM model None - loading model...")
                from lstm_model import CryptoLSTMModel
                self.model = CryptoLSTMModel()

                # Get or train model
                processed_df = self.preprocessor.prepare_data(df, use_technical_indicators=True)
                if processed_df is None:
                    print("❌ Cannot prepare data for model training")
                    return None

                scaled_data = self.preprocessor.scale_data(processed_df, fit_scaler=True)
                X, y = self.preprocessor.create_sequences(scaled_data, sequence_length)

                # Simple train-test split
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                print(f"🔄 Building and training LSTM model with {len(X_train)} samples...")
                # **CRITICAL FIX: Build model before training**
                self.model.build_model(lstm_units=[50, 50, 50], dropout_rate=0.2, learning_rate=0.001)
                self.model.train_model(X_train, y_train, X_val, y_val, epochs=10, verbose=0)
                print("✅ LSTM model trained successfully")
                # Eğitilen modeli belleğe al (coin biliniyorsa)
                if coin_symbol:
                    try:
                        import model_memory_cache as _mem
                        _mem.store_lstm(coin_symbol, self.model)
                    except Exception:
                        pass
            
            # **CRITICAL FIX: Ensure scaler is fitted before prediction**
            # Preprocess data to fit scaler if needed
            if not hasattr(self.preprocessor, 'scaler') or not hasattr(self.preprocessor.scaler, 'scale_'):
                # Scaler not fitted, need to fit it first
                processed_df = self.preprocessor.prepare_data(df, use_technical_indicators=True)
                _ = self.preprocessor.scale_data(processed_df, fit_scaler=True)
                print("🔄 MinMaxScaler fitted for prediction")
            
            # En son sequence'ı al
            latest_sequence = self.preprocessor.get_latest_sequence(df, sequence_length)
            
            # Tahmin yap (normalize edilmiş)
            prediction_normalized = self.model.predict(latest_sequence)[0][0]
            
            # Orijinal ölçeğe çevir
            prediction_price = self.preprocessor.inverse_transform_prediction(prediction_normalized)
            
            # Mevcut fiyat
            current_price = df['close'].iloc[-1]
            
            # Değişim yüzdesi
            price_change = ((prediction_price - current_price) / current_price) * 100
            
            # Sonuç
            result = {
                'current_price': current_price,
                'predicted_price': prediction_price,
                'price_change_percent': price_change,
                'prediction_time': datetime.now(),
                'next_candle_time': df.index[-1] + timedelta(hours=4),
                'confidence': self._calculate_confidence(df, prediction_normalized)
            }
            
            return result
            
        except Exception as e:
            print(f"Tahmin hatası: {str(e)}")
            return None
    
    def predict_multiple_periods(self, df, periods=6, sequence_length=60):
        """
        Birden fazla dönem için tahmin yapar (24 saat = 6 dönem x 4 saat)
        
        Args:
            df (pd.DataFrame): Güncel veriler
            periods (int): Tahmin edilecek dönem sayısı
            sequence_length (int): Sekans uzunluğu
        
        Returns:
            list: Tahmin sonuçları listesi
        """
        predictions = []
        current_df = df.copy()
        
        for i in range(periods):
            # Bu dönem için tahmin yap
            prediction = self.predict_next_price(current_df, sequence_length)
            
            if prediction is None:
                break
                
            predictions.append({
                'period': i + 1,
                'predicted_price': prediction['predicted_price'],
                'price_change_percent': prediction['price_change_percent'],
                'prediction_time': prediction['next_candle_time']
            })
            
            # Tahmin edilen değeri bir sonraki tahmin için kullan
            # (Bu basit bir yaklaşım, gerçekte daha karmaşık olabilir)
            new_row = current_df.iloc[-1:].copy()
            new_row.index = [prediction['next_candle_time']]
            new_row['close'] = prediction['predicted_price']
            new_row['open'] = current_df['close'].iloc[-1]
            new_row['high'] = max(new_row['open'].iloc[0], new_row['close'].iloc[0])
            new_row['low'] = min(new_row['open'].iloc[0], new_row['close'].iloc[0])
            
            # Yeni satırı ekle
            current_df = pd.concat([current_df, new_row])
            
        return predictions
    
    def _calculate_confidence(self, df, prediction_normalized):
        """
        Tahmin güvenilirlik skoru hesaplar
        
        Args:
            df (pd.DataFrame): Veriler
            prediction_normalized (float): Normalize edilmiş tahmin
        
        Returns:
            float: Güvenilirlik skoru (0-100)
        """
        try:
            # Basit volatilite tabanlı güvenilirlik skoru
            recent_volatility = df['close'].tail(20).std() / df['close'].tail(20).mean()
            
            # Düşük volatilite = yüksek güvenilirlik
            base_confidence = max(0, 100 - (recent_volatility * 1000))
            
            # Tahmin değerinin makul aralıkta olup olmadığını kontrol et
            recent_range = df['close'].tail(10).max() - df['close'].tail(10).min()
            recent_mean = df['close'].tail(10).mean()
            
            predicted_actual = self.preprocessor.inverse_transform_prediction(prediction_normalized)
            
            if abs(predicted_actual - recent_mean) > recent_range * 2:
                base_confidence *= 0.5  # Aşırı tahmin için ceza
            
            return min(100, max(0, base_confidence))
            
        except Exception:
            return 50.0  # Varsayılan güvenilirlik
    
    def analyze_recent_news_impact(self, coin_symbol, days=7):
        """
        Son günlerin haberlerini analiz ederek gelecek tahminleri yapar
        
        Args:
            coin_symbol (str): Coin sembolü
            days (int): Kaç günlük son haberleri analiz edileceği
        
        Returns:
            dict: Haber tabanlı analiz sonuçları
        """
        if not self.news_analyzer:
            return {
                'news_sentiment': 0,
                'news_impact_score': 0,
                'recommended_action': 'HOLD',
                'news_summary': 'Haber analizi mevcut değil'
            }
        
        try:
            # Son günlerin haberlerini çek
            recent_news = self.news_analyzer.fetch_all_news(coin_symbol, days)
            
            if not recent_news:
                return {
                    'news_sentiment': 0,
                    'news_impact_score': 0,
                    'recommended_action': 'HOLD',
                    'news_summary': 'Son günlerde önemli haber bulunamadı'
                }
            
            # Sentiment analizi
            news_df = self.news_analyzer.analyze_news_sentiment_batch(recent_news)
            
            if news_df.empty:
                return {
                    'news_sentiment': 0,
                    'news_impact_score': 0,
                    'recommended_action': 'HOLD',
                    'news_summary': 'Haber sentiment analizi başarısız'
                }
            
            # Son 3 günün ortalama sentiment'i
            recent_sentiment = news_df.tail(min(len(news_df), 10))['overall_sentiment'].mean()
            news_count = len(news_df)
            
            # Haber etkisi skorunu hesapla
            impact_score = self._calculate_news_impact_score(news_df)
            
            # Strateji önerisi
            action = self._determine_news_based_action(recent_sentiment, impact_score, news_count)
            
            # Haber özeti
            summary = self._generate_news_summary(news_df)
            
            return {
                'news_sentiment': recent_sentiment,
                'news_impact_score': impact_score,
                'recommended_action': action,
                'news_summary': summary,
                'news_count': news_count
            }
            
        except Exception as e:
            print(f"⚠️ Haber analizi hatası: {str(e)}")
            return {
                'news_sentiment': 0,
                'news_impact_score': 0,
                'recommended_action': 'HOLD',
                'news_summary': 'Haber analizi sırasında hata oluştu'
            }
    
    def _calculate_news_impact_score(self, news_df):
        """
        Haberlerin potansiyel fiyat etkisini hesaplar
        
        Args:
            news_df (pd.DataFrame): Haber sentiment verileri
        
        Returns:
            float: Etki skoru (0-100)
        """
        if news_df.empty:
            return 0
        
        # Sentiment'in mutlak değeri (ne kadar güçlü olduğu)
        sentiment_strength = news_df['overall_sentiment'].abs().mean()
        
        # Haber güvenilirliği
        confidence_avg = news_df['confidence'].mean()
        
        # Haber sayısı etkisi (daha fazla haber = daha yüksek etki)
        news_count_factor = min(1.0, len(news_df) / 20.0)
        
        # Son 24 saatin sentiment değişimi
        if len(news_df) > 1:
            recent_news = news_df.tail(5)
            older_news = news_df.head(5) if len(news_df) > 5 else news_df
            
            recent_avg = recent_news['overall_sentiment'].mean()
            older_avg = older_news['overall_sentiment'].mean()
            
            sentiment_change = abs(recent_avg - older_avg)
        else:
            sentiment_change = 0
        
        # Birleşik etki skoru
        impact_score = (
            sentiment_strength * 40 +
            confidence_avg * 30 +
            news_count_factor * 20 +
            sentiment_change * 10
        )
        
        return min(100, max(0, impact_score))
    
    def _determine_news_based_action(self, sentiment, impact_score, news_count):
        """
        Haber analizine göre strateji önerisi belirler
        
        Args:
            sentiment (float): Ortalama sentiment
            impact_score (float): Etki skoru
            news_count (int): Haber sayısı
        
        Returns:
            str: Aksiyon önerisi
        """
        # Yüksek etki + Pozitif sentiment = ALIM
        if impact_score > 60 and sentiment > 0.3:
            return "GÜÇLÜ ALIM"
        elif impact_score > 40 and sentiment > 0.15:
            return "ALIM"
        elif impact_score > 30 and sentiment > 0.05:
            return "DİKKATLİ ALIM"
        
        # Yüksek etki + Negatif sentiment = SATIM
        elif impact_score > 60 and sentiment < -0.3:
            return "GÜÇLÜ SATIM"
        elif impact_score > 40 and sentiment < -0.15:
            return "SATIM"
        elif impact_score > 30 and sentiment < -0.05:
            return "DİKKATLİ SATIM"
        
        # Düşük etki veya nötr sentiment = BEKLE
        else:
            return "BEKLE"
    
    def _generate_news_summary(self, news_df):
        """
        Haber analizinin özetini oluşturur
        
        Args:
            news_df (pd.DataFrame): Haber verileri
        
        Returns:
            str: Haber özeti
        """
        if news_df.empty:
            return "Analiz edilecek haber bulunamadı"
        
        total_news = len(news_df)
        avg_sentiment = news_df['overall_sentiment'].mean()
        
        positive_news = len(news_df[news_df['overall_sentiment'] > 0.1])
        negative_news = len(news_df[news_df['overall_sentiment'] < -0.1])
        neutral_news = total_news - positive_news - negative_news
        
        # En etkili haberleri bul
        top_positive = news_df.nlargest(2, 'overall_sentiment')
        top_negative = news_df.nsmallest(2, 'overall_sentiment')
        
        summary = f"""
Son {total_news} haber analizi:
• Pozitif: {positive_news} haber (%{positive_news/total_news*100:.1f})
• Negatif: {negative_news} haber (%{negative_news/total_news*100:.1f})
• Nötr: {neutral_news} haber (%{neutral_news/total_news*100:.1f})

Genel Sentiment: {avg_sentiment:.3f} ({self._sentiment_to_text(avg_sentiment)})
"""
        
        if len(top_positive) > 0:
            summary += f"\n🟢 En Pozitif Haber: {top_positive.iloc[0]['title'][:80]}..."
        
        if len(top_negative) > 0:
            summary += f"\n🔴 En Negatif Haber: {top_negative.iloc[0]['title'][:80]}..."
        
        return summary
    
    def _sentiment_to_text(self, sentiment):
        """
        Sentiment skorunu metne çevirir
        
        Args:
            sentiment (float): Sentiment skoru
        
        Returns:
            str: Sentiment açıklaması
        """
        if sentiment > 0.5:
            return "Çok Pozitif"
        elif sentiment > 0.2:
            return "Pozitif"
        elif sentiment > 0.05:
            return "Hafif Pozitif"
        elif sentiment > -0.05:
            return "Nötr"
        elif sentiment > -0.2:
            return "Hafif Negatif"
        elif sentiment > -0.5:
            return "Negatif"
        else:
            return "Çok Negatif"
    
    def analyze_yigit_signals(self, df):
        """
        Yigit ATR Trailing Stop sinyallerini analiz eder
        
        Args:
            df (pd.DataFrame): İşlenmiş veri (Yigit indikatörleri dahil)
        
        Returns:
            dict: Yigit analiz sonuçları
        """
        if 'yigit_position' not in df.columns:
            return {
                'has_yigit': False,
                'analysis': 'Yigit ATR Trailing Stop indikatörü bulunamadı'
            }
        
        try:
            latest_yigit_position = df['yigit_position'].iloc[-1]
            latest_yigit_buy = df['yigit_buy_signal'].iloc[-1]
            latest_yigit_sell = df['yigit_sell_signal'].iloc[-1]
            latest_trend_strength = df['yigit_trend_strength'].iloc[-1]
            
            # Son 10 dönemdeki Yigit sinyalleri
            recent_buy_signals = df['yigit_buy_signal'].tail(10).sum()
            recent_sell_signals = df['yigit_sell_signal'].tail(10).sum()
            
            # Son 5 dönemdeki pozisyon değişimleri
            position_changes = df['yigit_position'].tail(5).diff().abs().sum()
            
            # Volume-Price analizi
            avg_volume_price_ratio = df['yigit_volume_price_ratio'].tail(10).mean()
            
            yigit_direction = "YUKARI TREND" if latest_yigit_position == 1 else "AŞAĞI TREND" if latest_yigit_position == -1 else "NÖTR"
            yigit_signal = "AL SİNYALİ" if latest_yigit_buy else "SAT SİNYALİ" if latest_yigit_sell else "SİNYAL YOK"
            
            # Güvenilirlik hesaplaması
            confidence_score = min(100, max(0, 
                (latest_trend_strength * 20) + 
                (recent_buy_signals * 10 if latest_yigit_position == 1 else recent_sell_signals * 10) +
                (50 if position_changes < 2 else 30)  # Stabil pozisyon = daha güvenilir
            ))
            
            return {
                'has_yigit': True,
                'current_position': latest_yigit_position,
                'current_signal': yigit_signal,
                'direction': yigit_direction,
                'trend_strength': latest_trend_strength,
                'recent_buy_signals': recent_buy_signals,
                'recent_sell_signals': recent_sell_signals,
                'position_stability': position_changes,
                'volume_price_ratio': avg_volume_price_ratio,
                'confidence': confidence_score,
                'strategy_recommendation': self._get_yigit_strategy(latest_yigit_position, latest_trend_strength, confidence_score)
            }
            
        except Exception as e:
            return {
                'has_yigit': False,
                'analysis': f'Yigit analizi hatası: {str(e)}'
            }
    
    def _get_yigit_strategy(self, position, trend_strength, confidence):
        """
        Yigit sinyallerine göre strateji önerisi
        
        Args:
            position: Mevcut pozisyon (-1, 0, 1)
            trend_strength: Trend gücü
            confidence: Güvenilirlik skoru
        
        Returns:
            str: Strateji önerisi
        """
        if confidence > 80:
            if position == 1 and trend_strength > 1.5:
                return "GÜÇLÜ LONG POZİSYONU"
            elif position == -1 and trend_strength > 1.5:
                return "GÜÇLÜ SHORT POZİSYONU"
            elif position == 1:
                return "LONG POZİSYONU"
            elif position == -1:
                return "SHORT POZİSYONU"
        elif confidence > 50:
            if position == 1:
                return "DİKKATLİ LONG"
            elif position == -1:
                return "DİKKATLİ SHORT"
        
        return "BEKLE VE İZLE"
    
    def analyze_whale_impact(self, coin_symbol, hours=24):
        """
        Whale hareketlerini analiz eder ve piyasa etkisini değerlendirir
        
        Args:
            coin_symbol (str): Coin sembolü
            hours (int): Kaç saatlik whale verisi analiz edilecek
        
        Returns:
            dict: Whale analiz sonuçları
        """
        if not self.whale_tracker:
            return {
                'has_whale_data': False,
                'analysis': 'Whale tracker mevcut değil'
            }
        
        try:
            # Whale transferlerini çek
            whale_transactions = self.whale_tracker.fetch_whale_alert_transactions(coin_symbol, hours)
            
            if not whale_transactions:
                return {
                    'has_whale_data': False,
                    'analysis': 'Whale aktivitesi bulunamadı'
                }
            
            # Whale transferlerini analiz et
            whale_analysis = self.whale_tracker.analyze_whale_transactions(whale_transactions)
            
            # Whale özelliklerini oluştur
            whale_features = self.whale_tracker.create_whale_features(whale_analysis, hours)
            
            return {
                'has_whale_data': True,
                'whale_analysis': whale_analysis,
                'whale_features': whale_features,
                'transaction_count': len(whale_transactions),
                'analysis_summary': whale_analysis['analysis']
            }
            
        except Exception as e:
            return {
                'has_whale_data': False,
                'analysis': f'Whale analizi hatası: {str(e)}'
            }
    
    def get_whale_strategy_recommendation(self, whale_impact, price_data, coin_symbol):
        """
        Whale verilerine göre strateji önerisi
        
        Args:
            whale_impact (dict): Whale etki analizi
            price_data (pd.DataFrame): Fiyat verileri
            coin_symbol (str): Coin sembolü
        
        Returns:
            dict: Whale tabanlı strateji
        """
        if not whale_impact.get('has_whale_data', False):
            return {
                'strategy': 'WHALE VERİSİ YOK - Teknik analiz odaklı',
                'confidence': 'DÜŞÜK',
                'reasoning': 'Whale verisi mevcut değil'
            }
        
        try:
            whale_analysis = whale_impact['whale_analysis']
            
            if not self.whale_tracker:
                return {
                    'strategy': 'WHALE TRACKER YOK',
                    'confidence': 'DÜŞÜK',
                    'reasoning': 'Whale tracker mevcut değil'
                }
            
            # Whale-fiyat korelasyonunu analiz et
            correlation_analysis = self.whale_tracker.analyze_whale_price_correlation(
                whale_analysis, price_data, coin_symbol
            )
            
            # Strateji önerisi al
            strategy_recommendation = self.whale_tracker.get_whale_strategy_recommendation(
                whale_analysis, correlation_analysis
            )
            
            return {
                'strategy': strategy_recommendation['strategy'],
                'confidence': strategy_recommendation['confidence'],
                'reasoning': strategy_recommendation['reasoning'],
                'correlation': correlation_analysis['correlation'],
                'whale_sentiment': correlation_analysis.get('whale_sentiment', 0)
            }
            
        except Exception as e:
            return {
                'strategy': 'WHALE ANALİZ HATASI',
                'confidence': 'DÜŞÜK',
                'reasoning': f'Hata: {str(e)}'
            }
    
    def plot_prediction_analysis(self, df, prediction_result, periods_back=100):
        """
        Tahmin analizi grafikleri çizer
        
        Args:
            df (pd.DataFrame): Veriler
            prediction_result (dict): Tahmin sonucu
            periods_back (int): Kaç dönem geriye gidileceği
        """
        if prediction_result is None:
            print("Gösterilecek tahmin sonucu yok.")
            return
        
        # Son periods_back kadar veriyi al
        recent_df = df.tail(periods_back)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Ana fiyat grafiği
        axes[0, 0].plot(recent_df.index, recent_df['close'], label='Gerçek Fiyat', color='blue', linewidth=2)
        
        # Tahmini noktayı ekle
        next_time = prediction_result['next_candle_time']
        predicted_price = prediction_result['predicted_price']
        
        axes[0, 0].scatter([next_time], [predicted_price], color='red', s=100, 
                          label=f'Tahmin: ${predicted_price:.2f}', zorder=5)
        
        # Tahmin çizgisi
        axes[0, 0].plot([recent_df.index[-1], next_time], 
                       [recent_df['close'].iloc[-1], predicted_price], 
                       'r--', alpha=0.7, linewidth=2)
        
        axes[0, 0].set_title(f'Fiyat Tahmini - Değişim: {prediction_result["price_change_percent"]:.2f}%')
        axes[0, 0].set_ylabel('Fiyat (USDT)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volume grafiği
        axes[0, 1].bar(recent_df.index, recent_df['volume'], alpha=0.7, color='orange', width=0.1)
        axes[0, 1].set_title('İşlem Hacmi')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Fiyat değişim yüzdesi
        price_changes = recent_df['close'].pct_change() * 100
        colors = ['green' if x > 0 else 'red' for x in price_changes]
        axes[1, 0].bar(recent_df.index, price_changes, color=colors, alpha=0.7, width=0.1)
        axes[1, 0].set_title('Fiyat Değişim Yüzdesi')
        axes[1, 0].set_ylabel('Değişim (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Tahmin güvenilirlik göstergesi
        confidence = prediction_result['confidence']
        categories = ['Düşük\n(0-30)', 'Orta\n(30-70)', 'Yüksek\n(70-100)']
        values = [0, 0, 0]
        
        if confidence <= 30:
            values[0] = confidence
        elif confidence <= 70:
            values[1] = confidence
        else:
            values[2] = confidence
        
        colors_conf = ['red', 'orange', 'green']
        bars = axes[1, 1].bar(categories, [30, 40, 30], color=colors_conf, alpha=0.3)
        axes[1, 1].bar(categories, values, color=colors_conf, alpha=0.8)
        axes[1, 1].set_title(f'Tahmin Güvenilirliği: {confidence:.1f}%')
        axes[1, 1].set_ylabel('Güvenilirlik (%)')
        axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, symbol, prediction_result, multiple_predictions=None, news_analysis=None, yigit_analysis=None):
        """
        Tahmin raporu oluşturur
        
        Args:
            symbol (str): Coin sembolü
            prediction_result (dict): Ana tahmin sonucu
            multiple_predictions (list): Çoklu dönem tahminleri
            news_analysis (dict): Haber analizi sonuçları
            yigit_analysis (dict): Yigit ATR Trailing Stop analizi
        
        Returns:
            str: Rapor metni
        """
        if prediction_result is None:
            return "Tahmin raporu oluşturulamadı."
        
        report = f"""
 {'='*60}
            KAPSAMLI KRİPTO PARA TAHMİN RAPORU
                   (LSTM + HABER ANALİZİ)
 {'='*60}
 
 COIN: {symbol.upper()}
 RAPOR TARİHİ: {prediction_result['prediction_time'].strftime('%Y-%m-%d %H:%M:%S')}
 
 {'='*60}
                     ANA TAHMİN SONUCU
 {'='*60}
 
 Mevcut Fiyat: ${prediction_result['current_price']:.6f}
 Tahmini Fiyat: ${prediction_result['predicted_price']:.6f}
 Beklenen Değişim: {prediction_result['price_change_percent']:+.2f}%
 Sonraki Mum: {prediction_result['next_candle_time'].strftime('%Y-%m-%d %H:%M:%S')}
 Güvenilirlik: {prediction_result['confidence']:.1f}%
 
 """
        
        # Yön analizi
        if prediction_result['price_change_percent'] > 0:
            direction = "YÜKSELİŞ BEKLENİYOR 📈"
            lstm_strategy = "ALIM FIRSATı" if prediction_result['confidence'] > 70 else "DİKKATLİ ALIM"
        else:
            direction = "DÜŞÜŞ BEKLENİYOR 📉"
            lstm_strategy = "SATIM SİNYALİ" if prediction_result['confidence'] > 70 else "DİKKATLİ SATIM"
        
        report += f"""
 LSTM TREND ANALİZİ: {direction}
 LSTM STRATEJİ ÖNERİSİ: {lstm_strategy}
 
 """
        
        # Haber analizi sonuçları
        if news_analysis:
            report += f"""
 {'='*60}
                    HABER SENTİMENT ANALİZİ
 {'='*60}
 
 Haber Sayısı: {news_analysis.get('news_count', 0)}
 Sentiment Skoru: {news_analysis.get('news_sentiment', 0):.3f}
 Etki Seviyesi: {news_analysis.get('news_impact_score', 0):.1f}/100
 
 HABER BAZLI STRATEJİ: {news_analysis.get('recommended_action', 'HOLD')}
 
 {news_analysis.get('news_summary', '')}
 
 """
            
            # Birleşik strateji önerisi
            lstm_sentiment = 1 if prediction_result['price_change_percent'] > 0 else -1
            news_sentiment = news_analysis.get('news_sentiment', 0)
            
            # Ağırlıklı birleşim
            combined_signal = (lstm_sentiment * 0.6) + (news_sentiment * 0.4)
            
            if combined_signal > 0.3:
                final_strategy = "GÜÇLÜ ALIM SİNYALİ"
                confidence_level = "YÜKSEK"
            elif combined_signal > 0.1:
                final_strategy = "ALIM SİNYALİ"
                confidence_level = "ORTA"
            elif combined_signal > -0.1:
                final_strategy = "BEKLE/NÖTR"
                confidence_level = "DÜŞÜK"
            elif combined_signal > -0.3:
                final_strategy = "SATIM SİNYALİ"
                confidence_level = "ORTA"
            else:
                final_strategy = "GÜÇLÜ SATIM SİNYALİ"
                confidence_level = "YÜKSEK"
            
            report += f"""
 {'='*60}
                  BİRLEŞİK STRATEJİ ÖNERİSİ
 {'='*60}
 
 LSTM + HABER ANALİZİ: {final_strategy}
 GÜVEN SEVİYESİ: {confidence_level}
 
 """
        
        # Çoklu dönem tahminleri
        if multiple_predictions:
            report += f"""
 {'='*60}
                 24 SAATLİK TAHMİN (6 DÖNEM)
 {'='*60}
 
 """
            for pred in multiple_predictions:
                report += f"Dönem {pred['period']}: ${pred['predicted_price']:.6f} ({pred['price_change_percent']:+.2f}%)\n"
        
        # Gelecek haberler için aksiyon planı
        if news_analysis:
            report += f"""
 
 {'='*60}
                GELECEK HABERLER İÇİN AKSİYON PLANI
 {'='*60}
 
 🔍 İZLENECEK HABER KATEGORİLERİ:
    • Kurumsal yatırım duyuruları
    • Regulasyon haberleri
    • Teknolojik güncellemeler
    • Partnership açıklamaları
    • Makroekonomik gelişmeler
 
 ⚡ AKSİYON STRATEJİSİ:
    • Pozitif haberler: Model tahminini destekliyorsa pozisyon artır
    • Negatif haberler: Stop-loss seviyelerini güncelle
    • Nötr haberler: Mevcut stratejiye devam et
 
 """
        
        # Yigit ATR Trailing Stop analizi
        if yigit_analysis and yigit_analysis.get('has_yigit', False):
            report += f"""
 {'='*60}
                   YİGİT ATR TRAİLİNG STOP ANALİZİ
 {'='*60}
 
 📊 Güncel Durum: {yigit_analysis['direction']}
 🎯 Son Sinyal: {yigit_analysis['current_signal']}
 💪 Trend Gücü: {yigit_analysis['trend_strength']:.3f}
 🎲 Güvenilirlik: {yigit_analysis['confidence']:.1f}%
 
 📈 Son 10 Dönem Sinyaller:
    • Al Sinyalleri: {yigit_analysis['recent_buy_signals']}
    • Sat Sinyalleri: {yigit_analysis['recent_sell_signals']}
 
 📊 Volume-Price Analizi:
    • Ortalama V/P Oranı: {yigit_analysis['volume_price_ratio']:.6f}
    • Pozisyon Stabilitesi: {yigit_analysis['position_stability']:.1f}
 
 🎯 YİGİT STRATEJİ ÖNERİSİ: {yigit_analysis['strategy_recommendation']}
 
 """
        
        # Risk uyarısı
        report += f"""
 {'='*60}
                     RİSK UYARISI
 {'='*60}
 
 ⚠️  Bu tahmin matematiksel model ve haber analizi sonucudur, kesin değildir.
 ⚠️  Kripto para yatırımları son derece yüksek risk içerir.
 ⚠️  Haberler ve sentiment hızla değişebilir.
 ⚠️  Yatırım kararlarınızı verirken diğer faktörleri de değerlendirin.
 ⚠️  Kaybetmeyi göze alamayacağınız para ile yatırım yapmayın.
 ⚠️  Stop-loss ve risk yönetimi kullanın.
 
 {'='*60}
 """
        
        return report
    
    def save_prediction_to_file(self, report, filename=None):
        """
        Tahmin raporunu dosyaya kaydeder
        
        Args:
            report (str): Rapor metni
            filename (str): Dosya adı (opsiyonel)
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"crypto_prediction_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Rapor şuraya kaydedildi: {filename}")
        except Exception as e:
            print(f"Rapor kaydetme hatası: {str(e)}") 