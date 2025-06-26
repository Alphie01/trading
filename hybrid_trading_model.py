#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid LSTM+DQN Trading Model

Bu modül LSTM fiyat tahmin modeli ile DQN action selection modelini birleştirerek
daha güçlü ve adaptif trading stratejileri oluşturur.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import json
import pickle
import os

from lstm_model import CryptoLSTMModel
from dqn_trading_model import DQNTradingModel, TradingEnvironment
from data_preprocessor import CryptoDataPreprocessor

# **CRITICAL: Use centralized TensorFlow configuration**
from tf_config import get_tensorflow, is_tensorflow_available

# Optional database integration
try:
    from database import TradingDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    print("⚠️ Database module not available - Hybrid predictions will not be saved")
    DATABASE_AVAILABLE = False

tf = get_tensorflow()
TF_AVAILABLE = is_tensorflow_available()

print(f"🔗 Hybrid Model - TensorFlow Available: {TF_AVAILABLE}")

class HybridTradingModel:
    """
    Hybrid model combining LSTM price prediction with DQN action selection
    """
    
    def __init__(self, sequence_length=60, initial_balance=10000):
        """
        Initialize Hybrid Trading Model
        
        Args:
            sequence_length (int): LSTM sequence length
            initial_balance (float): Initial trading balance
        """
        self.sequence_length = sequence_length
        self.initial_balance = initial_balance
        
        # Model components
        self.lstm_model = None
        self.dqn_model = None
        self.preprocessor = CryptoDataPreprocessor()
        
        # Model states
        self.lstm_trained = False
        self.dqn_trained = False
        self.hybrid_trained = False
        
        # Performance tracking
        self.performance_history = []
        self.ensemble_weights = {
            'lstm': 0.35,
            'dqn': 0.45,     # Increased DQN weight
            'technical': 0.2
        }
        
    def prepare_models(self, df):
        """Prepare both LSTM and DQN models"""
        print("🔧 Preparing hybrid model components...")
        
        # Initialize LSTM model
        self.lstm_model = CryptoLSTMModel(self.sequence_length, df.shape[1])
        
        # Initialize DQN model with overfitting prevention
        self.dqn_model = DQNTradingModel(
            lookback_window=self.sequence_length,
            initial_balance=self.initial_balance
        )
        
        # Prepare data for both models
        self.dqn_model.prepare_data(df)
        
        print("✅ Hybrid model components prepared")
        
    def train_hybrid_model(self, df, lstm_epochs=30, dqn_episodes=100, verbose=True):
        """
        Train both LSTM and DQN components with comprehensive resource monitoring
        
        Args:
            df (pd.DataFrame): Training data
            lstm_epochs (int): LSTM training epochs
            dqn_episodes (int): DQN training episodes
            verbose (bool): Verbose output
        """
        if verbose:
            print("🚀 Starting hybrid model training...")
            
            # **NEW: Comprehensive resource monitoring for hybrid training**
            try:
                from tf_config import print_training_device_info, monitor_training_resources, get_current_device
                print("\n" + "🎯" + "="*58 + "🎯")
                print("🔥 HYBRID AI TRAINING - RESOURCE ALLOCATION")
                print("🎯" + "="*58 + "🎯")
                
                print_training_device_info()
                
                print(f"\n🧠 Hybrid Training Configuration:")
                print(f"   📊 Data points: {len(df)}")
                print(f"   🧠 LSTM epochs: {lstm_epochs}")
                print(f"   🤖 DQN episodes: {dqn_episodes}")
                print(f"   ⏰ Sequence length: {self.sequence_length}")
                
            except ImportError:
                print("⚠️ Resource monitoring not available")
        
        # Prepare models if not done
        if self.lstm_model is None or self.dqn_model is None:
            self.prepare_models(df)
        
        # Step 1: Train LSTM for price prediction
        if verbose:
            print("\n" + "="*60)
            print("1️⃣ LSTM PRICE PREDICTOR TRAINING")
            print("="*60)
        
        try:
            # Monitor resources before LSTM training
            if verbose:
                try:
                    current_device = get_current_device()
                    print(f"🎯 LSTM Training Device: {current_device}")
                    print("\n📊 Pre-LSTM Resource State:")
                    monitor_training_resources()
                except:
                    pass
            
            # Process data for LSTM
            processed_df = self.preprocessor.prepare_data(df, use_technical_indicators=True)
            scaled_data = self.preprocessor.scale_data(processed_df, fit_scaler=True)
            X, y = self.preprocessor.create_sequences(scaled_data, self.sequence_length)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build and train LSTM
            feature_count = X.shape[2]
            self.lstm_model = CryptoLSTMModel(self.sequence_length, feature_count)
            self.lstm_model.build_model([50, 50, 50], 0.2, 0.001)
            
            # **YENİ: İlk eğitim kontrolü - early stopping'i engelle**
            hybrid_cache_file = f"model_cache/hybrid_model.h5"  # Generic hybrid cache
            is_first_training = not os.path.exists(hybrid_cache_file)
            
            if verbose and is_first_training:
                print("🆕 İlk Hybrid eğitimi - Early stopping devre dışı")
            elif verbose:
                print("🔄 Hybrid re-training - Normal early stopping")
            
            history = self.lstm_model.train_model(
                X_train, y_train, X_val, y_val,
                epochs=lstm_epochs, 
                batch_size=32, 
                verbose=verbose,
                use_early_stopping=not is_first_training  # İlk eğitimde early stopping kapalı
            )
            
            self.lstm_trained = True
            if verbose:
                print("✅ LSTM training completed")
                
                # **NEW: Post-LSTM resource monitoring**
                try:
                    print("\n📊 Post-LSTM Resource State:")
                    monitor_training_resources()
                except:
                    pass
                
        except Exception as e:
            print(f"❌ LSTM training error: {e}")
            return False
        
        # Step 2: Enhanced DQN training with LSTM predictions
        if verbose:
            print("\n" + "="*60)
            print("2️⃣ DQN ACTION SELECTOR TRAINING (LSTM-Enhanced)")
            print("="*60)
        
        try:
            # Monitor resources before DQN training
            if verbose:
                try:
                    current_device = get_current_device()
                    print(f"🎯 DQN Training Device: {current_device}")
                    print("\n📊 Pre-DQN Resource State:")
                    monitor_training_resources()
                except:
                    pass
            
            # Train DQN with LSTM-enhanced environment
            self._train_lstm_enhanced_dqn(df, dqn_episodes, verbose)
            self.dqn_trained = True
            if verbose:
                print("✅ DQN training completed")
                
                # **NEW: Post-DQN resource monitoring**
                try:
                    print("\n📊 Post-DQN Resource State:")
                    monitor_training_resources()
                except:
                    pass
                
        except Exception as e:
            print(f"❌ DQN training error: {e}")
            return False
        
        # Step 3: Ensemble weight optimization
        if verbose:
            print("\n" + "="*60)
            print("3️⃣ ENSEMBLE WEIGHT OPTIMIZATION")
            print("="*60)
        
        self._optimize_ensemble_weights(df)
        self.hybrid_trained = True
        
        if verbose:
            print("🎉 Hybrid model training completed!")
            
            # **NEW: Final hybrid training summary**
            try:
                print("\n" + "🎯" + "="*58 + "🎯")
                print("🏁 HYBRID TRAINING COMPLETED - FINAL SUMMARY")
                print("🎯" + "="*58 + "🎯")
                
                print("\n📊 Final Resource Usage:")
                monitor_training_resources()
                
                final_device = get_current_device()
                print(f"\n🎯 All training completed on: {final_device}")
                
                print(f"\n🏆 Hybrid Model Status:")
                print(f"   🧠 LSTM trained: {'✅' if self.lstm_trained else '❌'}")
                print(f"   🤖 DQN trained: {'✅' if self.dqn_trained else '❌'}")
                print(f"   🔗 Hybrid ready: {'✅' if self.hybrid_trained else '❌'}")
                print(f"   ⚖️ Ensemble weights: {self.ensemble_weights}")
                
                print("\n🎯" + "="*58 + "🎯")
                
            except ImportError:
                print("⚠️ Final resource monitoring not available")
            
        return True
    
    def _train_lstm_enhanced_dqn(self, df, episodes, verbose):
        """Train DQN with LSTM predictions as additional features"""
        
        # Create enhanced environment that includes LSTM predictions
        enhanced_env = self._create_lstm_enhanced_environment(df)
        
        # Train DQN with enhanced state space
        training_rewards = []
        training_portfolio_values = []
        
        for episode in range(episodes):
            state = enhanced_env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Get LSTM prediction for current state
                lstm_features = self._get_lstm_features(enhanced_env.current_step, df)
                
                # Combine DQN state with LSTM features
                enhanced_state = np.concatenate([state[:-3], lstm_features])  # Replace placeholder LSTM features
                
                if self.dqn_model and self.dqn_model.agent is not None:
                    action = self.dqn_model.agent.act(enhanced_state, training=True)
                else:
                    action = 0  # Default to HOLD if agent not available
                    
                next_state, reward, done, info = enhanced_env.step(action)
                
                # Enhanced reward with LSTM confidence
                lstm_confidence = lstm_features[1]  # LSTM confidence
                enhanced_reward = reward * (1 + lstm_confidence * 0.1)  # Boost reward if LSTM is confident
                
                if self.dqn_model and self.dqn_model.agent is not None:
                    self.dqn_model.agent.remember(enhanced_state, action, enhanced_reward, next_state, done)
                    
                state = next_state
                total_reward += enhanced_reward
                steps += 1
                
                if done:
                    break
            
            # Train the agent
            if (self.dqn_model and self.dqn_model.agent is not None and 
                hasattr(self.dqn_model.agent, 'memory') and 
                hasattr(self.dqn_model.agent, 'batch_size') and
                len(self.dqn_model.agent.memory) >= self.dqn_model.agent.batch_size):
                self.dqn_model.agent.replay()
            
            # Update target network
            if episode % 10 == 0:
                self.dqn_model.agent.update_target_network()
            
            # Record metrics
            final_portfolio_value = enhanced_env.total_portfolio_value
            training_rewards.append(total_reward)
            training_portfolio_values.append(final_portfolio_value)
            
            if verbose and episode % 20 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward:.4f}, "
                      f"Portfolio Value: ${final_portfolio_value:.2f}")
        
        # Update DQN training history
        self.dqn_model.training_history = {
            'rewards': training_rewards,
            'portfolio_values': training_portfolio_values,
            'episodes': episodes
        }
    
    def _create_lstm_enhanced_environment(self, df):
        """Create trading environment enhanced with LSTM predictions"""
        return TradingEnvironment(
            data=df,
            initial_balance=self.initial_balance,
            lookback_window=self.sequence_length
        )
    
    def _get_lstm_features(self, current_step, df):
        """
        Extract LSTM features for DQN state enhancement
        
        Args:
            current_step (int): Current timestep
            df (pd.DataFrame): Market data
            
        Returns:
            np.array: LSTM features [price_change_pred, confidence, direction]
        """
        try:
            if current_step < self.sequence_length:
                return np.array([0.0, 0.0, 0.0])
            
            recent_data = df.iloc[current_step-self.sequence_length:current_step]
            
            # **CRITICAL FIX: Handle None return from prepare_data**
            processed_data = self.preprocessor.prepare_data(recent_data, use_technical_indicators=True)
            
            # If prepare_data returns None (insufficient data), return neutral features
            if processed_data is None:
                return np.array([0.0, 0.0, 0.0])
            
            scaled_data = self.preprocessor.scale_data(processed_data, fit_scaler=False)
            X, _ = self.preprocessor.create_sequences(scaled_data, self.sequence_length)
            
            if len(X) == 0:
                return np.array([0.0, 0.0, 0.0])
            
            # Get LSTM prediction
            prediction = self.lstm_model.predict(X[-1:])
            current_price = df.iloc[current_step-1]['close']
            
            if prediction is not None and len(prediction) > 0:
                predicted_price = prediction[0][0]
                price_change_pct = ((predicted_price - current_price) / current_price) * 100
                
                # Calculate confidence based on recent model performance
                confidence = min(1.0, abs(price_change_pct) / 10.0)  # Higher confidence for larger predictions
                
                # Direction: 1 for up, -1 for down, 0 for neutral
                direction = 1.0 if price_change_pct > 0.5 else -1.0 if price_change_pct < -0.5 else 0.0
                
                return np.array([
                    price_change_pct / 100.0,  # Normalized price change prediction
                    confidence,  # LSTM confidence
                    direction  # Predicted direction
                ])
            
        except Exception as e:
            # **CRITICAL FIX: Remove log spam - only print once per session**
            if not hasattr(self, '_lstm_feature_error_logged'):
                print(f"⚠️ LSTM features unavailable (insufficient data), using neutral values")
                self._lstm_feature_error_logged = True
        
        return np.array([0.0, 0.0, 0.0])
    
    def _optimize_ensemble_weights(self, df):
        """Optimize ensemble weights based on historical performance"""
        
        # Test different weight combinations with better DQN representation
        weight_combinations = [
            {'lstm': 0.35, 'dqn': 0.45, 'technical': 0.2},  # Baseline - DQN focused
            {'lstm': 0.4, 'dqn': 0.5, 'technical': 0.1},    # Strong DQN
            {'lstm': 0.3, 'dqn': 0.55, 'technical': 0.15},  # Very strong DQN
            {'lstm': 0.45, 'dqn': 0.4, 'technical': 0.15},  # LSTM focused
            {'lstm': 0.5, 'dqn': 0.35, 'technical': 0.15},  # Strong LSTM
            {'lstm': 0.25, 'dqn': 0.6, 'technical': 0.15},  # Maximum DQN weight
        ]
        
        best_weights = None
        best_score = float('-inf')
        
        # Test each weight combination on recent data
        test_data = df.tail(100)  # Use last 100 data points for testing
        
        for weights in weight_combinations:
            score = self._evaluate_ensemble_performance(test_data, weights)
            if score > best_score:
                best_score = score
                best_weights = weights
        
        if best_weights:
            self.ensemble_weights = best_weights
            print(f"🎯 Optimized ensemble weights: {best_weights}")
        else:
            print("⚠️ Using default ensemble weights")
    
    def _evaluate_ensemble_performance(self, test_data, weights):
        """Evaluate ensemble performance with given weights"""
        try:
            total_score = 0
            predictions = 0
            
            for i in range(self.sequence_length, len(test_data) - 1):
                # Get predictions from each component
                lstm_pred = self._get_lstm_prediction_score(test_data, i)
                dqn_pred = self._get_dqn_prediction_score(test_data, i)
                tech_pred = self._get_technical_prediction_score(test_data, i)
                
                # Ensemble prediction
                ensemble_pred = (
                    weights['lstm'] * lstm_pred +
                    weights['dqn'] * dqn_pred +
                    weights['technical'] * tech_pred
                )
                
                # Actual outcome
                current_price = test_data.iloc[i]['close']
                future_price = test_data.iloc[i + 1]['close']
                actual_change = (future_price - current_price) / current_price
                
                # Score based on prediction accuracy
                if (ensemble_pred > 0 and actual_change > 0) or (ensemble_pred < 0 and actual_change < 0):
                    total_score += abs(ensemble_pred) * abs(actual_change)
                else:
                    total_score -= abs(ensemble_pred) * abs(actual_change)
                
                predictions += 1
            
            return total_score / predictions if predictions > 0 else 0
            
        except Exception as e:
            print(f"⚠️ Error evaluating ensemble: {e}")
            return 0
    
    def _get_lstm_prediction_score(self, data, index):
        """Get LSTM prediction score for given index"""
        try:
            lstm_features = self._get_lstm_features(index, data)
            return lstm_features[0]  # Normalized price change prediction
        except:
            return 0.0
    
    def _get_dqn_prediction_score(self, data, index):
        """Get DQN prediction score for given index"""
        try:
            if self.dqn_model and self.dqn_model.environment:
                self.dqn_model.environment.current_step = index
                state = self.dqn_model.environment._get_state()
                action = self.dqn_model.agent.act(state, training=False)
                
                # Convert action to directional score
                if action == 0:  # HOLD
                    return 0.0
                elif action in [1, 2, 3, 4]:  # BUY actions
                    return 0.25 * action  # 0.25, 0.5, 0.75, 1.0
                else:  # SELL actions
                    return -0.25 * (action - 4)  # -0.25, -0.5, -0.75, -1.0
        except:
            return 0.0
    
    def _get_technical_prediction_score(self, data, index):
        """Get technical analysis prediction score"""
        try:
            current_data = data.iloc[index]
            
            # Simple technical score based on RSI and MACD
            rsi = current_data.get('rsi', 50)
            macd = current_data.get('macd', 0)
            macd_signal = current_data.get('macd_signal', 0)
            
            # RSI score
            rsi_score = 0
            if rsi < 30:
                rsi_score = 0.5  # Oversold, bullish
            elif rsi > 70:
                rsi_score = -0.5  # Overbought, bearish
            
            # MACD score
            macd_score = 0.3 if macd > macd_signal else -0.3
            
            return rsi_score + macd_score
            
        except:
            return 0.0
    
    def predict_hybrid_action(self, current_data, coin_symbol=None, save_to_db=True):
        """
        Generate hybrid prediction combining LSTM, DQN, and technical analysis
        
        Args:
            current_data (pd.DataFrame): Current market data
            coin_symbol (str): Coin symbol for database logging
            save_to_db (bool): Whether to save results to database
            
        Returns:
            dict: Comprehensive prediction with all model outputs
        """
        if not self.hybrid_trained:
            return {
                'success': False,
                'error': 'Hybrid model not trained',
                'lstm_prediction': {},
                'dqn_prediction': {},
                'ensemble_prediction': {},
                'confidence': 0.0
            }
        
        try:
            # Get LSTM prediction
            lstm_prediction = self._get_lstm_price_prediction(current_data)
            
            # Get DQN action prediction
            dqn_prediction = self._get_dqn_action_prediction(current_data)
            
            # Get technical analysis
            technical_analysis = self._get_technical_analysis(current_data)
            
            # Generate ensemble prediction
            ensemble_prediction = self._generate_ensemble_prediction(
                lstm_prediction, dqn_prediction, technical_analysis
            )
            
            # Prepare hybrid result
            hybrid_result = {
                'success': True,
                'lstm_prediction': lstm_prediction,
                'dqn_prediction': dqn_prediction,
                'technical_analysis': technical_analysis,
                'ensemble_prediction': ensemble_prediction,
                'confidence': ensemble_prediction.get('confidence', 0.0),
                'timestamp': datetime.now().isoformat(),
                'model_type': 'Hybrid_LSTM_DQN'
            }
            
            # **DATABASE INTEGRATION**: Save hybrid analysis to database
            if save_to_db and DATABASE_AVAILABLE and coin_symbol and hybrid_result['success']:
                try:
                    db = TradingDatabase()
                    analysis_id = db.save_hybrid_analysis(coin_symbol, hybrid_result)
                    if analysis_id:
                        hybrid_result['database_id'] = analysis_id
                        print(f"📊 Hybrid analysis saved to database (ID: {analysis_id})")
                except Exception as db_error:
                    print(f"⚠️ Database save failed: {db_error}")
                    # Continue without database - don't break the prediction
            
            return hybrid_result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'lstm_prediction': {},
                'dqn_prediction': {},
                'ensemble_prediction': {},
                'confidence': 0.0
            }
    
    def _get_lstm_price_prediction(self, current_data):
        """Get LSTM price prediction"""
        try:
            if not self.lstm_trained or self.lstm_model is None:
                return {'error': 'LSTM not trained'}
            
            # **CRITICAL FIX: Handle None return from prepare_data**
            processed_data = self.preprocessor.prepare_data(current_data, use_technical_indicators=True)
            
            # If prepare_data returns None (insufficient data), return graceful error
            if processed_data is None:
                return {'error': 'Insufficient data for LSTM (need 50+ data points)'}
            
            scaled_data = self.preprocessor.scale_data(processed_data, fit_scaler=False)
            X, _ = self.preprocessor.create_sequences(scaled_data, self.sequence_length)
            
            if len(X) == 0:
                return {'error': 'Insufficient data for LSTM prediction'}
            
            # Get prediction
            prediction = self.lstm_model.predict(X[-1:])
            current_price = current_data['close'].iloc[-1]
            
            if prediction is not None and len(prediction) > 0:
                predicted_price = prediction[0][0]
                price_change = predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100
                
                return {
                    'current_price': float(current_price),
                    'predicted_price': float(predicted_price),
                    'price_change': float(price_change),
                    'price_change_pct': float(price_change_pct),
                    'confidence': min(1.0, abs(price_change_pct) / 10.0),
                    'direction': 'UP' if price_change_pct > 0 else 'DOWN',
                    'model': 'LSTM'
                }
            
        except Exception as e:
            return {'error': f'LSTM prediction error: {e}'}
        
        return {'error': 'LSTM prediction failed'}
    
    def _get_dqn_action_prediction(self, current_data):
        """Get DQN action prediction"""
        try:
            if not self.dqn_trained or self.dqn_model is None:
                return {'error': 'DQN not trained'}
            
            # Prepare state for DQN
            if self.dqn_model.environment:
                self.dqn_model.environment.current_step = len(current_data) - 1
                state = self.dqn_model.environment._get_state()
                
                # Get LSTM features for enhanced state
                lstm_features = self._get_lstm_features(len(current_data) - 1, current_data)
                enhanced_state = np.concatenate([state[:-3], lstm_features])
                
                return self.dqn_model.predict_action(enhanced_state)
            
        except Exception as e:
            return {'error': f'DQN prediction error: {e}'}
        
        return {'error': 'DQN prediction failed'}
    
    def _get_technical_analysis(self, current_data):
        """Get technical analysis summary"""
        try:
            latest_data = current_data.iloc[-1]
            
            # RSI analysis
            rsi = latest_data.get('rsi', 50)
            rsi_signal = 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL'
            
            # MACD analysis
            macd = latest_data.get('macd', 0)
            macd_signal = latest_data.get('macd_signal', 0)
            macd_trend = 'BULLISH' if macd > macd_signal else 'BEARISH'
            
            # Moving averages
            price = latest_data['close']
            sma_7 = latest_data.get('sma_7', price)
            sma_25 = latest_data.get('sma_25', price)
            ma_trend = 'UPTREND' if sma_7 > sma_25 else 'DOWNTREND'
            
            # Overall technical score
            tech_score = 0
            if rsi < 30: tech_score += 1
            elif rsi > 70: tech_score -= 1
            if macd > macd_signal: tech_score += 1
            else: tech_score -= 1
            if sma_7 > sma_25: tech_score += 1
            else: tech_score -= 1
            
            return {
                'rsi': {'value': rsi, 'signal': rsi_signal},
                'macd': {'trend': macd_trend, 'bullish': macd > macd_signal},
                'moving_averages': {'trend': ma_trend, 'bullish': sma_7 > sma_25},
                'overall_score': tech_score,
                'overall_signal': 'BULLISH' if tech_score > 0 else 'BEARISH' if tech_score < 0 else 'NEUTRAL'
            }
            
        except Exception as e:
            return {'error': f'Technical analysis error: {e}'}
    
    def _generate_ensemble_prediction(self, lstm_pred, dqn_pred, tech_analysis):
        """Generate ensemble prediction from all models"""
        try:
            # Extract signals
            lstm_signal = 0
            if 'price_change_pct' in lstm_pred:
                lstm_signal = lstm_pred['price_change_pct'] / 100.0  # Normalize
            
            dqn_signal = 0
            if 'action' in dqn_pred and 'action_name' in dqn_pred:
                action = dqn_pred['action']
                if action == 0:  # HOLD
                    dqn_signal = 0
                elif action in [1, 2, 3, 4]:  # BUY
                    dqn_signal = 0.25 * action
                else:  # SELL
                    dqn_signal = -0.25 * (action - 4)
            
            tech_signal = 0
            if 'overall_score' in tech_analysis:
                tech_signal = tech_analysis['overall_score'] / 3.0  # Normalize
            
            # Weighted ensemble
            weights = self.ensemble_weights
            ensemble_signal = (
                weights['lstm'] * lstm_signal +
                weights['dqn'] * dqn_signal +
                weights['technical'] * tech_signal
            )
            
            # Generate final recommendation
            if ensemble_signal > 0.2:
                recommendation = 'STRONG_BUY'
                confidence = min(1.0, ensemble_signal * 2)
            elif ensemble_signal > 0.05:
                recommendation = 'BUY'
                confidence = min(1.0, ensemble_signal * 4)
            elif ensemble_signal < -0.2:
                recommendation = 'STRONG_SELL'
                confidence = min(1.0, abs(ensemble_signal) * 2)
            elif ensemble_signal < -0.05:
                recommendation = 'SELL'
                confidence = min(1.0, abs(ensemble_signal) * 4)
            else:
                recommendation = 'HOLD'
                confidence = 0.5
            
            return {
                'ensemble_signal': float(ensemble_signal),
                'recommendation': recommendation,
                'confidence': float(confidence),
                'model_weights': weights,
                'component_signals': {
                    'lstm': float(lstm_signal),
                    'dqn': float(dqn_signal),
                    'technical': float(tech_signal)
                },
                'reasoning': self._generate_ensemble_reasoning(
                    ensemble_signal, recommendation, lstm_pred, dqn_pred, tech_analysis
                )
            }
            
        except Exception as e:
            return {'error': f'Ensemble generation error: {e}'}
    
    def _generate_ensemble_reasoning(self, signal, recommendation, lstm_pred, dqn_pred, tech_analysis):
        """Generate human-readable reasoning for ensemble decision"""
        reasoning = f"Hybrid Analysis: {recommendation}\n"
        reasoning += f"Ensemble Signal: {signal:.3f}\n\n"
        
        # LSTM reasoning
        if 'price_change_pct' in lstm_pred:
            reasoning += f"📈 LSTM Prediction: {lstm_pred['price_change_pct']:+.2f}% price change\n"
            reasoning += f"   Confidence: {lstm_pred.get('confidence', 0):.2f}\n"
        
        # DQN reasoning
        if 'action_name' in dqn_pred:
            reasoning += f"🤖 DQN Action: {dqn_pred['action_name']}\n"
            reasoning += f"   Confidence: {dqn_pred.get('confidence', 0):.2f}\n"
        
        # Technical reasoning
        if 'overall_signal' in tech_analysis:
            reasoning += f"📊 Technical: {tech_analysis['overall_signal']}\n"
            reasoning += f"   RSI: {tech_analysis.get('rsi', {}).get('signal', 'N/A')}\n"
            reasoning += f"   MACD: {tech_analysis.get('macd', {}).get('trend', 'N/A')}\n"
        
        # Model weights
        reasoning += f"\n⚖️ Model Weights: LSTM:{self.ensemble_weights['lstm']:.1%}, "
        reasoning += f"DQN:{self.ensemble_weights['dqn']:.1%}, "
        reasoning += f"Technical:{self.ensemble_weights['technical']:.1%}"
        
        return reasoning
    
    def get_model_performance_summary(self):
        """Get performance summary for all models"""
        summary = {
            'hybrid_trained': self.hybrid_trained,
            'lstm_trained': self.lstm_trained,
            'dqn_trained': self.dqn_trained,
            'ensemble_weights': self.ensemble_weights
        }
        
        # LSTM performance
        if self.lstm_model and hasattr(self.lstm_model, 'training_history'):
            lstm_history = self.lstm_model.training_history
            if lstm_history:
                summary['lstm_performance'] = {
                    'final_loss': lstm_history.history['loss'][-1] if 'loss' in lstm_history.history else 0,
                    'final_val_loss': lstm_history.history['val_loss'][-1] if 'val_loss' in lstm_history.history else 0,
                    'epochs_trained': len(lstm_history.history['loss']) if 'loss' in lstm_history.history else 0
                }
        
        # DQN performance
        if self.dqn_model and self.dqn_model.training_history:
            dqn_summary = self.dqn_model.get_training_summary()
            summary['dqn_performance'] = dqn_summary
        
        return summary
    
    def save_hybrid_model(self, base_filepath):
        """Save all components of the hybrid model"""
        try:
            # Save LSTM model
            if self.lstm_model:
                lstm_path = base_filepath.replace('.h5', '_lstm.h5')
                self.lstm_model.save_model(lstm_path)
            
            # Save DQN model
            if self.dqn_model:
                dqn_path = base_filepath.replace('.h5', '_dqn.h5')
                self.dqn_model.save_model(dqn_path)
            
            # Save hybrid metadata
            metadata = {
                'sequence_length': self.sequence_length,
                'initial_balance': self.initial_balance,
                'lstm_trained': self.lstm_trained,
                'dqn_trained': self.dqn_trained,
                'hybrid_trained': self.hybrid_trained,
                'ensemble_weights': self.ensemble_weights,
                'performance_history': self.performance_history
            }
            
            metadata_path = base_filepath.replace('.h5', '_hybrid_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Hybrid model saved to {base_filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving hybrid model: {e}")
            return False
    
    def load_hybrid_model(self, base_filepath):
        """Load all components of the hybrid model"""
        try:
            # Load metadata
            metadata_path = base_filepath.replace('.h5', '_hybrid_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.sequence_length = metadata.get('sequence_length', self.sequence_length)
                self.initial_balance = metadata.get('initial_balance', self.initial_balance)
                self.lstm_trained = metadata.get('lstm_trained', False)
                self.dqn_trained = metadata.get('dqn_trained', False)
                self.hybrid_trained = metadata.get('hybrid_trained', False)
                self.ensemble_weights = metadata.get('ensemble_weights', self.ensemble_weights)
                self.performance_history = metadata.get('performance_history', [])
            
            # Load LSTM model
            lstm_path = base_filepath.replace('.h5', '_lstm.h5')
            if os.path.exists(lstm_path):
                self.lstm_model = CryptoLSTMModel(self.sequence_length, 20)  # Default feature count
                self.lstm_model.load_model(lstm_path)
            
            # Load DQN model
            dqn_path = base_filepath.replace('.h5', '_dqn.h5')
            if os.path.exists(dqn_path):
                self.dqn_model = DQNTradingModel(self.sequence_length, self.initial_balance)
                self.dqn_model.load_model(dqn_path)
            
            print(f"✅ Hybrid model loaded from {base_filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading hybrid model: {e}")
            return False 