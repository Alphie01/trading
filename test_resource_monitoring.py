#!/usr/bin/env python3
"""
Resource Monitoring Test Script

Bu script eğitim sırasında hangi resource'un (GPU/CPU) kullanıldığını test eder.
"""

print("🔧 Resource Monitoring Test başlıyor...")

# TensorFlow configuration import (FIRST!)
from tf_config import print_training_device_info, monitor_training_resources, get_current_device

print("\n🎯 Sistem resource bilgileri:")
print_training_device_info()

# Test data oluştur
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("\n📊 Test verisi oluşturuluyor...")

# Synthetic crypto data
dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='4H')
np.random.seed(42)

data = {
    'open': np.random.uniform(40000, 45000, len(dates)),
    'high': np.random.uniform(45000, 50000, len(dates)),
    'low': np.random.uniform(35000, 40000, len(dates)),
    'close': np.random.uniform(40000, 45000, len(dates)),
    'volume': np.random.uniform(1000, 5000, len(dates))
}

df = pd.DataFrame(data, index=dates)
print(f"✅ Test verisi hazır: {len(df)} data points")

# Test 1: LSTM Model Resource Monitoring
print("\n" + "="*60)
print("🧠 LSTM MODEL RESOURCE TEST")
print("="*60)

try:
    from lstm_model import CryptoLSTMModel
    from data_preprocessor import CryptoDataPreprocessor
    
    # Data preprocessing
    preprocessor = CryptoDataPreprocessor()
    processed_df = preprocessor.prepare_data(df, use_technical_indicators=True)
    
    if processed_df is not None:
        scaled_data = preprocessor.scale_data(processed_df, fit_scaler=True)
        X, y = preprocessor.create_sequences(scaled_data, sequence_length=30)  # Smaller sequence for test
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create and train LSTM model
        lstm_model = CryptoLSTMModel(sequence_length=30, n_features=X.shape[2])
        lstm_model.build_model([32, 32], dropout_rate=0.2, learning_rate=0.001)
        
        print(f"🎯 LSTM eğitimi başlıyor...")
        print(f"📊 Training device: {get_current_device()}")
        
        # Train with resource monitoring
        history = lstm_model.train_model(
            X_train, y_train, X_val, y_val,
            epochs=5,  # Kısa test için 5 epoch
            batch_size=16,
            verbose=True
        )
        
        if history:
            print("✅ LSTM test tamamlandı!")
        else:
            print("❌ LSTM test başarısız")
    else:
        print("❌ Data preprocessing başarısız")
        
except Exception as lstm_error:
    print(f"❌ LSTM test hatası: {lstm_error}")

# Test 2: DQN Model Resource Monitoring
print("\n" + "="*60)
print("🤖 DQN MODEL RESOURCE TEST")
print("="*60)

try:
    from dqn_trading_model import DQNTradingModel
    
    # Create DQN model
    dqn_model = DQNTradingModel(lookback_window=30, initial_balance=10000)
    
    if processed_df is not None:
        dqn_model.prepare_data(processed_df)
        
        print(f"🎯 DQN eğitimi başlıyor...")
        print(f"📊 Training device: {get_current_device()}")
        
        # Train with resource monitoring
        training_results = dqn_model.train(
            processed_df,
            episodes=20,  # Kısa test için 20 episode
            verbose=True
        )
        
        if training_results:
            print("✅ DQN test tamamlandı!")
        else:
            print("❌ DQN test başarısız")
    else:
        print("❌ Data preprocessing başarısız - DQN test atlanıyor")
        
except Exception as dqn_error:
    print(f"❌ DQN test hatası: {dqn_error}")

# Test 3: Hybrid Model Resource Monitoring  
print("\n" + "="*60)
print("🔥 HYBRID MODEL RESOURCE TEST")
print("="*60)

try:
    from hybrid_trading_model import HybridTradingModel
    
    # Create hybrid model
    hybrid_model = HybridTradingModel(sequence_length=30, initial_balance=10000)
    
    if processed_df is not None:
        print(f"🎯 Hybrid eğitimi başlıyor...")
        print(f"📊 Training device: {get_current_device()}")
        
        # Train with comprehensive resource monitoring
        success = hybrid_model.train_hybrid_model(
            processed_df,
            lstm_epochs=3,   # Çok kısa test için
            dqn_episodes=10, # Çok kısa test için  
            verbose=True
        )
        
        if success:
            print("✅ Hybrid test tamamlandı!")
            
            # Test prediction to verify model works
            prediction = hybrid_model.predict_hybrid_action(processed_df.tail(60))
            if prediction['success']:
                print(f"🎯 Test prediction confidence: {prediction['confidence']:.1%}")
            else:
                print("⚠️ Test prediction başarısız")
        else:
            print("❌ Hybrid test başarısız")
    else:
        print("❌ Data preprocessing başarısız - Hybrid test atlanıyor")
        
except Exception as hybrid_error:
    print(f"❌ Hybrid test hatası: {hybrid_error}")

# Final Summary
print("\n" + "🎯" + "="*58 + "🎯")
print("🏁 RESOURCE MONITORING TEST SUMMARY")
print("🎯" + "="*58 + "🎯")

print("\n📊 Final Resource State:")
monitor_training_resources()

final_device = get_current_device()
print(f"\n🎯 All tests completed on: {final_device}")

print(f"\n💡 Resource Monitoring Features:")
print(f"   ✅ System information detection")
print(f"   ✅ TensorFlow device configuration")
print(f"   ✅ Real-time CPU/RAM monitoring")
print(f"   ✅ Training device identification")
print(f"   ✅ Pre/post training resource tracking")

print(f"\n🎯 Usage in your training:")
print(f"   • LSTM: Kullan 'verbose=True' parametresini")
print(f"   • DQN: Kullan 'verbose=True' parametresini") 
print(f"   • Hybrid: Kullan 'verbose=True' parametresini")
print(f"   • Manuel: 'from tf_config import monitor_training_resources'")

print("\n🎯" + "="*58 + "🎯")
print("✅ Resource monitoring test tamamlandı!") 