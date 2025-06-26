#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Analysis Test - DQN, Hybrid ve Ensemble Sonuçlarını Test Etme

Bu script yeni database entegrasyonunu test eder ve analiz sonuçlarını
analysis_results tablosuna kaydeder.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

try:
    from database import TradingDatabase
    from dqn_trading_model import DQNTradingModel
    from hybrid_trading_model import HybridTradingModel
    from data_fetcher import CryptoDataFetcher
    from data_preprocessor import CryptoDataPreprocessor
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def generate_test_data(days=100):
    """Generate synthetic test data for testing"""
    print(f"📊 Generating {days} days of synthetic test data...")
    
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic crypto price data
    base_price = 50000  # Starting price
    prices = [base_price]
    
    for i in range(1, days):
        # Random walk with slight upward bias
        change = np.random.normal(0.002, 0.03)  # 0.2% average daily growth, 3% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Minimum price of $1000
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(1000000, 10000000) for _ in range(days)]
    })
    
    print(f"✅ Generated data from ${df['close'].iloc[0]:.2f} to ${df['close'].iloc[-1]:.2f}")
    return df

def test_dqn_analysis(df, coin_symbol="TESTCOIN"):
    """Test DQN analysis with database saving"""
    print("\n🤖 Testing DQN Analysis...")
    
    try:
        # Initialize DQN model
        dqn_model = DQNTradingModel(lookback_window=20, initial_balance=10000)
        
        # Quick training with few episodes for testing
        print("🔄 Training DQN model (quick test)...")
        dqn_model.prepare_data(df)
        training_result = dqn_model.train(df, episodes=20, verbose=False)
        
        if training_result:
            print(f"✅ DQN training completed: {len(training_result['rewards'])} episodes")
            
            # Test prediction with database saving
            if dqn_model.environment:
                current_state = dqn_model.environment._get_state()
                
                # Make prediction with database saving
                prediction = dqn_model.predict_action(
                    current_state, 
                    coin_symbol=coin_symbol, 
                    save_to_db=True
                )
                
                print(f"🎯 DQN Prediction: {prediction['action_name']} (Confidence: {prediction['confidence']:.1%})")
                
                if 'database_id' in prediction:
                    print(f"💾 Saved to database with ID: {prediction['database_id']}")
                
                return prediction
        else:
            print("❌ DQN training failed")
            return None
            
    except Exception as e:
        print(f"❌ DQN analysis error: {e}")
        return None

def test_hybrid_analysis(df, coin_symbol="TESTCOIN"):
    """Test Hybrid analysis with database saving"""
    print("\n🔬 Testing Hybrid Analysis...")
    
    try:
        # Initialize hybrid model
        hybrid_model = HybridTradingModel(sequence_length=20, initial_balance=10000)
        
        # Quick training
        print("🔄 Training Hybrid model (quick test)...")
        training_result = hybrid_model.train_hybrid_model(
            df, 
            lstm_epochs=5, 
            dqn_episodes=10, 
            verbose=False
        )
        
        if training_result and training_result.get('success', False):
            print("✅ Hybrid training completed")
            
            # Test prediction with database saving
            current_data = df.tail(30)  # Last 30 data points
            
            prediction = hybrid_model.predict_hybrid_action(
                current_data,
                coin_symbol=coin_symbol,
                save_to_db=True
            )
            
            if prediction['success']:
                ensemble_pred = prediction['ensemble_prediction']
                print(f"🎯 Hybrid Prediction: {ensemble_pred['recommendation']} "
                      f"(Confidence: {prediction['confidence']:.1%})")
                
                if 'database_id' in prediction:
                    print(f"💾 Saved to database with ID: {prediction['database_id']}")
                
                return prediction
            else:
                print(f"❌ Hybrid prediction failed: {prediction.get('error', 'Unknown error')}")
                return None
        else:
            print("❌ Hybrid training failed")
            return None
            
    except Exception as e:
        print(f"❌ Hybrid analysis error: {e}")
        return None

def test_ensemble_analysis(coin_symbol="TESTCOIN"):
    """Test Ensemble analysis with database saving"""
    print("\n🎭 Testing Ensemble Analysis...")
    
    try:
        db = TradingDatabase()
        
        # Create sample model results for ensemble
        model_results = [
            {
                'model_type': 'LSTM',
                'current_price': 52000.0,
                'predicted_price': 53000.0,
                'price_change_percent': 1.92,
                'confidence': 0.75
            },
            {
                'model_type': 'DQN',
                'current_price': 52000.0,
                'predicted_price': 51500.0,
                'price_change_percent': -0.96,
                'confidence': 0.65
            },
            {
                'model_type': 'TECHNICAL',
                'current_price': 52000.0,
                'predicted_price': 52500.0,
                'price_change_percent': 0.96,
                'confidence': 0.55
            }
        ]
        
        # Ensemble weights
        weights = {
            'lstm': 0.35,
            'dqn': 0.45,
            'technical': 0.20
        }
        
        # Save ensemble analysis
        analysis_id = db.save_ensemble_analysis(
            coin_symbol, 
            model_results, 
            weights
        )
        
        if analysis_id:
            print(f"✅ Ensemble analysis saved to database (ID: {analysis_id})")
            return analysis_id
        else:
            print("❌ Ensemble analysis save failed")
            return None
            
    except Exception as e:
        print(f"❌ Ensemble analysis error: {e}")
        return None

def test_database_queries():
    """Test database query functions"""
    print("\n📊 Testing Database Queries...")
    
    try:
        db = TradingDatabase()
        
        # Get analysis history
        print("🔍 Getting analysis history...")
        analyses = db.get_analysis_history(limit=10)
        
        if analyses:
            print(f"📈 Found {len(analyses)} analyses:")
            for analysis in analyses[:3]:  # Show first 3
                print(f"  • {analysis['coin_symbol']} ({analysis['model_type']}): "
                      f"{analysis['confidence']:.1%} confidence, "
                      f"{analysis['price_change_percent']:+.2f}% prediction")
        
        # Get model performance comparison
        print("\n📊 Getting model performance comparison...")
        performance = db.get_model_performance_comparison(days=7)
        
        if performance.get('model_performance'):
            print("🏆 Model Performance Summary:")
            for model_type, stats in performance['model_performance'].items():
                print(f"  • {model_type}: {stats['total_predictions']} predictions, "
                      f"avg confidence: {stats['avg_confidence']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database query error: {e}")
        return False

def test_news_analysis_integration(coin_symbol="BTCUSDT"):
    """Test news analysis integration with database"""
    print("\n📰 Testing News Analysis Integration...")
    
    try:
        from news_analyzer import CryptoNewsAnalyzer
        
        # Initialize news analyzer
        news_analyzer = CryptoNewsAnalyzer()
        print("✅ News analyzer initialized")
        
        # Fetch news for testing
        print("🔍 Fetching test news...")
        news_list = news_analyzer.fetch_all_news(coin_symbol, days=3)
        
        if news_list:
            print(f"📰 Fetched {len(news_list)} news items")
            
            # Analyze sentiment
            sentiment_df = news_analyzer.analyze_news_sentiment_batch(news_list)
            
            if not sentiment_df.empty:
                # Prepare news analysis for database
                news_analysis = {
                    'news_sentiment': sentiment_df['overall_sentiment'].mean(),
                    'news_count': len(news_list),
                    'avg_sentiment': sentiment_df['overall_sentiment'].mean(),
                    'sentiment_confidence': sentiment_df['confidence'].mean()
                }
                
                print(f"📊 News Analysis Results:")
                print(f"   📰 Total news: {news_analysis['news_count']}")
                print(f"   😊 Avg sentiment: {news_analysis['avg_sentiment']:+.3f}")
                print(f"   🎯 Confidence: {news_analysis['sentiment_confidence']:.1%}")
                
                return news_analysis
            else:
                print("⚠️ Sentiment analysis failed")
                return None
        else:
            print("⚠️ No news could be fetched")
            return None
            
    except Exception as e:
        print(f"❌ News analysis error: {e}")
        return None

def main():
    """Main test function"""
    print("🧪 Database Analysis Integration Test")
    print("=" * 50)
    
    # Initialize database
    try:
        db = TradingDatabase()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return
    
    # Generate test data
    test_data = generate_test_data(days=100)
    
    # Test coin symbol
    coin_symbol = "BTCUSDT"
    
    # Add coin to database
    db.add_coin(coin_symbol, "Bitcoin Test")
    
    # Test DQN analysis
    dqn_result = test_dqn_analysis(test_data, coin_symbol)
    
    # Test Hybrid analysis  
    hybrid_result = test_hybrid_analysis(test_data, coin_symbol)
    
    # Test Ensemble analysis
    ensemble_result = test_ensemble_analysis(coin_symbol)
    
    # Test news analysis integration
    news_result = test_news_analysis_integration(coin_symbol)
    
    # Test database queries
    query_success = test_database_queries()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print(f"✅ DQN Analysis: {'Success' if dqn_result else 'Failed'}")
    print(f"✅ Hybrid Analysis: {'Success' if hybrid_result else 'Failed'}")
    print(f"✅ Ensemble Analysis: {'Success' if ensemble_result else 'Failed'}")
    print(f"✅ News Analysis: {'Success' if news_result else 'Failed'}")
    print(f"✅ Database Queries: {'Success' if query_success else 'Failed'}")
    
    if dqn_result or hybrid_result or ensemble_result or news_result:
        print("\n🎉 Database integration test completed successfully!")
        print("📊 Check your database for the new analysis results.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main() 