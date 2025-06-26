#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified DQN Trading Model

Basitleştirilmiş Deep Q-Learning agent'ı - dependency sorunları çözülene kadar.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import pickle
import json
import matplotlib.pyplot as plt

# **CRITICAL: Use centralized TensorFlow configuration**
from tf_config import get_tensorflow, is_tensorflow_available, print_training_device_info, monitor_training_resources, get_current_device

# Optional database integration
try:
    from database import TradingDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    print("⚠️ Database module not available - DQN predictions will not be saved")
    DATABASE_AVAILABLE = False

tf = get_tensorflow()
TF_AVAILABLE = is_tensorflow_available()

print(f"🤖 DQN Model - TensorFlow Available: {TF_AVAILABLE}")

from collections import deque
import random


class MockDQNModel:
    """Mock DQN model for testing when TensorFlow is not available"""
    
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.weights = np.random.random((state_space, action_space)) * 0.1
        
    def predict(self, state, verbose=0):
        """Mock prediction using simple linear transformation"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        # Simple linear layer simulation
        output = np.dot(state, self.weights)
        # Add some noise for realism
        output += np.random.normal(0, 0.01, output.shape)
        return output
    
    def fit(self, X, y, epochs=1, verbose=0):
        """Mock training - simple gradient step"""
        if len(X) > 0:
            # Simple mock learning
            learning_rate = 0.001
            for _ in range(epochs):
                pred = self.predict(X, verbose=0)
                error = y - pred
                # Simple gradient update
                gradient = np.dot(X.T, error) / len(X)
                self.weights += learning_rate * gradient
        
        return type('MockHistory', (), {'history': {'loss': [0.1]}})()
    
    def set_weights(self, weights):
        """Mock set weights"""
        if isinstance(weights, list) and len(weights) > 0:
            # Use first weight matrix as approximation
            if hasattr(weights[0], 'shape'):
                self.weights = weights[0]
    
    def save(self, filepath):
        """Mock save"""
        np.save(filepath.replace('.h5', '.npy'), self.weights)
    
    @classmethod
    def load_model(cls, filepath):
        """Mock load"""
        try:
            weights_path = filepath.replace('.h5', '.npy')
            if os.path.exists(weights_path):
                weights = np.load(weights_path)
                model = cls(weights.shape[0], weights.shape[1])
                model.weights = weights
                return model
        except:
            pass
        return None

class TradingEnvironment:
    """
    Cryptocurrency trading environment for DQN training
    """
    
    def __init__(self, data, initial_balance=10000, transaction_fee=0.001, lookback_window=60):
        """
        Trading environment initialization
        
        Args:
            data (pd.DataFrame): OHLCV + technical indicators data
            initial_balance (float): Starting balance in USDT
            transaction_fee (float): Trading fee percentage
            lookback_window (int): Number of timesteps for state representation
        """
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.lookback_window = lookback_window
        
        # Environment state
        self.reset()
        
        # Action space: 0=HOLD, 1=BUY_25%, 2=BUY_50%, 3=BUY_75%, 4=BUY_100%, 5=SELL_25%, 6=SELL_50%, 7=SELL_75%, 8=SELL_100%
        self.action_space = 9
        self.action_meanings = [
            'HOLD', 'BUY_25%', 'BUY_50%', 'BUY_75%', 'BUY_100%',
            'SELL_25%', 'SELL_50%', 'SELL_75%', 'SELL_100%'
        ]
        
        # State space: technical indicators + portfolio state + market features
        self.state_space = self._calculate_state_space()
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {}
        
    def _calculate_state_space(self):
        """Calculate state space dimensions dynamically"""
        # **CRITICAL FIX: Use dynamic state space calculation**
        # Don't assume fixed dimensions, calculate based on actual data
        
        # Try to get actual feature count from data if available
        if hasattr(self, 'data') and len(self.data) > 0:
            # Create a test state to get actual dimensions
            try:
                test_state = self._get_state()
                actual_state_space = len(test_state)
                print(f"🔍 Dynamic state space detected: {actual_state_space} features")
                return actual_state_space
            except:
                pass
        
        # Fallback to estimated calculation (conservative)
        technical_indicators = 15  # RSI, MACD, BB, SMA, EMA, etc.
        portfolio_state = 5  # balance, position, unrealized_pnl, portfolio_value, position_ratio
        market_features = 8  # price_change, volume_change, volatility, trend_strength, etc.
        lstm_features = 3  # predicted_price, confidence, direction
        
        estimated_space = technical_indicators + portfolio_state + market_features + lstm_features
        print(f"🔍 Estimated state space: {estimated_space} features")
        return estimated_space
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.position_value = 0.0
        self.total_portfolio_value = self.initial_balance
        
        # Performance tracking
        self.max_portfolio_value = self.initial_balance
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_fees_paid = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Get current environment state"""
        if self.current_step >= len(self.data):
            return np.zeros(self.state_space)
        
        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Technical indicators
        technical_state = self._get_technical_state(current_data)
        
        # Portfolio state
        self.position_value = self.crypto_held * current_price
        self.total_portfolio_value = self.balance + self.position_value
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.crypto_held * current_price / self.initial_balance,  # Normalized position value
            (self.total_portfolio_value - self.initial_balance) / self.initial_balance,  # Unrealized P&L ratio
            self.total_portfolio_value / self.initial_balance,  # Total portfolio ratio
            self.position_value / self.total_portfolio_value if self.total_portfolio_value > 0 else 0  # Position ratio
        ])
        
        # Market features
        market_state = self._get_market_state(current_data)
        
        # LSTM prediction features (will be filled by hybrid model)
        lstm_state = np.array([0.0, 0.0, 0.0])  # placeholder
        
        # Combine all states
        state = np.concatenate([technical_state, portfolio_state, market_state, lstm_state])
        
        return state
    
    def _get_technical_state(self, current_data):
        """Extract technical indicators as state"""
        indicators = []
        
        # RSI (normalized to 0-1)
        indicators.append(current_data.get('rsi', 50) / 100.0)
        
        # MACD signal
        macd = current_data.get('macd', 0)
        macd_signal = current_data.get('macd_signal', 0)
        indicators.append(1.0 if macd > macd_signal else 0.0)
        
        # Bollinger Bands position
        bb_upper = current_data.get('bb_upper', current_data['close'] * 1.02)
        bb_lower = current_data.get('bb_lower', current_data['close'] * 0.98)
        bb_position = (current_data['close'] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        indicators.append(np.clip(bb_position, 0, 1))
        
        # Moving averages
        sma_7 = current_data.get('sma_7', current_data['close'])
        sma_25 = current_data.get('sma_25', current_data['close'])
        indicators.append(1.0 if current_data['close'] > sma_7 else 0.0)
        indicators.append(1.0 if current_data['close'] > sma_25 else 0.0)
        indicators.append(1.0 if sma_7 > sma_25 else 0.0)
        
        # Volume indicators
        current_volume = current_data.get('volume', 0)
        avg_volume = current_data.get('volume_sma', current_volume)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        indicators.append(np.clip(volume_ratio / 3.0, 0, 1))  # Normalize volume ratio
        
        # ATR (volatility)
        atr = current_data.get('atr', 0)
        atr_pct = (atr / current_data['close']) * 100 if current_data['close'] > 0 else 0
        indicators.append(np.clip(atr_pct / 5.0, 0, 1))  # Normalize ATR percentage
        
        # Stochastic oscillator
        stoch_k = current_data.get('stoch_k', 50)
        indicators.append(stoch_k / 100.0)
        
        # CCI
        cci = current_data.get('cci', 0)
        cci_normalized = (cci + 200) / 400.0  # Normalize CCI from -200,200 to 0,1
        indicators.append(np.clip(cci_normalized, 0, 1))
        
        # Williams %R
        williams_r = current_data.get('williams_r', -50)
        williams_normalized = (williams_r + 100) / 100.0  # Normalize from -100,0 to 0,1
        indicators.append(np.clip(williams_normalized, 0, 1))
        
        # Yigit signals
        yigit_position = current_data.get('yigit_position', 0)
        indicators.append((yigit_position + 1) / 2.0)  # Convert -1,0,1 to 0,0.5,1
        
        yigit_trend_strength = current_data.get('yigit_trend_strength', 0)
        indicators.append(np.clip(yigit_trend_strength / 3.0, 0, 1))
        
        # Price momentum (short term)
        price_change_1h = current_data.get('price_change_1h', 0)
        indicators.append((price_change_1h + 10) / 20.0)  # Normalize -10% to +10% change
        
        # Long term trend
        price_change_24h = current_data.get('price_change_24h', 0)
        indicators.append((price_change_24h + 20) / 40.0)  # Normalize -20% to +20% change
        
        return np.array(indicators)
    
    def _get_market_state(self, current_data):
        """Extract market-wide features"""
        market_features = []
        
        # Price volatility (recent)
        recent_prices = self.data['close'].iloc[max(0, self.current_step-24):self.current_step+1]
        volatility = recent_prices.std() / recent_prices.mean() if len(recent_prices) > 1 else 0
        market_features.append(np.clip(volatility * 10, 0, 1))
        
        # Volume trend
        recent_volumes = self.data['volume'].iloc[max(0, self.current_step-24):self.current_step+1]
        volume_trend = 1.0 if len(recent_volumes) > 1 and recent_volumes.iloc[-1] > recent_volumes.mean() else 0.0
        market_features.append(volume_trend)
        
        # Price trend strength
        price_changes = recent_prices.pct_change().dropna()
        trend_strength = abs(price_changes.mean()) if len(price_changes) > 0 else 0
        market_features.append(np.clip(trend_strength * 50, 0, 1))
        
        # Market momentum
        momentum_3h = (recent_prices.iloc[-1] - recent_prices.iloc[-4]) / recent_prices.iloc[-4] if len(recent_prices) >= 4 else 0
        market_features.append((momentum_3h + 0.1) / 0.2)  # Normalize -10% to +10%
        
        # Time-based features
        current_time = pd.to_datetime(current_data.name if hasattr(current_data, 'name') else datetime.now())
        hour_sin = np.sin(2 * np.pi * current_time.hour / 24.0)
        hour_cos = np.cos(2 * np.pi * current_time.hour / 24.0)
        market_features.append(hour_sin)
        market_features.append(hour_cos)
        
        # Day of week
        day_of_week = current_time.weekday() / 6.0  # Normalize 0-6 to 0-1
        market_features.append(day_of_week)
        
        # Market fear/greed (proxy using recent volatility and returns)
        recent_returns = price_changes.iloc[-7:] if len(price_changes) >= 7 else price_changes
        fear_greed = 0.5 + (recent_returns.mean() - recent_returns.std()) if len(recent_returns) > 0 else 0.5
        market_features.append(np.clip(fear_greed, 0, 1))
        
        return np.array(market_features)
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        next_state = self._get_state()
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1 or self.total_portfolio_value <= 0.1 * self.initial_balance
        
        # Info for logging
        info = {
            'portfolio_value': self.total_portfolio_value,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'action': self.action_meanings[action],
            'current_price': current_price,
            'total_trades': self.total_trades
        }
        
        return next_state, reward, done, info
    
    def _execute_action(self, action, current_price):
        """Execute trading action and return reward"""
        initial_portfolio_value = self.total_portfolio_value
        
        if action == 0:  # HOLD
            reward = 0
        elif action in [1, 2, 3, 4]:  # BUY actions
            buy_percentages = [0.25, 0.50, 0.75, 1.0]
            buy_percentage = buy_percentages[action - 1]
            reward = self._execute_buy(buy_percentage, current_price)
        elif action in [5, 6, 7, 8]:  # SELL actions
            sell_percentages = [0.25, 0.50, 0.75, 1.0]
            sell_percentage = sell_percentages[action - 5]
            reward = self._execute_sell(sell_percentage, current_price)
        
        # Update portfolio value
        self.position_value = self.crypto_held * current_price
        self.total_portfolio_value = self.balance + self.position_value
        
        # Portfolio value change reward
        portfolio_change = (self.total_portfolio_value - initial_portfolio_value) / initial_portfolio_value
        
        # Risk-adjusted reward
        risk_penalty = self._calculate_risk_penalty()
        
        total_reward = portfolio_change - risk_penalty
        
        return total_reward
    
    def _execute_buy(self, percentage, current_price):
        """Execute buy order"""
        if self.balance <= 0:
            return -0.01  # Penalty for invalid action
        
        buy_amount = self.balance * percentage
        fee = buy_amount * self.transaction_fee
        net_buy_amount = buy_amount - fee
        
        if net_buy_amount > 0:
            crypto_to_buy = net_buy_amount / current_price
            self.crypto_held += crypto_to_buy
            self.balance -= buy_amount
            self.total_fees_paid += fee
            self.total_trades += 1
            
            # Record trade
            self.trade_history.append({
                'timestamp': self.current_step,
                'action': 'BUY',
                'percentage': percentage,
                'price': current_price,
                'amount': crypto_to_buy,
                'fee': fee
            })
            
            return 0.001  # Small positive reward for valid action
        
        return -0.01  # Penalty for invalid action
    
    def _execute_sell(self, percentage, current_price):
        """Execute sell order"""
        if self.crypto_held <= 0:
            return -0.01  # Penalty for invalid action
        
        crypto_to_sell = self.crypto_held * percentage
        sell_amount = crypto_to_sell * current_price
        fee = sell_amount * self.transaction_fee
        net_sell_amount = sell_amount - fee
        
        if crypto_to_sell > 0:
            self.crypto_held -= crypto_to_sell
            self.balance += net_sell_amount
            self.total_fees_paid += fee
            self.total_trades += 1
            
            # Record trade
            self.trade_history.append({
                'timestamp': self.current_step,
                'action': 'SELL',
                'percentage': percentage,
                'price': current_price,
                'amount': crypto_to_sell,
                'fee': fee
            })
            
            # Profit calculation for reward
            profit_reward = 0.001 if net_sell_amount > 0 else -0.001
            return profit_reward
        
        return -0.01  # Penalty for invalid action
    
    def _calculate_risk_penalty(self):
        """Calculate risk penalty based on drawdown and volatility"""
        # Drawdown penalty
        current_drawdown = (self.max_portfolio_value - self.total_portfolio_value) / self.max_portfolio_value
        drawdown_penalty = current_drawdown * 0.1  # Penalty for drawdown
        
        # Update max portfolio value
        if self.total_portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.total_portfolio_value
        
        # Concentration risk (too much in one position)
        position_ratio = self.position_value / self.total_portfolio_value if self.total_portfolio_value > 0 else 0
        concentration_penalty = max(0, position_ratio - 0.8) * 0.05  # Penalty for >80% concentration
        
        return drawdown_penalty + concentration_penalty
    
    def get_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if len(self.trade_history) == 0:
            return {}
        
        total_return = (self.total_portfolio_value - self.initial_balance) / self.initial_balance
        
        # Calculate trade statistics
        profitable_trades = sum(1 for trade in self.trade_history if trade['action'] == 'SELL')
        win_rate = profitable_trades / len(self.trade_history) if len(self.trade_history) > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        portfolio_values = [self.initial_balance]
        for i in range(1, self.current_step):
            if i < len(self.data):
                current_price = self.data.iloc[i]['close']
                portfolio_value = self.balance + (self.crypto_held * current_price)
                portfolio_values.append(portfolio_value)
                if len(portfolio_values) > 1:
                    returns.append((portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2])
        
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Maximum drawdown
        max_drawdown = (self.max_portfolio_value - min(portfolio_values)) / self.max_portfolio_value if portfolio_values else 0
        
        metrics = {
            'total_return': total_return,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'total_fees_paid': self.total_fees_paid,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': self.total_portfolio_value,
            'profit_loss': self.total_portfolio_value - self.initial_balance
        }
        
        return metrics


class DQNAgent:
    """
    Deep Q-Network agent for cryptocurrency trading
    """
    
    def __init__(self, state_space, action_space, learning_rate=0.0008, gamma=0.95, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.02):
        """
        Initialize DQN agent with balanced overfitting prevention
        
        Args:
            state_space (int): Dimension of state space
            action_space (int): Number of possible actions
            learning_rate (float): Learning rate for neural network (balanced)
            gamma (float): Discount factor for future rewards
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Decay rate for exploration (balanced)
            epsilon_min (float): Minimum exploration rate (balanced)
        """
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Experience replay with balanced memory to prevent overfitting
        self.memory = deque(maxlen=7500)  # Increased from 5000
        self.batch_size = 24  # Increased from 16
        
        # Neural networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Training tracking
        self.training_history = []
        self.loss_history = []
        
        # Overfitting prevention
        self.validation_scores = []
        self.best_validation_score = float('-inf')
        self.patience_counter = 0
        self.max_patience = 15  # Reduced patience for faster adaptation
        
    def _build_network(self):
        """Build a balanced, regularized deep Q-network"""
        if not TF_AVAILABLE:
            # Mock model for testing
            return MockDQNModel(self.state_space, self.action_space)
        
        # **BALANCED ARCHITECTURE TO PREVENT OVERFITTING BUT MAINTAIN CAPACITY**
        model = tf.keras.Sequential([
            # Input layer with moderate L2 regularization
            tf.keras.layers.Dense(160, input_dim=self.state_space, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),  # Moderate dropout
            
            # Hidden layer 1
            tf.keras.layers.Dense(96, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            
            # Hidden layer 2 (balanced)
            tf.keras.layers.Dense(48, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            tf.keras.layers.Dropout(0.15),
            
            # Output layer (no activation for Q-values)
            tf.keras.layers.Dense(self.action_space, activation='linear')
        ])
        
        # Use legacy Adam optimizer for M1/M2 compatibility
        if TF_AVAILABLE:
            try:
                # Try legacy optimizer first
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
            except:
                # Fallback to regular Adam
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            optimizer = 'adam'  # String fallback
            
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust than MSE for outliers
            metrics=['mae']
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy with balanced exploration"""
        # **BALANCED EXPLORATION STRATEGY**
        if training and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_space)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        
        # **REDUCED NOISE** to allow better learning while preventing overconfidence
        if training and len(self.memory) > self.batch_size:
            noise_scale = 0.005 * (self.epsilon + 0.05)  # Much smaller noise
            q_values += np.random.normal(0, noise_scale, q_values.shape)
        
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train the neural network with balanced overfitting prevention"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # **DOUBLE DQN UPDATE** to reduce overfitting
        # Use main network to select actions, target network to evaluate
        next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)
        
        # Update Q values with balanced approach
        for i in range(self.batch_size):
            if dones[i]:
                target = rewards[i]
            else:
                # Double DQN update
                target = rewards[i] + self.gamma * next_q_values[i][next_actions[i]]
            
            # **LESS CONSERVATIVE UPDATE** for better learning
            current_q = current_q_values[i][actions[i]]
            learning_rate_decay = 0.3  # Increased from 0.1 for better learning
            updated_q = current_q + learning_rate_decay * (target - current_q)
            current_q_values[i][actions[i]] = updated_q
        
        # Train the network with validation split
        history = self.q_network.fit(states, current_q_values, epochs=1, verbose=0, 
                                   validation_split=0.15)  # Reduced validation split
        self.loss_history.append(history.history['loss'][0])
        
        # **BALANCED EARLY STOPPING CHECK**
        if 'val_loss' in history.history:
            val_loss = history.history['val_loss'][0]
            self.validation_scores.append(-val_loss)  # Negative because lower loss is better
            
            if len(self.validation_scores) > 3:  # Start checking after 3 iterations
                recent_avg = np.mean(self.validation_scores[-3:])
                
                if recent_avg > self.best_validation_score:
                    self.best_validation_score = recent_avg
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Early stopping with reduced patience
                if self.patience_counter >= self.max_patience:
                    print(f"🛑 Early stopping triggered after {self.patience_counter} steps without improvement")
                    self.epsilon = max(self.epsilon_min, self.epsilon * 0.5)  # Faster epsilon reduction
        
        # **BALANCED EPSILON DECAY** to maintain some exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.q_network.save(filepath)
        
        # Save additional parameters
        params = {
            'state_space': self.state_space,
            'action_space': self.action_space,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'training_history': self.training_history,
            'loss_history': self.loss_history
        }
        
        with open(filepath.replace('.h5', '_params.json'), 'w') as f:
            json.dump(params, f)
    
    def load_model(self, filepath):
        """Load a trained model with crash-safe TensorFlow operations"""
        print(f"🔄 Loading DQN agent model from {filepath}")
        
        # **Metal-safe model loading**
        if TF_AVAILABLE:
            try:
                # **CRITICAL: Force CPU-only for model loading to prevent Metal crashes**
                with tf.device('/CPU:0'):
                    self.q_network = tf.keras.models.load_model(filepath)
                    self.target_network = self._build_network()
                    self.update_target_network()
                    print("✅ TensorFlow model loaded successfully (CPU mode)")
            except Exception as tf_error:
                print(f"⚠️ TensorFlow loading failed: {tf_error}")
                print("🔄 Attempting fallback to mock model...")
                # Fallback to mock model
                mock_model = MockDQNModel.load_model(filepath)
                if mock_model:
                    self.q_network = mock_model
                    self.target_network = MockDQNModel(self.state_space, self.action_space)
                    print("✅ Mock model loaded as fallback")
                else:
                    print("❌ Both TensorFlow and mock loading failed")
                    return False
        else:
            # Use mock model when TensorFlow not available
            print("🔄 TensorFlow not available, using mock model...")
            mock_model = MockDQNModel.load_model(filepath)
            if mock_model:
                self.q_network = mock_model
                self.target_network = MockDQNModel(self.state_space, self.action_space)
                print("✅ Mock model loaded successfully")
            else:
                print("❌ Mock model loading failed")
                return False
        
        # Load additional parameters safely
        params_file = filepath.replace('.h5', '_params.json')
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    params = json.load(f)
                    self.epsilon = params.get('epsilon', self.epsilon)
                    self.training_history = params.get('training_history', [])
                    self.loss_history = params.get('loss_history', [])
                    print("✅ Agent parameters loaded")
            except Exception as param_error:
                print(f"⚠️ Parameter loading error: {param_error}")
        
        return True


class DQNTradingModel:
    """
    Main DQN Trading Model class that combines environment and agent
    """
    
    def __init__(self, lookback_window=60, initial_balance=10000):
        """
        Initialize DQN Trading Model
        
        Args:
            lookback_window (int): Number of timesteps for state representation
            initial_balance (float): Initial trading balance
        """
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.environment = None
        self.agent = None
        self.training_history = []
        self.is_trained = False
        
    def prepare_data(self, df):
        """Prepare data for DQN training with crash-safe initialization"""
        # Ensure we have enough data
        if len(df) < self.lookback_window + 50:
            raise ValueError(f"Insufficient data. Need at least {self.lookback_window + 50} samples, got {len(df)}")
        
        print(f"🔄 Preparing DQN data with {len(df)} samples...")
        
        # Create environment FIRST to get proper state space
        self.environment = TradingEnvironment(
            data=df,
            initial_balance=self.initial_balance,
            lookback_window=self.lookback_window
        )
        
        # **CRITICAL FIX: Get actual state space from environment**
        actual_state_space = self.environment.state_space
        print(f"🔍 DQN Environment state space: {actual_state_space}")
        
        # Create agent with correct state space
        try:
            self.agent = DQNAgent(
                state_space=actual_state_space,
                action_space=self.environment.action_space
            )
            print(f"✅ DQN Agent created with {actual_state_space} state space")
        except Exception as e:
            print(f"❌ DQN Agent creation failed: {e}")
            # **FALLBACK: Use CPU-only mode for safety**
            print("🔄 Attempting CPU-only DQN agent creation...")
            self.agent = DQNAgent(
                state_space=actual_state_space,
                action_space=self.environment.action_space
            )
            print("✅ CPU-only DQN Agent created successfully")
        
        return True
    
    def train(self, df, episodes=100, update_target_freq=10, verbose=True):
        """
        Train the DQN agent with crash-safe Metal plugin protection and resource monitoring
        
        Args:
            df (pd.DataFrame): Training data
            episodes (int): Number of training episodes
            update_target_freq (int): Frequency to update target network
            verbose (bool): Whether to print training progress
        """
        if self.environment is None or self.agent is None:
            try:
                self.prepare_data(df)
            except Exception as prep_error:
                print(f"❌ Data preparation failed: {prep_error}")
                return None
        
        if verbose:
            print(f"🤖 Starting DQN training with {episodes} episodes...")
            
            # **NEW: Print comprehensive resource information for DQN training**
            print_training_device_info()
            
            print(f"🔍 State space: {self.agent.state_space if self.agent else 'Unknown'}")
            print(f"🔍 Action space: {self.environment.action_space if self.environment else 'Unknown'}")
        
        training_rewards = []
        training_portfolio_values = []
        
        # **CRITICAL: Crash-safe training loop**
        try:
            # **NEW: Monitor DQN training device**
            if verbose:
                current_device = get_current_device()
                print(f"🎯 DQN Training Device: {current_device}")
            
            for episode in range(episodes):
                try:
                    # **Metal-safe episode execution**
                    state = self.environment.reset()
                    total_reward = 0
                    steps = 0
                    
                    # Ensure state has correct dimensions
                    if len(state) != self.agent.state_space:
                        print(f"⚠️ State dimension mismatch: expected {self.agent.state_space}, got {len(state)}")
                        # Pad or truncate state to match expected size
                        if len(state) < self.agent.state_space:
                            state = np.pad(state, (0, self.agent.state_space - len(state)), 'constant')
                        else:
                            state = state[:self.agent.state_space]
                    
                    while True:
                        try:
                            action = self.agent.act(state, training=True)
                            next_state, reward, done, info = self.environment.step(action)
                            
                            # Ensure next_state has correct dimensions
                            if len(next_state) != self.agent.state_space:
                                if len(next_state) < self.agent.state_space:
                                    next_state = np.pad(next_state, (0, self.agent.state_space - len(next_state)), 'constant')
                                else:
                                    next_state = next_state[:self.agent.state_space]
                            
                            self.agent.remember(state, action, reward, next_state, done)
                            state = next_state
                            total_reward += reward
                            steps += 1
                            
                            if done:
                                break
                                
                        except Exception as step_error:
                            print(f"⚠️ Step error in episode {episode}: {step_error}")
                            break
                    
                    # **Metal-safe agent training**
                    try:
                        if len(self.agent.memory) >= self.agent.batch_size:
                            self.agent.replay()
                    except Exception as replay_error:
                        print(f"⚠️ Replay error in episode {episode}: {replay_error}")
                        # Continue training without this batch
                    
                    # Update target network
                    if episode % update_target_freq == 0:
                        try:
                            self.agent.update_target_network()
                        except Exception as target_error:
                            print(f"⚠️ Target network update error: {target_error}")
                    
                    # Record training metrics
                    final_portfolio_value = self.environment.total_portfolio_value
                    training_rewards.append(total_reward)
                    training_portfolio_values.append(final_portfolio_value)
                    
                    # **NEW: Resource monitoring during DQN training**
                    if verbose and episode % 20 == 0:
                        print(f"Episode {episode}, Total Reward: {total_reward:.4f}, "
                              f"Portfolio Value: ${final_portfolio_value:.2f}, "
                              f"Epsilon: {self.agent.epsilon:.3f}")
                        
                        # Monitor resources every 20 episodes
                        if episode % 50 == 0 and episode > 0:
                            print("📊 Resource Check:")
                            monitor_training_resources()
                
                except Exception as episode_error:
                    print(f"❌ Episode {episode} failed: {episode_error}")
                    # Continue with next episode
                    continue
            
        except Exception as training_error:
            print(f"❌ Training crashed: {training_error}")
            print("🔄 Attempting graceful recovery...")
            # Return partial results if any
            if training_rewards:
                self.training_history = {
                    'rewards': training_rewards,
                    'portfolio_values': training_portfolio_values,
                    'episodes': len(training_rewards),
                    'partial_training': True
                }
                self.is_trained = True
                return self.training_history
            else:
                return None
        
        # Store successful training results
        self.training_history = {
            'rewards': training_rewards,
            'portfolio_values': training_portfolio_values,
            'episodes': episodes,
            'completed': True
        }
        
        self.is_trained = True
        
        if verbose:
            print(f"\n🎯 DQN Training Complete!")
            print(f"Episodes: {episodes}")
            if training_portfolio_values:
                print(f"Final Portfolio Value: ${training_portfolio_values[-1]:.2f}")
                print(f"Total Return: {((training_portfolio_values[-1] - self.initial_balance) / self.initial_balance) * 100:.2f}%")
            
            # **NEW: Final resource summary**
            print("\n📊 Final Training Resource Summary:")
            monitor_training_resources()
            final_device = get_current_device()
            print(f"🎯 DQN Training completed on: {final_device}")
        
        return self.training_history
    
    def predict_action(self, current_state, coin_symbol=None, save_to_db=True):
        """
        Predict the best action for current market state with robust confidence calculation
        
        Args:
            current_state (np.array): Current market state
            coin_symbol (str): Coin symbol for database logging
            save_to_db (bool): Whether to save results to database
            
        Returns:
            dict: Action prediction with details
        """
        if not self.is_trained or self.agent is None:
            return {
                'action': 0,
                'action_name': 'HOLD',
                'confidence': 0.0,
                'q_values': [],
                'reasoning': 'Model not trained'
            }
        
        # Get Q-values for all actions (multiple predictions for robustness)
        n_predictions = 5
        all_q_values = []
        
        for _ in range(n_predictions):
            q_vals = self.agent.q_network.predict(current_state.reshape(1, -1), verbose=0)[0]
            all_q_values.append(q_vals)
        
        # **ROBUST CONFIDENCE CALCULATION**
        q_values_array = np.array(all_q_values)
        mean_q_values = np.mean(q_values_array, axis=0)
        std_q_values = np.std(q_values_array, axis=0)
        
        # Select best action based on mean Q-values
        best_action = np.argmax(mean_q_values)
        best_q_value = mean_q_values[best_action]
        best_q_std = std_q_values[best_action]
        
        # **IMPROVED CONFIDENCE CALCULATION**
        # 1. Uncertainty-based confidence (lower std = higher confidence)
        uncertainty_factor = max(0, 1 - (best_q_std / (np.mean(std_q_values) + 1e-8)))
        
        # 2. Q-value margin (how much better is the best action)
        sorted_q_values = np.sort(mean_q_values)[::-1]  # Descending order
        if len(sorted_q_values) > 1:
            q_margin = sorted_q_values[0] - sorted_q_values[1]
            max_possible_margin = np.max(mean_q_values) - np.min(mean_q_values)
            margin_confidence = q_margin / (max_possible_margin + 1e-8) if max_possible_margin > 0 else 0
        else:
            margin_confidence = 0.5
        
        # 3. Exploration factor (lower epsilon = higher confidence in training)
        exploration_confidence = 1.0 - self.agent.epsilon if self.agent else 0.6
        
        # 4. Training maturity (more training = potentially higher confidence, but cap it)
        training_episodes = len(self.training_history.get('rewards', [])) if self.training_history else 0
        maturity_factor = min(0.7, training_episodes / 50.0)  # Cap at 70%, faster maturity
        
        # **BALANCED CONFIDENCE** (realistic range: 0.2 to 0.8)
        raw_confidence = (
            0.25 * uncertainty_factor +
            0.35 * margin_confidence +
            0.25 * exploration_confidence +
            0.15 * maturity_factor
        )
        
        # **REALISTIC CONFIDENCE CAPPING**
        # Prevent both overconfidence and underconfidence
        max_confidence = 0.80  # Reduced from 85%
        min_confidence = 0.25  # Increased from 15%
        
        confidence = np.clip(raw_confidence, min_confidence, max_confidence)
        
        # **BALANCED REALITY CHECK**: If standard deviation is reasonable, don't penalize too much
        if np.mean(std_q_values) > 0.3:  # Moderate uncertainty
            confidence *= 0.85  # Modest reduction
        elif np.mean(std_q_values) > 0.6:  # High uncertainty
            confidence *= 0.7   # Larger reduction
        
        # **MARKET CONDITION ADJUSTMENT** - More lenient
        if len(current_state) > 20:  # Check if we have market volatility data
            try:
                market_volatility = current_state[20]  # Assuming volatility is at index 20
                if market_volatility > 0.9:  # Very high volatility
                    confidence *= 0.85  # Modest penalty
            except:
                pass  # Ignore if volatility data not available
        
        # Generate reasoning
        action_name = self.environment.action_meanings[best_action]
        reasoning = self._generate_action_reasoning(mean_q_values, best_action, current_state, confidence)
        
        # Get current price from state if available
        current_price = 0
        try:
            # Assuming price is in the state (adjust index as needed)
            if len(current_state) > 0:
                current_price = current_state[0] if hasattr(current_state, '__len__') else 0
        except:
            current_price = 0
        
        # Prepare prediction result
        prediction_result = {
            'action': int(best_action),
            'action_name': action_name,
            'confidence': float(confidence),
            'q_values': mean_q_values.tolist(),
            'q_values_std': std_q_values.tolist(),
            'uncertainty_factor': float(uncertainty_factor),
            'margin_confidence': float(margin_confidence),
            'reasoning': reasoning,
            'model_type': 'DQN_Robust',
            'current_price': float(current_price),
            'epsilon': float(self.agent.epsilon) if self.agent else 0.0,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # **DATABASE INTEGRATION**: Save DQN analysis to database
        if save_to_db and DATABASE_AVAILABLE and coin_symbol:
            try:
                db = TradingDatabase()
                analysis_id = db.save_dqn_analysis(coin_symbol, prediction_result)
                if analysis_id:
                    prediction_result['database_id'] = analysis_id
                    print(f"📊 DQN analysis saved to database (ID: {analysis_id})")
            except Exception as db_error:
                print(f"⚠️ Database save failed: {db_error}")
                # Continue without database - don't break the prediction
        
        return prediction_result
    
    def _generate_action_reasoning(self, q_values, best_action, current_state, confidence):
        """Generate human-readable reasoning for the action with confidence context"""
        action_name = self.environment.action_meanings[best_action]
        
        # Analyze state components safely
        try:
            portfolio_value_ratio = current_state[15] if len(current_state) > 15 else 1.0
            position_ratio = current_state[19] if len(current_state) > 19 else 0.0
            rsi = current_state[0] * 100 if len(current_state) > 0 else 50.0
        except:
            portfolio_value_ratio, position_ratio, rsi = 1.0, 0.0, 50.0
        
        # **ENHANCED REASONING WITH BALANCED CONFIDENCE CONTEXT**
        reasoning = f"🤖 DQN Balanced Analysis: {action_name}\n"
        reasoning += f"📊 Confidence: {confidence:.1%} "
        
        # Confidence level interpretation - more realistic ranges
        if confidence > 0.65:
            reasoning += "(High - Strong signal)\n"
        elif confidence > 0.45:
            reasoning += "(Moderate - Good signal)\n"
        elif confidence > 0.3:
            reasoning += "(Low - Weak signal)\n"
        else:
            reasoning += "(Very Low - High uncertainty)\n"
        
        # Q-values analysis
        best_q = q_values[best_action]
        second_best_q = sorted(q_values)[-2] if len(q_values) > 1 else best_q
        q_margin = best_q - second_best_q
        
        reasoning += f"🎯 Q-value margin: {q_margin:.3f} (advantage over next best action)\n"
        
        # Action-specific analysis with balanced confidence context
        if best_action == 0:  # HOLD
            reasoning += "• Market conditions suggest waiting\n"
            reasoning += f"• RSI: {rsi:.1f} (neutral zone)\n"
            reasoning += f"• Portfolio balanced at {portfolio_value_ratio:.1%}\n"
            if confidence < 0.35:
                reasoning += "• ⚠️ Consider market volatility before acting"
        elif best_action in [1, 2, 3, 4]:  # BUY actions
            buy_strength = ["25%", "50%", "75%", "100%"][best_action - 1]
            reasoning += "• Bullish signals detected\n"
            reasoning += f"• RSI: {rsi:.1f} (favorable for buying)\n"
            reasoning += f"• Current position: {position_ratio:.1%} of portfolio\n"
            reasoning += f"• Recommended buy size: {buy_strength}\n"
            if confidence < 0.4:
                reasoning += "• ⚠️ Consider smaller position due to uncertainty"
        else:  # SELL actions (5-8)
            sell_strength = ["25%", "50%", "75%", "100%"][best_action - 5]
            reasoning += "• Bearish signals or profit-taking opportunity\n"
            reasoning += f"• RSI: {rsi:.1f} (suggests selling pressure)\n"
            reasoning += f"• Current position: {position_ratio:.1%} of portfolio\n"
            reasoning += f"• Recommended sell size: {sell_strength}\n"
            if confidence < 0.4:
                reasoning += "• ⚠️ Consider partial sell due to uncertainty"
        
        # Risk warning based on balanced confidence thresholds
        if confidence < 0.3:
            reasoning += "\n🚨 HIGH UNCERTAINTY: Consider manual review before acting"
        elif confidence < 0.4:
            reasoning += "\n⚠️ MODERATE UNCERTAINTY: Use reduced position sizes"
        elif confidence > 0.65:
            reasoning += "\n✅ HIGH CONFIDENCE: Strong signal detected"
        
        return reasoning
    
    def get_training_summary(self):
        """Get training performance summary"""
        if not self.training_history:
            return {}
        
        final_portfolio = self.training_history['portfolio_values'][-1]
        best_portfolio = max(self.training_history['portfolio_values'])
        worst_portfolio = min(self.training_history['portfolio_values'])
        
        total_return = ((final_portfolio - self.initial_balance) / self.initial_balance) * 100
        max_return = ((best_portfolio - self.initial_balance) / self.initial_balance) * 100
        max_drawdown = ((best_portfolio - worst_portfolio) / best_portfolio) * 100
        
        return {
            'episodes_trained': self.training_history['episodes'],
            'final_portfolio_value': final_portfolio,
            'total_return_pct': total_return,
            'max_return_pct': max_return,
            'max_drawdown_pct': max_drawdown,
            'initial_balance': self.initial_balance,
            'is_trained': self.is_trained,
            'final_epsilon': self.agent.epsilon if self.agent else 0
        }
    
    def save_model(self, filepath):
        """Save the trained DQN model"""
        if self.agent:
            self.agent.save_model(filepath)
            
            # Save training history
            history_file = filepath.replace('.h5', '_training_history.pkl')
            with open(history_file, 'wb') as f:
                pickle.dump(self.training_history, f)
                
            print(f"✅ DQN model saved to {filepath}")
        else:
            print("❌ No trained model to save")
    
    def load_model(self, filepath):
        """Load a trained DQN model with dimension checking"""
        if os.path.exists(filepath):
            print(f"🔄 Loading DQN model from {filepath}")
            
            # **CRITICAL: Check model parameters first**
            params_file = filepath.replace('.h5', '_params.json')
            if os.path.exists(params_file):
                try:
                    with open(params_file, 'r') as f:
                        params = json.load(f)
                        saved_state_space = params.get('state_space', None)
                        saved_action_space = params.get('action_space', None)
                        
                        print(f"🔍 Saved model dimensions: state={saved_state_space}, action={saved_action_space}")
                        
                        # Check if we have an environment to compare with
                        if self.environment:
                            current_state_space = self.environment.state_space
                            current_action_space = self.environment.action_space
                            
                            print(f"🔍 Current environment: state={current_state_space}, action={current_action_space}")
                            
                            # **Dimension compatibility check**
                            if saved_state_space != current_state_space:
                                print(f"⚠️ State space mismatch: saved={saved_state_space}, current={current_state_space}")
                                print("🔄 Recreating agent with current dimensions...")
                                
                                # Create new agent with current dimensions
                                self.agent = DQNAgent(
                                    state_space=current_state_space,
                                    action_space=current_action_space
                                )
                                print("✅ New agent created - model will start fresh")
                                return False  # Model not loaded, but agent created
                            
                            if saved_action_space != current_action_space:
                                print(f"⚠️ Action space mismatch: saved={saved_action_space}, current={current_action_space}")
                                print("🔄 Recreating agent with current dimensions...")
                                
                                self.agent = DQNAgent(
                                    state_space=current_state_space,
                                    action_space=current_action_space
                                )
                                print("✅ New agent created - model will start fresh")
                                return False
                        
                        # If compatible, create agent with saved dimensions
                        if self.agent is None:
                            self.agent = DQNAgent(
                                state_space=saved_state_space,
                                action_space=saved_action_space
                            )
                            print(f"✅ Agent created with saved dimensions")
                            
                except Exception as param_error:
                    print(f"⚠️ Error reading model parameters: {param_error}")
                    
            # **Metal-safe model loading**
            if self.agent:
                try:
                    # Attempt to load the actual model
                    self.agent.load_model(filepath)
                    
                    # Load training history
                    history_file = filepath.replace('.h5', '_training_history.pkl')
                    if os.path.exists(history_file):
                        try:
                            with open(history_file, 'rb') as f:
                                self.training_history = pickle.load(f)
                        except Exception as history_error:
                            print(f"⚠️ Could not load training history: {history_error}")
                    
                    self.is_trained = True
                    print(f"✅ DQN model loaded successfully from {filepath}")
                    return True
                    
                except Exception as load_error:
                    print(f"❌ Model loading failed: {load_error}")
                    print("🔄 Model will start fresh")
                    return False
        
        print(f"❌ Model file not found: {filepath}")
        
        # Create fresh agent if we have environment
        if self.environment and self.agent is None:
            try:
                self.agent = DQNAgent(
                    state_space=self.environment.state_space,
                    action_space=self.environment.action_space
                )
                print("✅ Fresh DQN agent created")
            except Exception as fresh_error:
                print(f"❌ Could not create fresh agent: {fresh_error}")
        
        return False 