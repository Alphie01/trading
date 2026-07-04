#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance Otomatik Trading Sistemi

Bu modül Binance API'sini kullanarak otomatik spot ve futures trading yapar.
Gelişmiş pozisyon yönetimi ve risk kontrolü içerir.
"""

import os
import ccxt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class BinanceTrader:
    """
    Binance otomatik trading sınıfı
    Spot ve Futures işlemlerini destekler
    """
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        """
        Binance trader'ı başlatır
        
        Args:
            api_key (str): Binance API anahtarı
            secret_key (str): Binance Secret anahtarı
            testnet (bool): Test modu (varsayılan: True)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        
        # Exchange bağlantıları
        self.spot_exchange = None
        self.futures_exchange = None
        
        # Trading ayarları
        self.default_risk_percent = 2.0  # Portföyün %2'si risk
        self.max_risk_percent = 5.0  # Maksimum %5 risk
        self.min_order_amount = 10.0  # Minimum işlem tutarı (USDT)
        self.max_leverage = 10  # Maksimum kaldıraç
        
        # Pozisyon takibi
        self.current_positions = {}
        self.trading_history = []
        
        # Log dosyası
        self.log_file = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Bağlantıları başlat
        self._initialize_exchanges()
        
        print(f"🚀 Binance Trader başlatıldı!")
        print(f"📊 Mod: {'TESTNET' if testnet else 'LIVE TRADING'}")
        print(f"⚠️ Risk Limiti: %{self.default_risk_percent}")
    
    def _initialize_exchanges(self):
        """
        Binance exchange bağlantılarını başlatır
        """
        try:
            # Spot Exchange (timeout: süresiz asılmayı önler)
            self.spot_exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'sandbox': self.testnet,
                'enableRateLimit': True,
                'timeout': 15000,
                'options': {
                    'defaultType': 'spot'
                }
            })

            # Futures Exchange
            self.futures_exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'sandbox': self.testnet,
                'enableRateLimit': True,
                'timeout': 15000,
                'options': {
                    'defaultType': 'future'
                }
            })
        except Exception as e:
            print(f"❌ Binance exchange oluşturma hatası: {str(e)}")
            raise

        # Test bağlantısı — NON-FATAL: load_markets() ağ ister. Yavaş/engelli/timeout olursa
        # LOGLA ve DEVAM et (raise ETME) → construction ASILMAZ, web boot bloklanmaz (50dk hang'in ikincil sebebi).
        try:
            self.spot_exchange.load_markets()
            print("✅ Binance spot bağlantısı başarılı!")
        except Exception as e:
            print(f"⚠️ Binance spot load_markets atlandı (ağ/erişim): {str(e)}")

        # Futures (fapi) çoğu bölgede coğrafi engelli → opsiyonel; BINANCE_FUTURES_ENABLED=false ile tamamen atlanır.
        if os.getenv('BINANCE_FUTURES_ENABLED', 'true').lower() == 'true':
            try:
                self.futures_exchange.load_markets()
                print("✅ Binance futures bağlantısı başarılı!")
            except Exception as e:
                print(f"⚠️ Binance futures load_markets atlandı (geo-engel/ağ): {str(e)}")
    
    def log_message(self, message: str):
        """
        Trading loglarını hem ekrana hem dosyaya yazar
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def get_account_balance(self, exchange_type: str = 'spot') -> Dict:
        """
        Hesap bakiyesini getirir
        
        Args:
            exchange_type (str): 'spot' veya 'futures'
        
        Returns:
            Dict: Bakiye bilgileri
        """
        try:
            exchange = self.spot_exchange if exchange_type == 'spot' else self.futures_exchange
            balance = exchange.fetch_balance()
            
            # USDT bakiyesi
            usdt_balance = balance.get('USDT', {})
            
            return {
                'total': usdt_balance.get('total', 0),
                'free': usdt_balance.get('free', 0),
                'used': usdt_balance.get('used', 0),
                'exchange_type': exchange_type
            }
            
        except Exception as e:
            self.log_message(f"❌ Bakiye getirme hatası: {str(e)}")
            return {'total': 0, 'free': 0, 'used': 0, 'exchange_type': exchange_type}
    
    def get_current_price(self, symbol: str) -> float:
        """
        Mevcut fiyatı getirir
        
        Args:
            symbol (str): Trading çifti (örn: BTC/USDT)
        
        Returns:
            float: Mevcut fiyat
        """
        try:
            ticker = self.spot_exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            self.log_message(f"❌ Fiyat getirme hatası: {str(e)}")
            return 0.0
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              target_price: float, stop_loss: float,
                              risk_percent: Optional[float] = None,
                              exchange_type: str = 'futures') -> Dict:
        """
        Pozisyon büyüklüğünü dinamik olarak hesaplar
        
        Args:
            symbol (str): Trading çifti
            entry_price (float): Giriş fiyatı
            target_price (float): Hedef fiyat
            stop_loss (float): Zarar durdur fiyatı
            risk_percent (float): Risk yüzdesi (opsiyonel)
            exchange_type (str): Exchange tipi
        
        Returns:
            Dict: Pozisyon hesaplama sonuçları
        """
        if risk_percent is None:
            risk_percent = self.default_risk_percent
        
        # Risk kontrolü
        risk_percent = min(risk_percent, self.max_risk_percent)
        
        # Bakiye al
        balance = self.get_account_balance(exchange_type)
        available_balance = balance['free']
        
        if available_balance < self.min_order_amount:
            return {
                'success': False,
                'error': f'Yetersiz bakiye: ${available_balance:.2f}',
                'position_size': 0,
                'risk_amount': 0
            }
        
        # Risk tutarını hesapla
        risk_amount = available_balance * (risk_percent / 100)
        
        # Stop loss mesafesini hesapla
        if entry_price > stop_loss:  # Long pozisyon
            stop_distance = entry_price - stop_loss
            direction = 'long'
        else:  # Short pozisyon
            stop_distance = stop_loss - entry_price
            direction = 'short'
        
        # Hedef mesafesi
        if direction == 'long':
            target_distance = target_price - entry_price
        else:
            target_distance = entry_price - target_price
        
        # Risk/Reward oranı
        if target_distance <= 0 or stop_distance <= 0:
            return {
                'success': False,
                'error': 'Geçersiz fiyat seviyeleri',
                'position_size': 0,
                'risk_amount': 0
            }
        
        risk_reward_ratio = target_distance / stop_distance
        
        # Pozisyon büyüklüğünü hesapla
        if exchange_type == 'futures':
            # Futures için kaldıraç kullanarak pozisyon büyüklüğü
            leverage = min(self.max_leverage, max(1, int(available_balance / risk_amount)))
            position_value = risk_amount * leverage / (stop_distance / entry_price)
            quantity = position_value / entry_price
        else:
            # Spot için basit hesaplama
            quantity = risk_amount / stop_distance
            position_value = quantity * entry_price
        
        # Minimum işlem kontrolü
        if position_value < self.min_order_amount:
            quantity = self.min_order_amount / entry_price
            position_value = self.min_order_amount
            actual_risk = (stop_distance / entry_price) * position_value
        else:
            actual_risk = risk_amount
        
        return {
            'success': True,
            'quantity': quantity,
            'position_value': position_value,
            'risk_amount': actual_risk,
            'risk_percent': (actual_risk / available_balance) * 100,
            'risk_reward_ratio': risk_reward_ratio,
            'direction': direction,
            'leverage': leverage if exchange_type == 'futures' else 1,
            'available_balance': available_balance
        }
    
    def get_current_positions(self, exchange_type: str = 'futures') -> Dict:
        """
        Mevcut pozisyonları getirir
        
        Args:
            exchange_type (str): Exchange tipi
        
        Returns:
            Dict: Mevcut pozisyonlar
        """
        try:
            if exchange_type == 'futures':
                positions = self.futures_exchange.fetch_positions()
                active_positions = {}
                
                for pos in positions:
                    if pos['contracts'] != 0:  # Aktif pozisyon
                        symbol = pos['symbol']
                        active_positions[symbol] = {
                            'symbol': symbol,
                            'side': pos['side'],
                            'size': abs(pos['contracts']),
                            'entry_price': pos['entryPrice'],
                            'mark_price': pos['markPrice'],
                            'pnl': pos['unrealizedPnl'],
                            'pnl_percent': pos['percentage']
                        }
                
                return active_positions
            
            else:  # Spot
                balance = self.spot_exchange.fetch_balance()
                spot_positions = {}
                
                for asset, info in balance.items():
                    if asset != 'USDT' and info['total'] > 0:
                        current_price = self.get_current_price(f"{asset}/USDT")
                        value = info['total'] * current_price
                        
                        spot_positions[f"{asset}/USDT"] = {
                            'symbol': f"{asset}/USDT",
                            'side': 'long',
                            'size': info['total'],
                            'entry_price': 0,  # Spot'ta entry price takibi yok
                            'mark_price': current_price,
                            'value': value
                        }
                
                return spot_positions
                
        except Exception as e:
            self.log_message(f"❌ Pozisyon getirme hatası: {str(e)}")
            return {}
    
    def close_position(self, symbol: str, exchange_type: str = 'futures') -> Dict:
        """
        Mevcut pozisyonu kapatır
        
        Args:
            symbol (str): Trading çifti
            exchange_type (str): Exchange tipi
        
        Returns:
            Dict: İşlem sonucu
        """
        try:
            if exchange_type == 'futures':
                # Futures pozisyonunu kapat
                positions = self.get_current_positions('futures')
                
                if symbol not in positions:
                    return {
                        'success': False,
                        'error': f'{symbol} için aktif pozisyon bulunamadı'
                    }
                
                position = positions[symbol]
                side = 'sell' if position['side'] == 'long' else 'buy'
                quantity = position['size']
                
                # Market order ile kapat
                order = self.futures_exchange.create_market_order(
                    symbol, side, quantity, None, None, {
                        'reduceOnly': True
                    }
                )
                
                self.log_message(f"✅ {symbol} pozisyonu kapatıldı: {side} {quantity}")
                
                return {
                    'success': True,
                    'order_id': order['id'],
                    'side': side,
                    'quantity': quantity,
                    'price': order.get('price', 0)
                }
            
            else:  # Spot
                # Spot pozisyonunu sat
                balance = self.spot_exchange.fetch_balance()
                asset = symbol.split('/')[0]
                
                if asset not in balance or balance[asset]['free'] <= 0:
                    return {
                        'success': False,
                        'error': f'{asset} bakiyesi bulunamadı'
                    }
                
                quantity = balance[asset]['free']
                
                order = self.spot_exchange.create_market_order(
                    symbol, 'sell', quantity
                )
                
                self.log_message(f"✅ {symbol} spot pozisyonu satıldı: {quantity}")
                
                return {
                    'success': True,
                    'order_id': order['id'],
                    'side': 'sell',
                    'quantity': quantity,
                    'price': order.get('price', 0)
                }
                
        except Exception as e:
            self.log_message(f"❌ Pozisyon kapatma hatası: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def open_position(self, symbol: str, side: str, entry_price: float,
                     target_price: float, stop_loss: float,
                     risk_percent: Optional[float] = None,
                     exchange_type: str = 'futures') -> Dict:
        """
        Yeni pozisyon açar
        
        Args:
            symbol (str): Trading çifti
            side (str): 'long' veya 'short'
            entry_price (float): Giriş fiyatı
            target_price (float): Hedef fiyat
            stop_loss (float): Zarar durdur
            risk_percent (float): Risk yüzdesi
            exchange_type (str): Exchange tipi
        
        Returns:
            Dict: İşlem sonucu
        """
        try:
            # Pozisyon büyüklüğünü hesapla
            size_calc = self.calculate_position_size(
                symbol, entry_price, target_price, stop_loss, 
                risk_percent, exchange_type
            )
            
            if not size_calc['success']:
                return size_calc
            
            quantity = size_calc['quantity']
            
            # Futures işlemi
            if exchange_type == 'futures':
                # Kaldıracı ayarla
                leverage = size_calc['leverage']
                self.futures_exchange.set_leverage(leverage, symbol)
                
                # Market order ile pozisyon aç
                order = self.futures_exchange.create_market_order(
                    symbol, 'buy' if side == 'long' else 'sell', quantity
                )
                
                # Stop loss ve take profit emirlerini yerleştir
                self._place_stop_orders(symbol, side, quantity, stop_loss, target_price)
                
            else:  # Spot işlemi
                if side == 'short':
                    return {
                        'success': False,
                        'error': 'Spot trading\'de short pozisyon açılamaz'
                    }
                
                # Spot satın alma
                order = self.spot_exchange.create_market_order(
                    symbol, 'buy', quantity
                )
            
            # İşlemi kaydet
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'exchange_type': exchange_type,
                'quantity': quantity,
                'entry_price': order.get('price', entry_price),
                'target_price': target_price,
                'stop_loss': stop_loss,
                'risk_amount': size_calc['risk_amount'],
                'risk_percent': size_calc['risk_percent'],
                'order_id': order['id']
            }
            
            self.trading_history.append(trade_record)
            self.current_positions[symbol] = trade_record
            
            self.log_message(f"✅ {symbol} pozisyonu açıldı:")
            self.log_message(f"   Yön: {side.upper()}")
            self.log_message(f"   Miktar: {quantity:.6f}")
            self.log_message(f"   Giriş: ${entry_price:.6f}")
            self.log_message(f"   Hedef: ${target_price:.6f}")
            self.log_message(f"   Stop: ${stop_loss:.6f}")
            self.log_message(f"   Risk: ${size_calc['risk_amount']:.2f} (%{size_calc['risk_percent']:.1f})")
            
            return {
                'success': True,
                'order_id': order['id'],
                'quantity': quantity,
                'entry_price': order.get('price', entry_price),
                'trade_record': trade_record
            }
            
        except Exception as e:
            self.log_message(f"❌ Pozisyon açma hatası: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _place_stop_orders(self, symbol: str, side: str, quantity: float,
                          stop_loss: float, target_price: float):
        """
        Stop loss ve take profit emirlerini yerleştirir
        """
        try:
            # Stop loss emri
            stop_side = 'sell' if side == 'long' else 'buy'
            
            self.futures_exchange.create_order(
                symbol, 'stop_market', stop_side, quantity, None, None, {
                    'stopPrice': stop_loss,
                    'reduceOnly': True
                }
            )
            
            # Take profit emri
            self.futures_exchange.create_order(
                symbol, 'limit', stop_side, quantity, target_price, None, {
                    'reduceOnly': True
                }
            )
            
            self.log_message(f"✅ Stop emirleri yerleştirildi: SL ${stop_loss:.6f}, TP ${target_price:.6f}")
            
        except Exception as e:
            self.log_message(f"⚠️ Stop emirleri yerleştirilemedi: {str(e)}")
    
    def reverse_position(self, symbol: str, new_side: str, entry_price: float,
                        target_price: float, stop_loss: float,
                        risk_percent: Optional[float] = None,
                        exchange_type: str = 'futures') -> Dict:
        """
        Mevcut pozisyonu kapatıp tersine çevirir
        
        Args:
            symbol (str): Trading çifti
            new_side (str): Yeni pozisyon yönü
            entry_price (float): Yeni giriş fiyatı
            target_price (float): Yeni hedef fiyat
            stop_loss (float): Yeni stop loss
            risk_percent (float): Risk yüzdesi
            exchange_type (str): Exchange tipi
        
        Returns:
            Dict: İşlem sonucu
        """
        try:
            # Mevcut pozisyonları kontrol et
            current_positions = self.get_current_positions(exchange_type)
            
            if symbol in current_positions:
                current_side = current_positions[symbol]['side']
                
                if current_side == new_side:
                    return {
                        'success': False,
                        'error': f'Zaten {new_side} pozisyondasınız'
                    }
                
                self.log_message(f"🔄 {symbol} pozisyonu tersine çevriliyor: {current_side} -> {new_side}")
                
                # 1. Mevcut pozisyonu kapat
                close_result = self.close_position(symbol, exchange_type)
                
                if not close_result['success']:
                    return {
                        'success': False,
                        'error': f"Pozisyon kapatılamadı: {close_result['error']}"
                    }
                
                # Kısa bir bekleme
                time.sleep(1)
                
                # 2. Yeni pozisyon aç
                open_result = self.open_position(
                    symbol, new_side, entry_price, target_price, 
                    stop_loss, risk_percent, exchange_type
                )
                
                if open_result['success']:
                    self.log_message(f"✅ {symbol} pozisyonu başarıyla tersine çevrildi!")
                
                return open_result
            
            else:
                # Mevcut pozisyon yok, direkt yeni pozisyon aç
                return self.open_position(
                    symbol, new_side, entry_price, target_price,
                    stop_loss, risk_percent, exchange_type
                )
                
        except Exception as e:
            self.log_message(f"❌ Pozisyon tersine çevirme hatası: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def execute_signal(self, signal: Dict) -> Dict:
        """
        Trading sinyalini işler ve gerekli işlemleri yapar
        
        Args:
            signal (Dict): Trading sinyali
            
        Signal format:
        {
            'symbol': 'BTC/USDT',
            'action': 'long' | 'short' | 'close',
            'entry_price': float,
            'target_price': float,
            'stop_loss': float,
            'confidence': float,
            'exchange_type': 'spot' | 'futures',
            'risk_percent': float (optional)
        }
        
        Returns:
            Dict: İşlem sonucu
        """
        try:
            symbol = signal['symbol']
            action = signal['action']
            exchange_type = signal.get('exchange_type', 'futures')
            
            self.log_message(f"📡 Sinyal alındı: {symbol} - {action.upper()}")
            
            if action == 'close':
                # Pozisyonu kapat
                return self.close_position(symbol, exchange_type)
            
            elif action in ['long', 'short']:
                # Pozisyon kontrolü ve tersine çevirme
                return self.reverse_position(
                    symbol=symbol,
                    new_side=action,
                    entry_price=signal['entry_price'],
                    target_price=signal['target_price'],
                    stop_loss=signal['stop_loss'],
                    risk_percent=signal.get('risk_percent'),
                    exchange_type=exchange_type
                )
            
            else:
                return {
                    'success': False,
                    'error': f'Geçersiz sinyal aksiyonu: {action}'
                }
                
        except Exception as e:
            self.log_message(f"❌ Sinyal işleme hatası: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_portfolio_summary(self) -> Dict:
        """
        Portföy özetini getirir
        
        Returns:
            Dict: Portföy özeti
        """
        try:
            # Spot bakiye
            spot_balance = self.get_account_balance('spot')
            
            # Futures bakiye
            futures_balance = self.get_account_balance('futures')
            
            # Aktif pozisyonlar
            spot_positions = self.get_current_positions('spot')
            futures_positions = self.get_current_positions('futures')
            
            # Toplam PnL hesapla
            total_pnl = 0
            for pos in futures_positions.values():
                total_pnl += pos.get('pnl', 0)
            
            # Spot değeri
            spot_value = 0
            for pos in spot_positions.values():
                spot_value += pos.get('value', 0)
            
            return {
                'spot_balance': spot_balance,
                'futures_balance': futures_balance,
                'spot_positions': spot_positions,
                'futures_positions': futures_positions,
                'total_spot_value': spot_value,
                'total_futures_pnl': total_pnl,
                'total_positions': len(spot_positions) + len(futures_positions),
                'trading_history_count': len(self.trading_history)
            }
            
        except Exception as e:
            self.log_message(f"❌ Portföy özeti hatası: {str(e)}")
            return {}
    
    def save_trading_data(self, filename: Optional[str] = None):
        """
        Trading verilerini dosyaya kaydeder
        """
        if filename is None:
            filename = f"trading_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            data = {
                'portfolio_summary': self.get_portfolio_summary(),
                'trading_history': self.trading_history,
                'current_positions': self.current_positions,
                'settings': {
                    'default_risk_percent': self.default_risk_percent,
                    'max_risk_percent': self.max_risk_percent,
                    'min_order_amount': self.min_order_amount,
                    'max_leverage': self.max_leverage,
                    'testnet': self.testnet
                },
                'export_time': datetime.now().isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.log_message(f"💾 Trading verileri kaydedildi: {filename}")
            
        except Exception as e:
            self.log_message(f"❌ Veri kaydetme hatası: {str(e)}")

def main():
    """
    Test fonksiyonu
    """
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║               🚀 BINANCE OTOMATIK TRADING SİSTEMİ 🚀             ║
║                                                                    ║
║  Bu sistem LSTM tahminlerinizi Binance'de otomatik işlemlere      ║
║  dönüştürür. Spot ve Futures trading destekler.                   ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    print("⚠️ BU SİSTEM GERÇEK PARA KULLANIR!")
    print("💡 İlk testlerinizi mutlaka TESTNET modunda yapın!")
    
    # API bilgileri
    api_key = input("\n🔑 Binance API Key: ").strip()
    secret_key = input("🔐 Binance Secret Key: ").strip()
    
    use_testnet = input("\n🧪 Testnet kullanılsın mı? (y/n, varsayılan: y): ").strip().lower()
    testnet = use_testnet != 'n'
    
    try:
        # Trader oluştur
        trader = BinanceTrader(api_key, secret_key, testnet)
        
        # Portföy durumunu göster
        portfolio = trader.get_portfolio_summary()
        
        print(f"\n📊 PORTFÖY DURUMU:")
        print(f"💰 Spot USDT: ${portfolio['spot_balance']['free']:.2f}")
        print(f"🚀 Futures USDT: ${portfolio['futures_balance']['free']:.2f}")
        print(f"📈 Aktif Pozisyonlar: {portfolio['total_positions']}")
        
        # Örnek sinyal test et
        test_symbol = input("\n📊 Test trading çifti (örn: BTC/USDT): ").strip()
        if not test_symbol:
            test_symbol = "BTC/USDT"
        
        current_price = trader.get_current_price(test_symbol)
        print(f"💵 {test_symbol} mevcut fiyat: ${current_price:.6f}")
        
        # Örnek trading sinyali
        example_signal = {
            'symbol': test_symbol,
            'action': 'long',
            'entry_price': current_price,
            'target_price': current_price * 1.02,  # %2 hedef
            'stop_loss': current_price * 0.98,     # %2 stop
            'confidence': 75.0,
            'exchange_type': 'futures',
            'risk_percent': 1.0
        }
        
        print(f"\n🔮 Örnek sinyal:")
        print(f"   Sembol: {example_signal['symbol']}")
        print(f"   Aksiyon: {example_signal['action'].upper()}")
        print(f"   Giriş: ${example_signal['entry_price']:.6f}")
        print(f"   Hedef: ${example_signal['target_price']:.6f}")
        print(f"   Stop: ${example_signal['stop_loss']:.6f}")
        
        execute_test = input("\n✅ Bu sinyali test etmek ister misiniz? (y/n): ").strip().lower()
        
        if execute_test == 'y':
            result = trader.execute_signal(example_signal)
            
            if result['success']:
                print("✅ Test işlemi başarılı!")
                print(f"📋 Order ID: {result['order_id']}")
            else:
                print(f"❌ Test işlemi başarısız: {result['error']}")
        
        # Verileri kaydet
        trader.save_trading_data()
        
    except Exception as e:
        print(f"\n❌ Sistem hatası: {str(e)}")
        print("💡 API anahtarlarınızı ve internet bağlantınızı kontrol edin.")

if __name__ == "__main__":
    main() 