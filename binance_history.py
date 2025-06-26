#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance İşlem Geçmişi Modülü

Bu modül Binance API'den kullanıcının gerçek işlem geçmişini çeker
ve sisteme entegre eder.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
from dotenv import load_dotenv

# Environment variables yükle
load_dotenv()

class BinanceHistoryFetcher:
    """Binance işlem geçmişi çekme sınıfı"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Binance history fetcher'ı başlatır
        
        Args:
            api_key (str): Binance API key
            api_secret (str): Binance API secret
            testnet (bool): Testnet kullanımı
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Binance exchange instance oluştur
        try:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'sandbox': testnet,  # Testnet
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'  # Spot trading
                }
            })
            
            # Test connection
            balance = self.exchange.fetch_balance()
            print(f"✅ Binance bağlantısı başarılı ({'Testnet' if testnet else 'Mainnet'})")
            
        except Exception as e:
            print(f"❌ Binance bağlantı hatası: {str(e)}")
            self.exchange = None
    
    def fetch_account_info(self) -> Dict:
        """Hesap bilgilerini çeker"""
        try:
            if not self.exchange:
                return {'error': 'Binance bağlantısı yok'}
            
            balance = self.exchange.fetch_balance()
            
            # Balance response'ını kontrol et
            if not isinstance(balance, dict):
                return {'error': 'Balance response geçersiz format'}
            
            # Sıfır olmayan bakiyeleri filtrele
            non_zero_balances = {}
            
            # Balance dict'inin nasıl olduğunu kontrol et
            for currency, amounts in balance.items():
                if currency not in ['info', 'free', 'used', 'total']:
                    # amounts'ın dict olduğunu ve 'total' key'i olduğunu kontrol et
                    if isinstance(amounts, dict) and 'total' in amounts:
                        if amounts['total'] > 0:
                            non_zero_balances[currency] = amounts
                    elif isinstance(amounts, (int, float)) and amounts > 0:
                        # Eğer amounts direkt sayıysa
                        non_zero_balances[currency] = {
                            'free': amounts,
                            'used': 0,
                            'total': amounts
                        }
            
            account_info = {
                'timestamp': datetime.now().isoformat(),
                'total_balances': len(non_zero_balances),
                'balances': non_zero_balances,
                'trading_enabled': True,
                'account_type': 'SPOT'
            }
            
            return account_info
            
        except Exception as e:
            print(f"❌ Hesap bilgisi alma hatası: {str(e)}")
            return {'error': str(e)}
    
    def fetch_trade_history(self, symbol: str = None, days: int = 30, limit: int = 500) -> List[Dict]:
        """
        İşlem geçmişini çeker
        
        Args:
            symbol (str): Belirli bir sembol (opsiyonel)
            days (int): Kaç gün geriye gidilecek
            limit (int): Maksimum işlem sayısı
        
        Returns:
            List[Dict]: İşlem listesi
        """
        try:
            if not self.exchange:
                return []
            
            # Başlangıç zamanı hesapla
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            trades = []
            
            if symbol:
                # Belirli bir sembol için işlemler
                symbol_trades = self.exchange.fetch_my_trades(symbol, since, limit)
                trades.extend(symbol_trades)
            else:
                # Tüm işlemler - aktif sembolleri al
                markets = self.exchange.load_markets()
                
                # Popüler USDT çiftleri
                priority_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOGE/USDT']
                
                for symbol in priority_symbols:
                    try:
                        if symbol in markets:
                            symbol_trades = self.exchange.fetch_my_trades(symbol, since, 100)
                            trades.extend(symbol_trades)
                            
                            if len(trades) >= limit:
                                break
                            
                    except Exception as e:
                        print(f"⚠️ {symbol} işlemlerini çekerken hata: {str(e)}")
                        continue
            
            # Tarihe göre sırala (en yeni önce)
            trades = sorted(trades, key=lambda x: x['timestamp'], reverse=True)
            
            # Limit uygula
            if len(trades) > limit:
                trades = trades[:limit]
            
            print(f"✅ {len(trades)} işlem geçmişi çekildi")
            return trades
            
        except Exception as e:
            print(f"❌ İşlem geçmişi çekme hatası: {str(e)}")
            return []
    
    def get_trading_summary(self, days: int = 30) -> Dict:
        """
        Trading özeti çıkarır
        
        Args:
            days (int): Kaç günlük özet
        
        Returns:
            Dict: Trading özeti
        """
        try:
            trades = self.fetch_trade_history(days=days)
            
            if not trades:
                return {
                    'total_trades': 0,
                    'total_volume': 0,
                    'total_fees': 0,
                    'buy_trades': 0,
                    'sell_trades': 0,
                    'unique_symbols': 0,
                    'average_trade_size': 0,
                    'date_range': f'{days} gün',
                    'success': True
                }
            
            # İstatistikleri hesapla
            total_trades = len(trades)
            total_volume = sum(trade['cost'] for trade in trades)
            total_fees = sum(trade['fee']['cost'] if trade['fee'] else 0 for trade in trades)
            buy_trades = len([t for t in trades if t['side'] == 'buy'])
            sell_trades = len([t for t in trades if t['side'] == 'sell'])
            unique_symbols = len(set(trade['symbol'] for trade in trades))
            average_trade_size = total_volume / total_trades if total_trades > 0 else 0
            
            # En çok işlem yapılan semboller
            symbol_counts = {}
            for trade in trades:
                symbol = trade['symbol']
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'total_trades': total_trades,
                'total_volume': round(total_volume, 2),
                'total_fees': round(total_fees, 4),
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'unique_symbols': unique_symbols,
                'average_trade_size': round(average_trade_size, 2),
                'date_range': f'{days} gün',
                'top_symbols': top_symbols,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Trading özeti hatası: {str(e)}")
            return {'success': False, 'error': str(e)}

def main():
    """Test fonksiyonu"""
    print("🧪 Binance History Test")
    
    # Environment'den API bilgilerini al
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_SECRET_KEY')
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    
    if not api_key or not api_secret:
        print("❌ Binance API anahtarları bulunamadı!")
        print("🔧 .env dosyasında BINANCE_API_KEY ve BINANCE_SECRET_KEY ayarlayın")
        return
    
    # Fetcher test et
    fetcher = BinanceHistoryFetcher(api_key, api_secret, testnet)
    
    if not fetcher.exchange:
        print("❌ Binance bağlantısı kurulamadı!")
        return
    
    # Hesap bilgilerini test et
    print("\n📊 Hesap Bilgileri:")
    account_info = fetcher.fetch_account_info()
    if 'error' not in account_info:
        print(f"💰 Toplam bakiye sayısı: {account_info['total_balances']}")
        for currency, amounts in list(account_info['balances'].items())[:5]:
            print(f"   {currency}: {amounts['total']}")
    
    # Trade history test et
    print("\n📈 Son İşlemler:")
    trades = fetcher.fetch_trade_history(days=7, limit=10)
    print(f"🔄 {len(trades)} işlem bulundu")
    
    for i, trade in enumerate(trades[:3]):
        print(f"   {i+1}. {trade['symbol']} - {trade['side'].upper()} - {trade['amount']} @ {trade['price']}")
    
    # Trading özeti
    print("\n📋 Trading Özeti:")
    summary = fetcher.get_trading_summary(days=30)
    if summary['success']:
        print(f"📊 Toplam işlem: {summary['total_trades']}")
        print(f"💵 Toplam hacim: ${summary['total_volume']}")
        print(f"💰 Toplam komisyon: ${summary['total_fees']}")
        print(f"🔀 Buy/Sell: {summary['buy_trades']}/{summary['sell_trades']}")

if __name__ == "__main__":
    main()
