#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Service - AI Service ile iletişim kuran Flask uygulaması

Bu servis:
- Web Dashboard sunar
- AI Service ile HTTP üzerinden iletişim kurar
- Veritabanı işlemlerini yönetir
- Authentication sağlar
"""

import os
import sys
import asyncio
import httpx
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_socketio import SocketIO, emit
from flask_login import login_required, login_user, logout_user, current_user
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Web servis modülleri
from binance_trader import BinanceTrader

# Paylaşılan veri katmanı (PostgreSQL + SQLAlchemy, schema-per-tenant multi-tenancy)
from trading_db import (
    TradingDatabase as DatabaseClass,
    AuthManager,
    setup_login_manager,
    SystemPersistence,
    set_current_tenant,
    clear_current_tenant,
    provision_tenant,
    list_tenants,
)
DATABASE_TYPE = "PostgreSQL"
print("🗄️ PostgreSQL (SQLAlchemy, schema-per-tenant) kullanılıyor")

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'trading-dashboard-secret-key-change-in-production')

# SocketIO setup
socketio = SocketIO(app, cors_allowed_origins="*")

# Global objects
database = DatabaseClass() if DatabaseClass else None

# Binance trader setup with environment variables
binance_api_key = os.getenv('BINANCE_API_KEY')
binance_secret_key = os.getenv('BINANCE_SECRET_KEY')
binance_testnet = os.getenv('BINANCE_TESTNET', 'True').lower() == 'true'

if binance_api_key and binance_secret_key:
    binance_trader = BinanceTrader(binance_api_key, binance_secret_key, binance_testnet)
else:
    # Fallback: dummy trader
    class DummyBinanceTrader:
        def __init__(self):
            self.connected = False
        def get_balance(self):
            return {"error": "Binance API keys not configured"}
    binance_trader = DummyBinanceTrader()

system_persistence = SystemPersistence()

# Auth setup - database instance ile birlikte
if database:
    auth_manager = AuthManager(database)
else:
    # Fallback: basit auth manager
    class SimpleAuthManager:
        def authenticate_user(self, username, password):
            # Basit hardcoded authentication
            if username == "admin" and password == "admin":
                from trading_db import User
                return User("1", "admin", "admin_hash")
            return None
    auth_manager = SimpleAuthManager()

setup_login_manager(app, auth_manager)


# ─────────────────────────────────────────────────────────────────────────
# Multi-tenancy: her istekte aktif tenant bağlamını (search_path) kur/temizle
# ─────────────────────────────────────────────────────────────────────────
@app.before_request
def _bind_tenant_context():
    try:
        if current_user.is_authenticated and getattr(current_user, "tenant_schema", None):
            set_current_tenant(current_user.tenant_schema)
        else:
            clear_current_tenant()
    except Exception:
        clear_current_tenant()


@app.teardown_request
def _clear_tenant_context(exc=None):
    clear_current_tenant()


# ─────────────────────────────────────────────────────────────────────────
# Platform admin: tenant provisioning (yeni tenant + şema oluşturma)
# ─────────────────────────────────────────────────────────────────────────
@app.route("/api/tenants", methods=["GET", "POST"])
@login_required
def api_tenants():
    if getattr(current_user, "role", None) != "platform_admin":
        return jsonify({"success": False, "error": "Yetkisiz (platform_admin gerekli)"}), 403
    if request.method == "GET":
        return jsonify({"success": True, "tenants": list_tenants()})
    data = request.get_json(force=True, silent=True) or {}
    slug = data.get("slug")
    if not slug:
        return jsonify({"success": False, "error": "slug gerekli"}), 400
    result = provision_tenant(
        slug,
        name=data.get("name"),
        admin_username=data.get("admin_username"),
        admin_password=data.get("admin_password"),
        admin_email=data.get("admin_email"),
    )
    return jsonify(result), (200 if result.get("success") else 500)


# AI Service Client
class AIServiceClient:
    """AI Service ile iletişim sağlayan client"""
    
    def __init__(self):
        self.base_url = os.getenv('AI_SERVICE_URL', 'http://localhost:8000')
        self.api_key = os.getenv('AI_SERVICE_API_KEY')
        self.timeout = 300  # 5 minutes timeout for training
        
    async def analyze_coin(self, coin_symbol: str, analysis_type: str = "comprehensive", 
                          use_news: bool = True, use_whale: bool = True) -> Dict:
        """
        AI Service'ten coin analizi iste
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "coin_symbol": coin_symbol,
                    "analysis_type": analysis_type,
                    "use_news": use_news,
                    "use_whale": use_whale
                }
                
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                response = await client.post(
                    f"{self.base_url}/analyze",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "success": False,
                        "error": f"AI Service hatası: {response.status_code} - {response.text}"
                    }
                    
        except httpx.TimeoutException:
            return {"success": False, "error": "AI Service timeout"}
        except Exception as e:
            return {"success": False, "error": f"AI Service bağlantı hatası: {str(e)}"}
    
    async def train_model(self, coin_symbol: str, training_type: str = "comprehensive",
                         is_fine_tune: bool = False, epochs: int = None, data_days: int = None) -> Dict:
        """
        AI Service'ten model eğitimi iste
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "coin_symbol": coin_symbol,
                    "training_type": training_type,
                    "is_fine_tune": is_fine_tune,
                    "epochs": epochs,
                    "data_days": data_days
                }
                
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                response = await client.post(
                    f"{self.base_url}/train",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "success": False,
                        "error": f"Training hatası: {response.status_code} - {response.text}"
                    }
                    
        except httpx.TimeoutException:
            return {"success": False, "error": "Training timeout"}
        except Exception as e:
            return {"success": False, "error": f"Training bağlantı hatası: {str(e)}"}
    
    async def get_model_status(self, coin_symbol: str) -> Dict:
        """
        AI Service'ten model durumunu al
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                response = await client.get(
                    f"{self.base_url}/models/{coin_symbol}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"Model status hatası: {response.status_code}"}
                    
        except Exception as e:
            return {"error": f"Model status bağlantı hatası: {str(e)}"}
    
    async def health_check(self) -> Dict:
        """
        AI Service sağlık kontrolü
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

# Global AI client
ai_client = AIServiceClient()

# CoinMonitor class - sadece web işlemleri için
class WebCoinMonitor:
    """Web servisi için coin izleme sınıfı - AI Service kullanır"""
    
    def __init__(self):
        # SystemPersistence'den state yükleme
        try:
            if hasattr(system_persistence, 'load_state'):
                self.tracked_coins = system_persistence.load_state('tracked_coins', [])
            else:
                # Fallback: basit dosya tabanlı yükleme
                import json
                import os
                try:
                    if os.path.exists('tracked_coins.json'):
                        with open('tracked_coins.json', 'r') as f:
                            self.tracked_coins = json.load(f)
                    else:
                        self.tracked_coins = []
                except:
                    self.tracked_coins = []
        except Exception as e:
            print(f"⚠️ Tracked coins yükleme hatası: {e}")
            self.tracked_coins = []
            
        print(f"📊 Tracked coins loaded: {self.tracked_coins}")
    
    async def analyze_coin_web(self, coin_symbol: str, analysis_type: str = "comprehensive"):
        """
        Web arayüzü için coin analizi - AI Service kullanır
        """
        try:
            print(f"🌐 Web analizi başlıyor: {coin_symbol} - Tip: {analysis_type}")
            
            # AI Service'ten analiz iste
            analysis_result = await ai_client.analyze_coin(
                coin_symbol=coin_symbol,
                analysis_type=analysis_type,
                use_news=True,
                use_whale=True
            )
            
            if not analysis_result.get('success', False):
                return {
                    'success': False,
                    'error': analysis_result.get('error', 'AI Service analiz hatası'),
                    'coin_symbol': coin_symbol
                }
            
            # Veritabanına kaydet
            if database:
                try:
                    analysis_id = database.save_analysis_result(
                        coin_symbol=coin_symbol,
                        prediction_result=analysis_result.get('predictions', {}),
                        news_analysis=analysis_result.get('news_analysis'),
                        whale_analysis=analysis_result.get('whale_analysis'),
                        model_type=analysis_type.upper()
                    )
                    analysis_result['analysis_id'] = analysis_id
                except Exception as db_error:
                    print(f"⚠️ Veritabanı kayıt hatası: {db_error}")
            
            # Web formatına dönüştür
            web_result = self._format_for_web(analysis_result)
            
            print(f"✅ Web analizi tamamlandı: {coin_symbol}")
            return web_result
            
        except Exception as e:
            print(f"❌ Web analizi hatası: {e}")
            return {
                'success': False,
                'error': str(e),
                'coin_symbol': coin_symbol
            }
    
    async def train_coin_web(self, coin_symbol: str, training_type: str = "comprehensive", 
                           is_fine_tune: bool = False):
        """
        Web arayüzü için model eğitimi - AI Service kullanır
        """
        try:
            print(f"🎯 Web eğitimi başlıyor: {coin_symbol} - Tip: {training_type}")
            
            # AI Service'ten eğitim iste
            training_result = await ai_client.train_model(
                coin_symbol=coin_symbol,
                training_type=training_type,
                is_fine_tune=is_fine_tune
            )
            
            if training_result.get('success', False):
                # Tracked coins'e ekle
                if coin_symbol not in self.tracked_coins:
                    self.tracked_coins.append(coin_symbol)
                    # SystemPersistence save_state kontrolü
                    try:
                        if hasattr(system_persistence, 'save_state'):
                            system_persistence.save_state('tracked_coins', self.tracked_coins)
                        else:
                            # Fallback: basit dosya kaydetme
                            import json
                            with open('tracked_coins.json', 'w') as f:
                                json.dump(self.tracked_coins, f)
                        print(f"✅ {coin_symbol} tracked coins'e eklendi")
                    except Exception as save_error:
                        print(f"⚠️ Tracked coins kaydetme hatası: {save_error}")
            
            return training_result
            
        except Exception as e:
            print(f"❌ Web eğitimi hatası: {e}")
            return {'success': False, 'error': str(e)}
    
    def _format_for_web(self, ai_result: Dict) -> Dict:
        """
        AI Service sonucunu web formatına dönüştürür
        """
        try:
            # AI Service formatından web formatına dönüşüm
            web_result = {
                'success': ai_result.get('success', False),
                'coin_symbol': ai_result.get('coin_symbol'),
                'model_type': ai_result.get('model_type'),
                'timestamp': ai_result.get('timestamp'),
                'current_price': ai_result.get('current_price', 0),
                
                # Backward compatibility için mevcut format
                'prediction': ai_result.get('predictions', {}),
                'technical_analysis': ai_result.get('technical_analysis', {}),
                'news_analysis': ai_result.get('news_analysis', {}),
                'whale_analysis': ai_result.get('whale_analysis', {}),
                
                # Yeni multi-model format
                'multi_model_results': {
                    'lstm_analysis': {
                        'success': True,
                        'prediction': ai_result.get('predictions', {})
                    },
                    'technical_analysis': ai_result.get('technical_analysis', {}),
                    'news_analysis': ai_result.get('news_analysis', {}),
                    'whale_analysis': ai_result.get('whale_analysis', {})
                }
            }
            
            return web_result
            
        except Exception as e:
            print(f"⚠️ Web format dönüşüm hatası: {e}")
            return ai_result

# Global monitor
coin_monitor = WebCoinMonitor()

# Flask routes
@app.route('/')
def index():
    """Ana sayfa"""
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Ana dashboard"""
    try:
        # Son analizleri al
        recent_analyses = []
        if database:
            recent_analyses = database.get_analysis_history(limit=10)
        
        # Portfolio özeti
        portfolio = {
            'current_value': 0.0,
            'total_pnl': 0.0,
            'open_positions': 0
        }
        
        # AI Service sağlık kontrolü
        ai_status = {"status": "unknown"}
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ai_status = loop.run_until_complete(ai_client.health_check())
            loop.close()
        except Exception as e:
            ai_status = {"status": "error", "error": str(e)}
        
        return render_template('dashboard.html', 
                             analyses=recent_analyses,
                             tracked_coins=coin_monitor.tracked_coins,
                             ai_service_status=ai_status,
                             portfolio=portfolio)
                             
    except Exception as e:
        flash(f'Dashboard yüklenirken hata: {str(e)}', 'error')
        return render_template('dashboard.html', 
                             analyses=[], 
                             tracked_coins=[],
                             portfolio={'current_value': 0, 'total_pnl': 0, 'open_positions': 0})

@app.route('/analyze_coin')
@login_required
def analyze_coin_page():
    """Coin analiz sayfası"""
    coin_symbol = request.args.get('coin', '')
    return render_template('analyze_coin.html', coin_symbol=coin_symbol)

@app.route('/api/analyze_coin', methods=['POST'])
@login_required
def api_analyze_coin():
    """API: Coin analizi"""
    try:
        # Check if request has JSON content-type
        if request.is_json:
            data = request.get_json()
        else:
            # Try to parse JSON from request data anyway
            try:
                data = request.get_json(force=True)
            except Exception:
                return jsonify({
                    'success': False, 
                    'error': 'Request must be JSON. Content-Type should be application/json'
                }), 415
        
        if not data:
            return jsonify({'success': False, 'error': 'JSON data gerekli'}), 400
            
        coin_symbol = data.get('coin_symbol', '').upper()
        analysis_type = data.get('analysis_type', 'comprehensive')
        
        if not coin_symbol:
            return jsonify({'success': False, 'error': 'Coin sembolü gerekli'}), 400
        
        # Async analizi sync çalıştır
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            coin_monitor.analyze_coin_web(coin_symbol, analysis_type)
        )
        loop.close()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train_coin', methods=['POST'])
@login_required
def api_train_coin():
    """API: Model eğitimi"""
    try:
        # Check if request has JSON content-type
        if request.is_json:
            data = request.get_json()
        else:
            # Try to parse JSON from request data anyway
            try:
                data = request.get_json(force=True)
            except Exception:
                return jsonify({
                    'success': False, 
                    'error': 'Request must be JSON. Content-Type should be application/json'
                }), 415
        
        if not data:
            return jsonify({'success': False, 'error': 'JSON data gerekli'}), 400
            
        coin_symbol = data.get('coin_symbol', '').upper()
        training_type = data.get('training_type', 'comprehensive')
        is_fine_tune = data.get('is_fine_tune', False)
        
        if not coin_symbol:
            return jsonify({'success': False, 'error': 'Coin sembolü gerekli'}), 400
        
        # Async eğitimi sync çalıştır
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            coin_monitor.train_coin_web(coin_symbol, training_type, is_fine_tune)
        )
        loop.close()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ai_service_status')
@login_required
def api_ai_service_status():
    """API: AI Service durum kontrolü"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        status = loop.run_until_complete(ai_client.health_check())
        loop.close()
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/api/model_status/<coin_symbol>')
@login_required
def api_model_status(coin_symbol):
    """API: Model durumu"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        status = loop.run_until_complete(ai_client.get_model_status(coin_symbol.upper()))
        loop.close()
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)})

# Portfolio ve diğer mevcut endpoint'ler
@app.route('/portfolio')
@login_required
def portfolio():
    """Portfolio sayfası"""
    try:
        # Mevcut portfolio logic'i koru
        summary = {'total_positions': 0, 'unrealized_pnl': 0}
        positions = []
        
        if database:
            positions = database.get_user_positions(current_user.get_id())
            
        return render_template('portfolio.html', 
                             summary=summary, 
                             positions=positions)
                             
    except Exception as e:
        flash(f'Portfolio yüklenirken hata: {str(e)}', 'error')
        return render_template('portfolio.html', summary={}, positions=[])

@app.route('/settings')
@login_required
def settings():
    """Ayarlar sayfası"""
    try:
        return render_template('settings.html')
    except Exception as e:
        flash(f'Ayarlar yüklenirken hata: {str(e)}', 'error')
        return render_template('settings.html')

# Login routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login sayfası"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = auth_manager.authenticate_user(username, password)
        if user:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Geçersiz kullanıcı adı veya şifre', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Logout"""
    logout_user()
    return redirect(url_for('login'))

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║           🌐 CRYPTO TRADING WEB SERVICE 🌐                       ║
║                                                                    ║
║  AI Service ile iletişim kuran web arayüzü başlatılıyor...       ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    ai_service_url = os.getenv('AI_SERVICE_URL', 'http://localhost:8000')
    web_port = int(os.getenv('WEB_PORT', 5001))
    web_host = os.getenv('WEB_HOST', '0.0.0.0')
    web_debug = os.getenv('WEB_DEBUG', 'True').lower() == 'true'
    
    print(f"🔗 AI Service URL: {ai_service_url}")
    print(f"🗄️ Database Type: {DATABASE_TYPE}")
    print(f"🌐 Web Service: http://{web_host}:{web_port}")
    
    socketio.run(
        app,
        host=web_host,
        port=web_port,
        debug=web_debug
    )
