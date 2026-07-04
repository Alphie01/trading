#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kripto Trading Dashboard Web Uygulaması

Bu web arayüzü şunları sağlar:
- Çoklu coin izleme
- Gerçek zamanlı analiz
- İşlem geçmişi takibi
- Kar/zarar analizi
- Portfolio yönetimi
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_socketio import SocketIO, emit
from flask_login import login_required, login_user, logout_user, current_user
import json
import threading
import time
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
import logging
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Kendi modüllerimiz
from data_fetcher import CryptoDataFetcher
from data_preprocessor import CryptoDataPreprocessor
from predictor import CryptoPricePredictor
from news_analyzer import CryptoNewsAnalyzer
from whale_tracker import CryptoWhaleTracker
from binance_trader import BinanceTrader
from auto_trader_integration import LSTMAutoTrader
# Paylaşılan veri katmanı (PostgreSQL + SQLAlchemy, schema-per-tenant multi-tenancy)
from trading_db import (
    TradingDatabase as DatabaseClass,
    AuthManager,
    setup_login_manager,
    SystemPersistence,
    set_current_tenant,
    clear_current_tenant,
)
DATABASE_TYPE = "PostgreSQL"
print("🗄️ PostgreSQL (SQLAlchemy, schema-per-tenant) kullanılıyor")

# Performans instrumentation (config ile aç/kapa: ANALYSIS_TIMING_ENABLED)
try:
    from perf_timing import analysis_timer, step as perf_step, cache_event
except Exception:  # perf_timing yoksa no-op fallback
    from contextlib import contextmanager as _cm
    @_cm
    def analysis_timer(label):
        yield None
    @_cm
    def perf_step(name):
        yield
    def cache_event(kind, key, hit):
        pass

try:
    from model_cache import CachedModelManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# Flask uygulaması - Environment variables ile konfigürasyon
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'crypto_trading_dashboard_2024_change_this')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'

# SocketIO konfigürasyonu
socketio = SocketIO(app, cors_allowed_origins="*")

# Global değişkenler
db = DatabaseClass()
data_fetcher = CryptoDataFetcher()
persistence = SystemPersistence()
analysis_queue = queue.Queue()
active_analyses = {}
monitoring_active = False

# Training scheduler'ı initialize et
try:
    from training_scheduler import init_scheduler
    training_scheduler = init_scheduler(
        schedule_day=os.getenv('TRAINING_SCHEDULE_DAY', 'sunday'),
        schedule_time=os.getenv('TRAINING_SCHEDULE_TIME', '02:00'),
        enable_notifications=True
    )
    # Scheduler'ı başlat
    training_scheduler.start_scheduler()
    print("✅ Training Scheduler başlatıldı ve aktif!")
except Exception as scheduler_init_error:
    print(f"⚠️ Training Scheduler başlatma hatası: {scheduler_init_error}")
    training_scheduler = None

# Authentication setup
auth_manager = AuthManager(db)
login_manager = setup_login_manager(app, auth_manager)


# Multi-tenancy: her istekte aktif tenant bağlamını (search_path) kur/temizle
@app.before_request
def _bind_tenant_context():
    try:
        from flask_login import current_user
        if current_user.is_authenticated and getattr(current_user, "tenant_schema", None):
            set_current_tenant(current_user.tenant_schema)
        else:
            clear_current_tenant()
    except Exception:
        clear_current_tenant()


# Tüm template'lere çalışma modu (Binance testnet/live) enjekte edilir → topbar rozeti gerçek modu gösterir.
@app.context_processor
def _inject_mode():
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    exchange_type = os.getenv('AUTO_EXCHANGE_TYPE', 'spot')
    return {
        'binance_testnet': testnet,
        'binance_mode': 'TESTNET' if testnet else 'LIVE',
        'exchange_type': exchange_type.upper(),
    }


@app.teardown_request
def _clear_tenant_context(exc=None):
    clear_current_tenant()


# ───────────────────────────────────────────────────────────────────────────
# Automation Engine API (Faz 4-7) — lazy import (ağır modüller yalnız çağrıda)
# ───────────────────────────────────────────────────────────────────────────
@app.route('/api/automation/status')
@login_required
def api_automation_status():
    try:
        from automation.scheduler import get_scheduler
        from automation.config import AutomationConfig as AC
        return jsonify({'success': True, 'status': get_scheduler().status(), 'config': {
            'min_24h_volume_usdt': AC.MIN_24H_VOLUME_USDT,
            'interval_minutes': AC.SCAN_INTERVAL_MINUTES,
            'trade_enabled': AC.TRADE_ENABLED,
            'demo_mode': AC.DEMO_MODE,
        }})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/automation/discover', methods=['POST'])
@login_required
def api_automation_discover():
    try:
        from automation.engine import run_discovery
        out = run_discovery(persist=True)
        return jsonify({'success': True,
                        'summary': {k: out[k] for k in ('scanned', 'passed', 'watchlisted')},
                        'top': out['top']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/automation/research', methods=['POST'])
@login_required
def api_automation_research():
    try:
        from automation.engine import run_research
        # allow_trades None → AUTO_TRADE_ENABLED (fail-closed; trade yalnız onaylı + AUTO açık ile)
        out = run_research(persist=True)
        return jsonify({'success': True,
                        'summary': {k: out[k] for k in ('researched', 'watchlisted')},
                        'signals': out['signals'][:20]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/automation/candidates')
@login_required
def api_automation_candidates():
    try:
        from automation import repository as arepo
        return jsonify({'success': True,
                        'candidates': arepo.get_candidates(status=request.args.get('status'))})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/automation/scores')
@login_required
def api_automation_scores():
    try:
        from automation import repository as arepo
        return jsonify({'success': True, 'scores': arepo.get_latest_scores()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/automation/signals')
@login_required
def api_automation_signals():
    try:
        from flask_login import current_user
        from automation import tenant_repo
        schema = getattr(current_user, 'tenant_schema', None) or tenant_repo.default_tenant_schema()
        return jsonify({
            'success': True,
            'signals': tenant_repo.recent_signals(schema) if schema else [],
            'alerts': tenant_repo.recent_alerts(schema) if schema else [],
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/automation/coin/<symbol>')
@login_required
def api_automation_coin(symbol):
    try:
        from automation import repository as arepo
        return jsonify({
            'success': True,
            'detail': arepo.get_coin_detail(symbol),
            'history': arepo.get_score_history(symbol),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/automation/coin/<symbol>/research', methods=['POST'])
@login_required
def api_automation_coin_research(symbol):
    try:
        from automation.engine import run_research
        out = run_research(symbols=[symbol.upper()], persist=True)
        from automation import repository as arepo
        return jsonify({'success': True, 'detail': arepo.get_coin_detail(symbol),
                        'history': arepo.get_score_history(symbol),
                        'signal': (out.get('signals') or [None])[0]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/automation/settings', methods=['GET', 'POST'])
@login_required
def api_automation_settings():
    try:
        from automation.config import AutomationConfig as AC
        from automation import settings_store
        if request.method == 'GET':
            # kaydedilmiş override'ları uygula + güncel değerleri döndür
            saved = settings_store.load_automation_settings()
            if saved:
                AC.apply_overrides(saved)
            return jsonify({'success': True, 'settings': AC.current_settings()})
        data = request.get_json(force=True, silent=True) or {}
        ok = settings_store.save_automation_settings(data)
        if not ok:
            return jsonify({'success': False, 'error': 'Kaydedilemedi (default tenant yok olabilir)'}), 500
        return jsonify({'success': True, 'settings': AC.current_settings(), 'message': 'Otomasyon ayarları kaydedildi'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ───────────────────────────────────────────────────────────────────────────
# Simulation / Paper Trading API (SANAL — canlı Binance işlemi YOK)
# ───────────────────────────────────────────────────────────────────────────
def _sim_schema():
    from flask_login import current_user
    schema = getattr(current_user, 'tenant_schema', None)
    if schema:
        return schema
    from automation import tenant_repo
    return tenant_repo.default_tenant_schema()


@app.route('/api/simulation/runs', methods=['GET', 'POST'])
@login_required
def api_simulation_runs():
    try:
        from automation import simulation
        schema = _sim_schema()
        if not schema:
            return jsonify({'success': False, 'error': 'tenant bulunamadı'}), 400
        if request.method == 'GET':
            return jsonify({'success': True, 'simulations': simulation.list_simulations(schema)})
        from flask_login import current_user
        data = request.get_json(force=True, silent=True) or {}
        res = simulation.create_simulation(schema, data, created_by=getattr(current_user, 'id', None))
        return jsonify(res), (200 if res.get('success') else 400)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulation/runs/<int:sid>')
@login_required
def api_simulation_get(sid):
    try:
        from automation import simulation
        schema = _sim_schema()
        sim = simulation.get_simulation(schema, sid) if schema else None
        if not sim:
            return jsonify({'success': False, 'error': 'bulunamadı'}), 404
        return jsonify({'success': True, 'simulation': sim, 'metrics': simulation.get_metrics(schema, sid)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulation/runs/<int:sid>/<action>', methods=['POST'])
@login_required
def api_simulation_action(sid, action):
    try:
        from automation import simulation
        schema = _sim_schema()
        if not schema:
            return jsonify({'success': False, 'error': 'tenant bulunamadı'}), 400
        mapping = {'pause': 'PAUSED', 'resume': 'RUNNING', 'stop': 'STOPPED'}
        if action not in mapping:
            return jsonify({'success': False, 'error': 'geçersiz aksiyon'}), 400
        return jsonify(simulation.set_status(schema, sid, mapping[action]))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulation/runs/<int:sid>', methods=['DELETE'])
@login_required
def api_simulation_delete(sid):
    try:
        from automation import simulation
        schema = _sim_schema()
        return jsonify(simulation.delete_simulation(schema, sid) if schema else {'success': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulation/runs/<int:sid>/positions')
@login_required
def api_simulation_positions(sid):
    try:
        from automation.simulation import repository as srepo
        schema = _sim_schema()
        status = request.args.get('status')
        if status:
            pos = srepo.get_positions(schema, sid, status=status.upper())
        else:
            pos = srepo.get_open_positions(schema, sid) + srepo.get_closed_positions(schema, sid)
        return jsonify({'success': True, 'positions': pos})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulation/runs/<int:sid>/trades')
@login_required
def api_simulation_trades(sid):
    try:
        from automation.simulation import repository as srepo
        return jsonify({'success': True, 'trades': srepo.get_trades(_sim_schema(), sid)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulation/runs/<int:sid>/metrics')
@login_required
def api_simulation_metrics(sid):
    try:
        from automation import simulation
        return jsonify({'success': True, 'metrics': simulation.get_metrics(_sim_schema(), sid)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulation/runs/<int:sid>/reports/<period>')
@login_required
def api_simulation_reports(sid, period):
    try:
        from automation import simulation
        if period.upper() not in ('DAILY', 'WEEKLY', 'MONTHLY'):
            return jsonify({'success': False, 'error': 'period DAILY/WEEKLY/MONTHLY olmalı'}), 400
        return jsonify({'success': True, 'reports': simulation.get_period_reports(_sim_schema(), sid, period.upper())})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulation/runs/<int:sid>/coin-performance')
@login_required
def api_simulation_coin_perf(sid):
    try:
        from automation import simulation
        return jsonify({'success': True, 'coins': simulation.get_coin_performance(_sim_schema(), sid)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulation/runs/<int:sid>/signal-accuracy')
@login_required
def api_simulation_signal_accuracy(sid):
    try:
        from automation import simulation
        return jsonify({'success': True, 'accuracy': simulation.get_signal_accuracy(_sim_schema(), sid)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Settings formunu GERÇEK backend'e bağla (eski settings.html mock idi → system_state'e yaz)
@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def api_settings():
    try:
        if request.method == 'GET':
            return jsonify({'success': True, 'settings': db.load_system_state('user_settings', {})})
        data = request.get_json(force=True, silent=True) or {}
        db.save_system_state('user_settings', data)
        return jsonify({'success': True, 'message': 'Ayarlar kaydedildi'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoinMonitor:
    """Coinleri izleyen sınıf - Environment variables ve persistence destekli"""
    
    def __init__(self):
        self.db = db
        self.fetcher = data_fetcher
        self.persistence = persistence
        self.cache_manager = CachedModelManager() if CACHE_AVAILABLE else None
        self.news_analyzer = None
        self.whale_tracker = None
        self.auto_trader = None
        self.running = False
        self.monitoring_interval = int(os.getenv('DEFAULT_MONITORING_INTERVAL', '15'))
        
        # Auto-initialize from environment variables
        self._auto_setup_from_env()
        
    def _auto_setup_from_env(self):
        """Environment variables'dan otomatik ayar"""
        try:
            # News API setup
            newsapi_key = os.getenv('NEWSAPI_KEY')
            newsapi_enabled = os.getenv('NEWSAPI_ENABLED', 'true').lower() == 'true'
            
            if newsapi_key and newsapi_enabled:
                self.news_analyzer = CryptoNewsAnalyzer(newsapi_key)
                logger.info("📰 Haber analizi otomatik aktif (environment)")
            
            # Whale Alert setup
            whale_key = os.getenv('WHALE_ALERT_API_KEY')
            whale_enabled = os.getenv('WHALE_TRACKER_ENABLED', 'true').lower() == 'true'
            
            if whale_key and whale_enabled:
                self.whale_tracker = CryptoWhaleTracker(whale_key)
                logger.info("🐋 Whale tracker otomatik aktif (environment)")
            
            # Auto trading setup
            trading_enabled = os.getenv('AUTO_TRADING_ENABLED', 'false').lower() == 'true'
            binance_key = os.getenv('BINANCE_API_KEY')
            binance_secret = os.getenv('BINANCE_SECRET_KEY')
            testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
            
            if trading_enabled and binance_key and binance_secret:
                trader = BinanceTrader(binance_key, binance_secret, testnet)
                self.auto_trader = LSTMAutoTrader(trader)
                logger.info("🤖 Otomatik trading otomatik aktif (environment)")
                
        except Exception as e:
            logger.error(f"Environment auto-setup hatası: {str(e)}")
        
    def setup_analyzers(self, newsapi_key=None, whale_api_key=None):
        """Analiz araçlarını manuel ayarlar"""
        try:
            if newsapi_key:
                self.news_analyzer = CryptoNewsAnalyzer(newsapi_key)
                logger.info("📰 Haber analizi aktif")
            
            if whale_api_key:
                self.whale_tracker = CryptoWhaleTracker(whale_api_key)
                logger.info("🐋 Whale tracker aktif")
                
        except Exception as e:
            logger.error(f"Analyzer setup hatası: {str(e)}")
    
    def setup_auto_trader(self, api_key, api_secret, testnet=True):
        """Otomatik trading ayarlar"""
        try:
            trader = BinanceTrader(api_key, api_secret, testnet)
            self.auto_trader = LSTMAutoTrader(trader)
            logger.info("🤖 Otomatik trading aktif")
            return True
        except Exception as e:
            logger.error(f"Auto trader setup hatası: {str(e)}")
            return False
    
    def analyze_coin(self, coin_symbol):
        """Tek coin analizi"""
        try:
            logger.info(f"Buradayız Baba")
            logger.info(f"🔍 {coin_symbol} analizi başlıyor...")
            
            # Environment variables'dan ayarları al
            training_days = int(os.getenv('LSTM_TRAINING_DAYS', 100))
            
            # Veri çek
            df = self.fetcher.fetch_ohlcv_data(coin_symbol, days=training_days)
            if df is None:
                return {'success': False, 'error': 'Veri çekilemedi'}
            
            # Environment variables'dan ayarları al
            news_days = int(os.getenv('NEWS_ANALYSIS_DAYS', 7))
            
            # Haber analizi
            news_analysis = None
            if self.news_analyzer:
                news_data = self.news_analyzer.fetch_all_news(coin_symbol, days=news_days)
                if news_data:
                    news_df = self.news_analyzer.analyze_news_sentiment_batch(news_data)
                    if not news_df.empty:
                        news_analysis = {
                            'news_sentiment': news_df['overall_sentiment'].mean(),
                            'news_count': len(news_df)
                        }
            
            # Whale analizi
            whale_analysis = None
            if self.whale_tracker:
                whale_txs = self.whale_tracker.fetch_whale_alert_transactions(coin_symbol, 24)
                if whale_txs:
                    whale_data = self.whale_tracker.analyze_whale_transactions(whale_txs)
                    whale_analysis = {
                        'whale_activity_score': whale_data.get('whale_activity_score', 0),
                        'total_volume': whale_data.get('total_volume', 0)
                    }
            
            # Veri ön işleme
            preprocessor = CryptoDataPreprocessor()
            sentiment_df = None
            whale_features = None
            
            if news_analysis:
                # Basit sentiment özellikleri oluştur (timezone safe)
                last_date = pd.to_datetime(df.index[-1], utc=True).tz_localize(None)
                sentiment_df = pd.DataFrame({
                    'date': [last_date],
                    'daily_sentiment': [news_analysis['news_sentiment']]
                }).set_index('date')
            
            if whale_analysis:
                whale_features = {
                    'whale_activity_score': whale_analysis['whale_activity_score'],
                    'whale_volume': whale_analysis['total_volume']
                }
            
            processed_df = preprocessor.prepare_data(df, True, sentiment_df, whale_features)
            
            # Model config - Environment variables'dan ayarları al
            epochs = int(os.getenv('LSTM_EPOCHS', 30))
            
            model_config = {
                'sequence_length': 60,
                'lstm_units': [50, 50, 50],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'epochs': epochs,
                'batch_size': 32,
                'use_technical_indicators': True,
                'use_news_analysis': news_analysis is not None,
                'use_whale_analysis': whale_analysis is not None,
                'training_days': training_days,
                'news_days': news_days
            }
            
            print(f"🔧 Model Config: Epochs={epochs}, Training Days={training_days}, News Days={news_days}")
            
            # Model al/eğit
            if self.cache_manager:
                model, preprocessor_cached, training_info = self.cache_manager.get_or_train_model(
                    coin_symbol, processed_df, model_config, preprocessor
                )
            else:
                # Manuel eğitim (cache yok)
                scaled_data = preprocessor.scale_data(processed_df)
                X, y = preprocessor.create_sequences(scaled_data, 60)
                X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
                
                from lstm_model import CryptoLSTMModel
                model = CryptoLSTMModel(60, X_train.shape[2])
                model.build_model([50, 50, 50], 0.2, 0.001)
                model.train_model(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=32)
            
            # Tahmin yap
            predictor = CryptoPricePredictor(model, preprocessor, self.news_analyzer, self.whale_tracker)
            prediction_result = predictor.predict_next_price(processed_df, 60)
            
            if prediction_result is None:
                return {'success': False, 'error': 'Tahmin yapılamadı'}
            
            # Yigit analizi
            yigit_analysis = predictor.analyze_yigit_signals(processed_df)
            
            # Teknik analiz sonuçları oluştur
            technical_analysis = self._generate_technical_analysis(processed_df, prediction_result)
            
            # Sonuçları kaydet
            analysis_id = self.db.save_analysis_result(
                coin_symbol, prediction_result, news_analysis, whale_analysis, yigit_analysis
            )
            
            # Otomatik trading (eğer aktifse)
            trade_signal = None
            if self.auto_trader:
                try:
                    signal = self.auto_trader.generate_trading_signal(
                        coin_symbol, prediction_result, news_analysis, whale_analysis, yigit_analysis
                    )
                    if signal and signal['action'] != 'HOLD':
                        trade_result = self.auto_trader.execute_trade_signal(coin_symbol, signal)
                        if trade_result['success']:
                            # İşlemi veritabanına kaydet
                            self.db.record_trade(
                                coin_symbol, signal['action'], signal['target_price'],
                                signal['quantity'], prediction_result['confidence'],
                                news_analysis.get('news_sentiment') if news_analysis else None,
                                whale_analysis.get('whale_activity_score') if whale_analysis else None,
                                yigit_analysis.get('current_signal') if yigit_analysis else None,
                                f"Auto trade: {signal['reason']}",
                                is_simulated=False
                            )
                            trade_signal = signal
                except Exception as e:
                    logger.error(f"Auto trading hatası: {str(e)}")
            
            result = {
                'success': True,
                'coin_symbol': coin_symbol,
                'prediction': prediction_result,
                'technical_analysis': technical_analysis,
                'news_analysis': news_analysis,
                'whale_analysis': whale_analysis,
                'yigit_analysis': yigit_analysis,
                'trade_signal': trade_signal,
                'analysis_id': analysis_id,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ {coin_symbol} analizi tamamlandı")
            return result
            
        except Exception as e:
            logger.error(f"❌ {coin_symbol} analiz hatası: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def analyze_coin_multi_model(self, coin_symbol):
        """
        Multi-model coin analizi (LSTM + DQN + Hybrid) - İlk eğitim kontrolü ile
        
        Args:
            coin_symbol (str): Coin sembolü
        
        Returns:
            dict: Multi-model analiz sonuçları
        """
        try:
            logger.info(f"🚀 {coin_symbol} Multi-Model analizi başlıyor...")
            
            # **YENİ: İlk eğitim kontrolü - Model cache dosyalarını kontrol et**
            lstm_cache_file = f"model_cache/lstm_{coin_symbol.lower()}_model.h5"
            dqn_cache_file = f"model_cache/dqn_{coin_symbol.lower()}_model.h5"
            hybrid_cache_file = f"model_cache/hybrid_{coin_symbol.lower()}_model.h5"
            
            # Cache dosyalarından herhangi biri var mı kontrol et
            has_cached_models = (
                os.path.exists(lstm_cache_file) or 
                os.path.exists(dqn_cache_file) or 
                os.path.exists(hybrid_cache_file)
            )
            
            # İlk eğitim ise 1000 gün, değilse normal gün sayısı
            if not has_cached_models:
                training_days = 1000  # İlk eğitim için 1000 günlük data
                print(f"🆕 {coin_symbol} için İLK EĞİTİM tespit edildi!")
                print(f"📊 Accuracy artışı için {training_days} günlük data kullanılacak")
                logger.info(f"🔥 FIRST TRAINING for {coin_symbol}: Using {training_days} days for better accuracy")
            else:
                training_days = int(os.getenv('LSTM_TRAINING_DAYS', 200))  # Normal eğitim
                print(f"🔄 {coin_symbol} için mevcut model cache bulundu")
                print(f"📊 Normal eğitim: {training_days} günlük data kullanılacak")
                logger.info(f"📈 RETRAIN for {coin_symbol}: Using {training_days} days (cached models exist)")
            
            # Veri çek
            print(f"🔽 {coin_symbol}/USDT için {training_days} günlük 4h verileri çekiliyor...")
            with perf_step("fetch_ohlcv"):
                df = self.fetcher.fetch_ohlcv_data(coin_symbol, days=training_days)
            if df is None:
                return {'success': False, 'error': 'Veri çekilemedi'}

            print(f"✅ Başarıyla {len(df)} adet veri çekildi")
            print(f"📅 Tarih aralığı: {df.index[0]} - {df.index[-1]}")

            # Veri ön işleme
            preprocessor = CryptoDataPreprocessor()
            with perf_step("preprocess"):
                processed_df = preprocessor.prepare_data(df, use_technical_indicators=True)

            if len(processed_df) < 100:
                logger.warning(f"⚠️ {coin_symbol} için yetersiz veri: {len(processed_df)} < 100")
                return {'success': False, 'error': f'Yetersiz veri: {len(processed_df)} veri noktası'}

            print(f"🔧 Veri hazırlama tamamlandı. Toplam {len(processed_df)} veri noktası.")

            # Predictor oluştur (lazy loading — modeller bellek/disk cache'ten gelir, yeniden eğitim yok)
            predictor = CryptoPricePredictor(
                model=None,  # Lazy loading
                preprocessor=preprocessor,
                news_analyzer=self.news_analyzer,
                whale_tracker=self.whale_tracker
            )

            # **NEW: Use synchronous wrapper for async multi-model analysis**
            print("🔄 Running NEW async multi-model analysis...")
            print("📋 Execution order: LSTM → DQN → Hybrid (sequential)")
            with perf_step("model_predict"):
                multi_results = predictor.predict_multi_model_analysis_sync(processed_df, coin_symbol)
            
            # **CRITICAL FIX: Check if ANY advanced model succeeded, not just ensemble**
            advanced_models_working = (
                multi_results.get('dqn_analysis', {}).get('status') == 'success' or
                multi_results.get('dqn_analysis', {}).get('success') == True or
                multi_results.get('hybrid_analysis', {}).get('status') == 'success' or
                multi_results.get('hybrid_analysis', {}).get('success') == True or
                multi_results.get('ensemble_recommendation', {}).get('success', False)
            )
            
            if not advanced_models_working:
                # Fallback: LSTM only analysis if NO advanced models work
                logger.warning("⚠️ Hiçbir advanced model çalışmıyor, LSTM fallback kullanılıyor...")
                return self.analyze_coin(coin_symbol)
            
            logger.info("✅ En az bir advanced model başarılı - Multi_Model_Analysis modu aktif")
            print(f"🔍 DQN Status: {multi_results.get('dqn_analysis', {}).get('status', 'not_present')}")
            print(f"🔍 Hybrid Status: {multi_results.get('hybrid_analysis', {}).get('status', 'not_present')}")
            print(f"🔍 Ensemble Success: {multi_results.get('ensemble_recommendation', {}).get('success', False)}")
            
            # Sonuçları veritabanına kaydet
            try:
                lstm_pred = multi_results.get('lstm_analysis', {}).get('prediction', {})
                analysis_id = self.db.save_analysis_result(
                    coin_symbol, 
                    lstm_pred,
                    multi_results.get('news_analysis', {}),
                    multi_results.get('whale_analysis', {}),
                    multi_results.get('yigit_analysis', {})
                )
                multi_results['analysis_id'] = analysis_id
            except Exception as e:
                logger.warning(f"⚠️ Veritabanı kaydetme hatası: {e}")
                multi_results['analysis_id'] = None
            
            # **CRITICAL FIX: Generate technical analysis for multi-model results**
            lstm_prediction = multi_results.get('lstm_analysis', {}).get('prediction', {})
            if lstm_prediction and processed_df is not None:
                technical_analysis = self._generate_technical_analysis(processed_df, lstm_prediction)
                multi_results['technical_analysis'] = technical_analysis
                print(f"✅ {coin_symbol} için teknik analiz oluşturuldu (Multi-model)")
            else:
                technical_analysis = {}
                print(f"⚠️ {coin_symbol} için teknik analiz oluşturulamadı")
            
            # **CRITICAL FIX: Generate trading signal**
            ensemble_rec = multi_results.get('ensemble_recommendation', {})
            trade_signal = {}
            
            if ensemble_rec and 'action' in ensemble_rec:
                trade_signal = {
                    'action': ensemble_rec.get('action', 'HOLD'),
                    'confidence': ensemble_rec.get('confidence', 50.0),
                    'reason': ensemble_rec.get('reasoning', 'Ensemble recommendation'),
                    'price_target': ensemble_rec.get('price_target'),
                    'timestamp': datetime.now().isoformat()
                }
                print(f"📈 {coin_symbol} trading sinyali: {trade_signal['action']} (Güven: {trade_signal['confidence']:.1f}%)")
            else:
                # Default trading signal based on LSTM prediction
                if lstm_prediction:
                    price_change = lstm_prediction.get('price_change_percent', 0)
                    confidence = lstm_prediction.get('confidence', 50.0)
                    
                    if price_change > 2 and confidence > 60:
                        action = 'BUY'
                    elif price_change < -2 and confidence > 60:
                        action = 'SELL'
                    else:
                        action = 'HOLD'
                    
                    trade_signal = {
                        'action': action,
                        'confidence': confidence,
                        'reason': f'LSTM prediction: {price_change:+.1f}% expected change',
                        'price_target': lstm_prediction.get('predicted_price'),
                        'timestamp': datetime.now().isoformat()
                    }
                    print(f"📊 {coin_symbol} LSTM trading sinyali: {action} (Değişim: {price_change:+.1f}%)")

            # Web UI için format
            result = {
                'success': True,
                'coin_symbol': coin_symbol,
                'model_type': 'Multi_Model_Analysis',
                'timestamp': datetime.now().isoformat(),
                'multi_model_results': multi_results,
                
                # Backward compatibility için mevcut format - FIXED
                'prediction': lstm_prediction,
                'technical_analysis': technical_analysis,
                'news_analysis': multi_results.get('news_analysis', {}),
                'whale_analysis': multi_results.get('whale_analysis', {}),
                'yigit_analysis': multi_results.get('yigit_analysis', {}),
                'trade_signal': trade_signal,
                'analysis_id': multi_results.get('analysis_id'),
                
                # New multi-model specific fields
                'lstm_analysis': multi_results.get('lstm_analysis', {}),
                'dqn_analysis': multi_results.get('dqn_analysis', {}),
                'hybrid_analysis': multi_results.get('hybrid_analysis', {}),
                'ensemble_recommendation': multi_results.get('ensemble_recommendation', {}),
                'model_comparison': multi_results.get('model_comparison', {})
            }
            
            logger.info(f"✅ {coin_symbol} Multi-Model analizi tamamlandı")
            
            # **CRITICAL FIX: Proper model status checking**
            lstm_success = multi_results['lstm_analysis'].get('success', False)
            dqn_success = (multi_results['dqn_analysis'].get('success', False) or 
                          multi_results['dqn_analysis'].get('status') == 'success')
            hybrid_success = (multi_results['hybrid_analysis'].get('success', False) or 
                             multi_results['hybrid_analysis'].get('status') == 'success')
            
            logger.info(f"📈 LSTM: {'✅' if lstm_success else '❌'}")
            logger.info(f"🤖 DQN: {'✅' if dqn_success else '❌'}")
            logger.info(f"🔗 Hybrid: {'✅' if hybrid_success else '❌'}")
            
            return result
            
        except Exception as e:
            error_msg = f"{coin_symbol} multi-model analiz hatası: {str(e)}"
            logger.error(f"❌ {error_msg}")
            # Fallback to LSTM only
            logger.info("🔄 Fallback: LSTM-only analizi deneniyor...")
            return self.analyze_coin(coin_symbol)
    
    def start_monitoring(self, interval_minutes=15):
        """Sürekli izleme başlatır"""
        self.running = True
        logger.info(f"🔄 Coin izleme başlatıldı ({interval_minutes} dakika aralık)")
        
        def monitor_loop():
            while self.running:
                try:
                    # Aktif coinleri al
                    coins = self.db.get_active_coins()
                    
                    for coin in coins:
                        if not self.running:
                            break
                        
                        symbol = coin['symbol']
                        logger.info(f"📊 {symbol} izleniyor...")
                        
                        # Analiz yap
                        result = self.analyze_coin(symbol)
                        
                        # Otomatik trading kontrolü
                        if TRADING_CONFIG['auto_trading_enabled'] and result.get('success'):
                            trade_result = self._check_and_execute_auto_trade(symbol, result)
                            if trade_result:
                                result['auto_trade'] = trade_result
                        
                        # WebSocket ile sonucu gönder
                        socketio.emit('analysis_update', {
                            'coin': symbol,
                            'result': result,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Fiyat güncelleme
                        if result.get('success'):
                            current_price = result['prediction']['current_price']
                            # Basit 24h değişim hesabı (gerçekte daha karmaşık olmalı)
                            price_change_24h = 0  # Placeholder
                            
                            # DB güncelle (ortak katalog fiyatı + tenant izleme)
                            self.db.update_coin_price(symbol, current_price, price_change_24h)
                    
                    # Bekleme
                    for _ in range(interval_minutes * 60):  # Saniye cinsinden
                        if not self.running:
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Monitor loop hatası: {str(e)}")
                    time.sleep(60)  # Hata durumunda 1 dakika bekle
        
        # Background thread'de çalıştır
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """İzlemeyi durdurur"""
        self.running = False
        logger.info("⏹️ Coin izleme durduruldu")
    
    def _generate_technical_analysis(self, processed_df, prediction_result):
        """
        Teknik analiz sonuçlarını oluşturur
        
        Args:
            processed_df (pd.DataFrame): İşlenmiş veriler
            prediction_result (dict): Tahmin sonuçları
        
        Returns:
            dict: Teknik analiz sonuçları
        """
        try:
            current_price = prediction_result['current_price']
            
            # RSI analizi
            rsi = processed_df['rsi'].iloc[-1] if 'rsi' in processed_df.columns else 50
            rsi_signal = "SATILDI" if rsi > 70 else "AŞIRI SATILDI" if rsi < 30 else "NÖTR"
            
            # MACD analizi
            macd = processed_df['macd'].iloc[-1] if 'macd' in processed_df.columns else 0
            macd_signal = processed_df['macd_signal'].iloc[-1] if 'macd_signal' in processed_df.columns else 0
            macd_trend = "ALIM" if macd > macd_signal else "SATIM"
            
            # Moving Average analizi
            sma_7 = processed_df['sma_7'].iloc[-1] if 'sma_7' in processed_df.columns else current_price
            sma_25 = processed_df['sma_25'].iloc[-1] if 'sma_25' in processed_df.columns else current_price
            ma_trend = "YUKARI TREND" if sma_7 > sma_25 else "AŞAĞI TREND"
            
            # Bollinger Bands analizi
            bb_upper = processed_df['bb_upper'].iloc[-1] if 'bb_upper' in processed_df.columns else current_price * 1.02
            bb_lower = processed_df['bb_lower'].iloc[-1] if 'bb_lower' in processed_df.columns else current_price * 0.98
            bb_position = "ÜST BAND" if current_price > bb_upper else "ALT BAND" if current_price < bb_lower else "ORTA"
            
            # Volume analizi
            current_volume = processed_df['volume'].iloc[-1] if 'volume' in processed_df.columns else 0
            avg_volume = processed_df['volume'].tail(20).mean() if 'volume' in processed_df.columns else 0
            volume_trend = "YÜKSEK" if current_volume > avg_volume * 1.2 else "DÜŞÜK" if current_volume < avg_volume * 0.8 else "NORMAL"
            
            # Genel değerlendirme
            signals = []
            if rsi < 30: signals.append("RSI: ALIM")
            elif rsi > 70: signals.append("RSI: SATIM")
            
            if macd > macd_signal: signals.append("MACD: ALIM")
            else: signals.append("MACD: SATIM")
            
            if sma_7 > sma_25: signals.append("MA: YUKARI")
            else: signals.append("MA: AŞAĞI")
            
            return {
                'rsi': {
                    'value': rsi,
                    'signal': rsi_signal,
                    'description': f"RSI: {rsi:.1f} - {rsi_signal}"
                },
                'macd': {
                    'value': macd,
                    'signal_value': macd_signal,
                    'trend': macd_trend,
                    'description': f"MACD: {macd:.4f} vs Signal: {macd_signal:.4f} - {macd_trend}"
                },
                'moving_averages': {
                    'sma_7': sma_7,
                    'sma_25': sma_25,
                    'trend': ma_trend,
                    'description': f"SMA7: {sma_7:.4f} vs SMA25: {sma_25:.4f} - {ma_trend}"
                },
                'bollinger_bands': {
                    'upper': bb_upper,
                    'lower': bb_lower,
                    'position': bb_position,
                    'description': f"Fiyat Bollinger Bands {bb_position} bölgesinde"
                },
                'volume': {
                    'current': current_volume,
                    'average': avg_volume,
                    'trend': volume_trend,
                    'description': f"Volume {volume_trend} seviyede"
                },
                'overall_signals': signals,
                'summary': f"Toplam {len(signals)} sinyal tespit edildi"
            }
            
        except Exception as e:
            print(f"⚠️ Teknik analiz hatası: {str(e)}")
            return {
                                 'error': str(e),
                 'summary': 'Teknik analiz sırasında hata oluştu'
             }
    
    def _check_and_execute_auto_trade(self, symbol, analysis_result):
        """
        Otomatik trading kontrolü ve işlem gerçekleştirme
        
        Args:
            symbol (str): Coin sembolü
            analysis_result (dict): Analiz sonuçları
        
        Returns:
            dict: İşlem sonucu (eğer işlem yapıldıysa)
        """
        try:
            prediction = analysis_result.get('prediction', {})
            price_change_percent = prediction.get('price_change_percent', 0)
            
            # Minimum kar hedefi kontrolü (%3)
            if price_change_percent < TRADING_CONFIG['minimum_profit_threshold']:
                return None
            
            print(f"🎯 {symbol} için kar hedefi tespit edildi: %{price_change_percent:.2f}")
            
            # Mevcut açık pozisyon kontrolü
            active_trades = self._get_active_trades_count()
            if active_trades >= TRADING_CONFIG['max_concurrent_trades']:
                print(f"⚠️ Maksimum eş zamanlı işlem limitine ulaşıldı: {active_trades}")
                return None
            
            # USDT cüzdan bakiyesini al
            usdt_balance = self._get_usdt_balance()
            if usdt_balance <= TRADING_CONFIG['minimum_trade_amount']:
                print(f"💰 Yetersiz USDT bakiyesi: ${usdt_balance}")
                return None
            
            # Yatırım oranını belirle
            investment_percentage = self._calculate_investment_percentage(price_change_percent)
            trade_amount = (usdt_balance * investment_percentage / 100)
            
            # Minimum tutar kontrolü
            if trade_amount < TRADING_CONFIG['minimum_trade_amount']:
                trade_amount = TRADING_CONFIG['minimum_trade_amount']
                if trade_amount > usdt_balance:
                    print(f"💰 Minimum işlem tutarı için yetersiz bakiye")
                    return None
            
            # Stop loss ve take profit hesapla
            current_price = prediction.get('current_price', 0)
            stop_loss_price = current_price * (1 - TRADING_CONFIG['stop_loss_percentage'] / 100)
            take_profit_price = current_price * (1 + (price_change_percent * TRADING_CONFIG['take_profit_multiplier']) / 100)
            
            # Quantity hesapla
            quantity = trade_amount / current_price
            
            print(f"💡 {symbol} Otomatik İşlem Planı:")
            print(f"   💰 Yatırım Tutarı: ${trade_amount:.2f} (Cüzdan: %{investment_percentage})")
            print(f"   📊 Giriş Fiyatı: ${current_price:.4f}")
            print(f"   🎯 Kar Hedefi: ${take_profit_price:.4f} (%{price_change_percent:.2f})")
            print(f"   🔴 Stop Loss: ${stop_loss_price:.4f} (-%{TRADING_CONFIG['stop_loss_percentage']:.1f}%)")
            print(f"   🪙 Miktar: {quantity:.6f} {symbol}")
            
            # İşlemi kaydet (simülasyon modunda)
            trade_id = self.db.record_trade(
                symbol, 'BUY', current_price, quantity,
                confidence=prediction.get('confidence', 0),
                news_sentiment=analysis_result.get('news_analysis', {}).get('news_sentiment', 0) if analysis_result.get('news_analysis') else 0,
                whale_activity=analysis_result.get('whale_analysis', {}).get('whale_activity_score', 0) if analysis_result.get('whale_analysis') else 0,
                yigit_signal=analysis_result.get('yigit_analysis', {}).get('current_signal', 'NONE') if analysis_result.get('yigit_analysis') else 'NONE',
                trade_reason=f"Auto trade: Kar hedefi %{price_change_percent:.2f}",
                is_simulated=True  # Şimdilik simülasyon modunda
            )
            
            # Pozisyon oluştur
            position_id = self.db.update_position(
                symbol, 'LONG', current_price, quantity, current_price,
                leverage=1, stop_loss=stop_loss_price, take_profit=take_profit_price
            )
            
            trade_result = {
                'symbol': symbol,
                'action': 'BUY',
                'price': current_price,
                'quantity': quantity,
                'amount': trade_amount,
                'investment_percentage': investment_percentage,
                'profit_target': price_change_percent,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'trade_id': trade_id,
                'position_id': position_id,
                'timestamp': datetime.now().isoformat(),
                'mode': 'SIMULATION'
            }
            
            print(f"✅ {symbol} otomatik işlem gerçekleştirildi!")
            
            # WebSocket ile bildir
            socketio.emit('auto_trade_executed', trade_result)
            
            return trade_result
            
        except Exception as e:
            print(f"❌ {symbol} otomatik işlem hatası: {str(e)}")
            return None
    
    def _get_usdt_balance(self):
        """USDT cüzdan bakiyesini alır"""
        try:
            # Binance API'den gerçek bakiye al
            from binance_history import BinanceHistoryFetcher
            
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_SECRET_KEY')
            testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
            
            if api_key and api_secret:
                fetcher = BinanceHistoryFetcher(api_key, api_secret, testnet)
                if fetcher.exchange:
                    account_info = fetcher.fetch_account_info()
                    if 'balances' in account_info:
                        usdt_balance = account_info['balances'].get('USDT', {}).get('free', 0)
                        return float(usdt_balance) if usdt_balance else 0
            
            # Fallback: Test bakiyesi
            return 1000.0  # Test için 1000 USDT
            
        except Exception as e:
            print(f"⚠️ USDT bakiye alma hatası: {str(e)}")
            return 1000.0  # Test için 1000 USDT
    
    def _get_active_trades_count(self):
        """Aktif işlem sayısını döndürür"""
        try:
            open_positions = self.db.get_open_positions()
            return len(open_positions)
        except:
            return 0
    
    def _calculate_investment_percentage(self, profit_target_percent):
        """Kar hedefine göre yatırım yüzdesini hesaplar"""
        risk_levels = sorted(TRADING_CONFIG['risk_percentages'].keys(), reverse=True)
        
        for threshold in risk_levels:
            if profit_target_percent >= threshold:
                return TRADING_CONFIG['risk_percentages'][threshold]
        
        # Minimum threshold (%3) altında ise en düşük oran
        return TRADING_CONFIG['risk_percentages'][min(risk_levels)]

# Trading Configuration - Statik Kontrol Bloğu
TRADING_CONFIG = {
    'auto_trading_enabled': False,  # Manuel kontrol için
    'minimum_profit_threshold': 3.0,  # Minimum %3 kar hedefi
    'minimum_trade_amount': 10.0,  # Minimum 10 USDT
    'risk_percentages': {
        # Kar hedefine göre USDT yatırım oranları
        15.0: 60,  # %15+ kar hedefi → %60 USDT
        10.0: 40,  # %10-15 kar hedefi → %40 USDT
        5.0: 30,   # %5-10 kar hedefi → %30 USDT
        3.0: 25    # %3-5 kar hedefi → %25 USDT
    },
    'max_concurrent_trades': 3,  # Maksimum eş zamanlı işlem sayısı
    'stop_loss_percentage': 5.0,  # %5 stop loss
    'take_profit_multiplier': 1.5  # Kar hedefinin 1.5 katında kar al
}

# Global monitor instance
coin_monitor = CoinMonitor()

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login sayfası"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Kullanıcı adı ve şifre gerekli!', 'error')
            return render_template('login.html')
        
        # Kullanıcı doğrulama
        user = auth_manager.authenticate_user(username, password)
        
        if user:
            login_user(user)
            flash(f'Hoş geldiniz, {username}!', 'success')
            
            # Next URL varsa oraya yönlendir
            next_url = request.args.get('next')
            if next_url:
                return redirect(next_url)
            
            return redirect(url_for('dashboard'))
        else:
            flash('Kullanıcı adı veya şifre hatalı!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Logout işlemi"""
    username = current_user.username
    logout_user()
    flash(f'Çıkış yapıldı! Güle güle {username}', 'info')
    return redirect(url_for('login'))

# Web Routes
def get_coins_with_live_data():
    """Aktif coinler için canlı fiyat ve 24h değişimi çekip coins tablosunu günceller"""
    try:
        # Veritabanından aktif coinleri al
        coins = db.get_active_coins()
        updated_coins = []
        
        for coin in coins:
            symbol = coin['symbol']
            try:
                # Canlı fiyat ve 24h değişimi çek
                ticker = data_fetcher.exchange.fetch_ticker(f"{symbol}/USDT")
                
                current_price = ticker['last']
                price_change_24h = ticker['percentage']
                
                # Coins tablosunu güncelle (ortak katalog + tenant izleme)
                try:
                    db.update_coin_price(symbol, current_price, price_change_24h)
                    print(f"💰 {symbol}: ${current_price:.2f} ({price_change_24h:+.2f}%)")
                except Exception as update_error:
                    print(f"⚠️ {symbol} coins tablosu güncelleme hatası: {update_error}")
                
                # Güncel veriyi listeye ekle
                coin_updated = coin.copy()
                coin_updated['current_price'] = current_price
                coin_updated['price_change_24h'] = price_change_24h
                updated_coins.append(coin_updated)
                
            except Exception as price_error:
                print(f"⚠️ {symbol} canlı fiyat çekme hatası: {price_error}")
                # Hata varsa eski veriyi koru
                updated_coins.append(coin)
        
        print(f"📊 {len(updated_coins)} coin için canlı fiyat güncellendi")
        return updated_coins
        
    except Exception as e:
        print(f"❌ Canlı fiyat güncelleme genel hatası: {e}")
        # Hata varsa eski veriyi döndür
        return db.get_active_coins()

@app.route('/')
@login_required
def dashboard():
    """Ana dashboard - System persistence destekli"""
    try:
        # Startup summary (sistem durumu)
        startup_summary = persistence.get_startup_summary()
        
        # Portfolio özeti
        portfolio = db.get_portfolio_summary()
        
        # Aktif coinler - Canlı fiyat güncellemesi ile
        coins = get_coins_with_live_data()
        
        # Son işlemler
        recent_trades = db.get_recent_trades(10)
        
        # Açık pozisyonlar
        open_positions = db.get_open_positions()
        
        # Monitoring durumu kontrol et
        monitoring_state = persistence.load_monitoring_state()
        should_resume = monitoring_state.get('should_resume', False)
        
        # Eğer resume edilecek session varsa bilgi göster
        resume_info = None
        if should_resume and not coin_monitor.running:
            last_session = monitoring_state.get('last_session', {})
            resume_info = {
                'can_resume': True,
                'session_id': last_session.get('session_id', 'unknown'),
                'coin_count': len(monitoring_state.get('active_coins', [])),
                'interval': monitoring_state.get('interval_minutes', 15),
                'last_activity': last_session.get('last_activity', 'unknown')
            }
        
        return render_template('dashboard.html',
                             portfolio=portfolio,
                             coins=coins,
                             recent_trades=recent_trades,
                             open_positions=open_positions,
                             monitoring_active=coin_monitor.running,
                             database_type=DATABASE_TYPE,
                             startup_summary=startup_summary,
                             resume_info=resume_info)
    except Exception as e:
        flash(f'Dashboard yükleme hatası: {str(e)}', 'error')
        return render_template('dashboard.html',
                             portfolio={}, coins=[], recent_trades=[],
                             open_positions=[], monitoring_active=False,
                             database_type=DATABASE_TYPE,
                             startup_summary={}, resume_info=None)

@app.route('/add_coin', methods=['POST'])
@login_required
def add_coin():
    """Coin ekleme ve comprehensive training başlatma"""
    try:
        symbol = request.form.get('symbol', '').upper()
        name = request.form.get('name', '')
        auto_analyze = request.form.get('auto_analyze', 'true').lower() == 'true'
        comprehensive_training = request.form.get('comprehensive_training', 'true').lower() == 'true'
        
        if not symbol:
            flash('Coin sembolü gerekli!', 'error')
            return redirect(url_for('dashboard'))
        
        # Symbol doğrulama
        if not data_fetcher.validate_symbol(symbol):
            flash(f'{symbol} geçerli bir sembol değil!', 'error')
            return redirect(url_for('dashboard'))
        
        # Veritabanına ekle
        success = db.add_coin(symbol, name)
        
        if success:
            flash(f'{symbol} izleme listesine eklendi!', 'success')
            
            # Scheduler'a ekle
            try:
                from training_scheduler import get_scheduler
                scheduler = get_scheduler()
                scheduler.add_coin_to_schedule(symbol)
                flash(f'📅 {symbol} haftalık training schedule\'a eklendi!', 'info')
            except Exception as scheduler_error:
                logger.warning(f"Scheduler ekleme hatası: {scheduler_error}")
            
            # Comprehensive training yap (eğer isteniyorsa)
            if comprehensive_training:
                try:
                    flash(f'🚀 {symbol} için comprehensive training başlatılıyor...', 'info')
                    
                    def background_comprehensive_training():
                        try:
                            from comprehensive_trainer import ComprehensiveTrainer
                            trainer = ComprehensiveTrainer()
                            
                            logger.info(f"🔥 {symbol} comprehensive training başlıyor...")
                            result = trainer.train_coin_sync(symbol, is_fine_tune=False)
                            
                            if result['success']:
                                successful_models = result.get('successful_models', [])
                                failed_models = result.get('failed_models', [])
                                predictions = result.get('predictions', {})
                                
                                # WebSocket ile sonucu gönder
                                socketio.emit('comprehensive_training_complete', {
                                    'coin': symbol,
                                    'success': True,
                                    'successful_models': successful_models,
                                    'failed_models': failed_models,
                                    'predictions_4h': len(predictions.get('4h', {})),
                                    'predictions_1d': len(predictions.get('1d', {})),
                                    'message': f'{symbol} comprehensive training tamamlandı! '
                                             f'Başarılı modeller: {len(successful_models)}/{len(successful_models + failed_models)}',
                                    'timestamp': datetime.now().isoformat()
                                })
                                
                                logger.info(f"✅ {symbol} comprehensive training tamamlandı")
                                
                                # Ana tahmin sonucu varsa, normal analiz sonucu gibi de gönder
                                if 'LSTM' in predictions.get('4h', {}):
                                    lstm_prediction = predictions['4h']['LSTM']
                                    socketio.emit('analysis_complete', {
                                        'coin': symbol,
                                        'result': {
                                            'success': True,
                                            'prediction': lstm_prediction,
                                            'model_type': 'Comprehensive_Training',
                                            'comprehensive_results': result
                                        },
                                        'message': f'{symbol} comprehensive training tamamlandı! '
                                                 f'4h Tahmin: ${lstm_prediction.get("predicted_price", 0):.4f}',
                                        'timestamp': datetime.now().isoformat()
                                    })
                            else:
                                error_msg = result.get('error', 'Bilinmeyen hata')
                                socketio.emit('comprehensive_training_error', {
                                    'coin': symbol,
                                    'error': error_msg,
                                    'timestamp': datetime.now().isoformat()
                                })
                                logger.error(f"❌ {symbol} comprehensive training başarısız: {error_msg}")
                                
                        except Exception as training_error:
                            error_msg = str(training_error)
                            socketio.emit('comprehensive_training_error', {
                                'coin': symbol,
                                'error': error_msg,
                                'timestamp': datetime.now().isoformat()
                            })
                            logger.error(f"❌ {symbol} comprehensive training exception: {error_msg}")
                    
                    # Thread'de başlat
                    training_thread = threading.Thread(target=background_comprehensive_training, daemon=True)
                    training_thread.start()
                    
                    flash(f'🧠 {symbol} için COMPREHENSIVE TRAINING (LSTM+DQN+Hybrid) arka planda başlatıldı!', 'success')
                    flash(f'📊 4 saatlik ve 1 günlük tahminler oluşturulacak!', 'info')
                    
                except Exception as training_error:
                    flash(f'Comprehensive training hatası: {str(training_error)}', 'warning')
                    
                    # Fallback: Normal analiz yap
                    if auto_analyze:
                        try:
                            def background_analysis():
                                result = coin_monitor.analyze_coin(symbol)
                                if result['success']:
                                    socketio.emit('analysis_complete', {
                                        'coin': symbol,
                                        'result': result,
                                        'message': f'{symbol} analizi tamamlandı! Tahmin: ${result["prediction"]["predicted_price"]:.4f}',
                                        'timestamp': datetime.now().isoformat()
                                    })
                                else:
                                    socketio.emit('analysis_error', {
                                        'coin': symbol,
                                        'error': result.get('error', 'Bilinmeyen hata'),
                                        'timestamp': datetime.now().isoformat()
                                    })
                            
                            analysis_thread = threading.Thread(target=background_analysis, daemon=True)
                            analysis_thread.start()
                            flash(f'🔄 Fallback: {symbol} için normal LSTM analizi başlatıldı', 'info')
                            
                        except Exception as fallback_error:
                            flash(f'Fallback analiz hatası: {str(fallback_error)}', 'warning')
            
            elif auto_analyze:
                # Sadece normal analiz yap
                try:
                    flash(f'🔍 {symbol} için normal analiz başlatılıyor...', 'info')
                    
                    def background_analysis():
                        result = coin_monitor.analyze_coin(symbol)
                        if result['success']:
                            logger.info(f"✅ {symbol} otomatik analizi tamamlandı")
                            socketio.emit('analysis_complete', {
                                'coin': symbol,
                                'result': result,
                                'message': f'{symbol} analizi tamamlandı! Tahmin: ${result["prediction"]["predicted_price"]:.4f}',
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            logger.error(f"❌ {symbol} otomatik analizi başarısız: {result.get('error')}")
                            socketio.emit('analysis_error', {
                                'coin': symbol,
                                'error': result.get('error', 'Bilinmeyen hata'),
                                'timestamp': datetime.now().isoformat()
                            })
                    
                    analysis_thread = threading.Thread(target=background_analysis, daemon=True)
                    analysis_thread.start()
                    
                    flash(f'🧠 {symbol} için LSTM eğitimi ve tahmin arka planda başlatıldı!', 'success')
                    
                except Exception as analysis_error:
                    flash(f'Otomatik analiz hatası: {str(analysis_error)}', 'warning')
        else:
            flash(f'{symbol} eklenirken hata oluştu!', 'error')
            
    except Exception as e:
        flash(f'Coin ekleme hatası: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/remove_coin/<symbol>')
def remove_coin(symbol):
    """Coin çıkarma"""
    try:
        success = db.remove_coin(symbol)
        
        if success:
            flash(f'{symbol} izleme listesinden çıkarıldı!', 'success')
        else:
            flash(f'{symbol} çıkarılırken hata oluştu!', 'error')
            
    except Exception as e:
        flash(f'Coin çıkarma hatası: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/analyze_coin/<symbol>')
@login_required
def analyze_coin_route(symbol):
    """Detaylı coin analizi sayfası - Multi-model destekli"""
    try:
        symbol = symbol.upper()

        # **YENİ: force_refresh — kullanıcı taze (canlı) analiz isterse cache atlanır**
        force_refresh = request.args.get('force_refresh', '').lower() in ('1', 'true', 'yes')

        # **NEW: Database-based cache kontrolü**
        cached_analysis = None
        try:
            # Cache temizle (süresi dolmuş olanları)
            db.cleanup_expired_cache()

            # Geçerli cache var mı kontrol et (force_refresh ise cache atlanır)
            if force_refresh:
                print(f"🔄 {symbol} force_refresh — cache atlanıyor, canlı analiz yapılacak")
                cached_analysis = None
            else:
                cached_analysis = db.get_prediction_cache(symbol)
            
            if cached_analysis:
                cache_age = datetime.now() - cached_analysis['cache_timestamp']
                print(f"📦 {symbol} için geçerli cache bulundu (Yaş: {int(cache_age.total_seconds()//60)} dakika)")
                flash(f'{symbol} için cache\'den analiz gösteriliyor (Yaş: {int(cache_age.total_seconds()//60)} dakika)', 'info')
            else:
                print(f"🔄 {symbol} için geçerli cache bulunamadı, yeni analiz yapılacak")
                
        except Exception as e:
            print(f"⚠️ Database cache kontrolü hatası: {str(e)}")
            cached_analysis = None
        
        # Eğer geçerli cache var ise, onu kullan
        if cached_analysis:
            # **CRITICAL FIX: Ensure current_price is always available in cached results**
            cached_prediction = cached_analysis.get('prediction', {})
            
            # **CRITICAL FIX: ALWAYS refresh current_price for cached predictions**
            try:
                # Fetch fresh price data to update cached current_price
                df = data_fetcher.fetch_ohlcv_data(f"{symbol}/USDT", timeframe="4h", days=1)
                if df is not None and len(df) > 0:
                    fresh_current_price = df['close'].iloc[-1]
                    old_price = cached_prediction.get('current_price', 0)
                    cached_prediction['current_price'] = fresh_current_price
                    
                    # **NEW: Also update predicted_price proportionally if it exists**
                    if 'predicted_price' in cached_prediction and old_price > 0:
                        price_ratio = fresh_current_price / old_price
                        # Only update if the price change is reasonable (< 20%)
                        if 0.8 <= price_ratio <= 1.2:
                            cached_prediction['predicted_price'] = cached_prediction['predicted_price'] * price_ratio
                            cached_prediction['price_change_percent'] = ((cached_prediction['predicted_price'] - fresh_current_price) / fresh_current_price) * 100
                            print(f"🔧 Cache updated: ${old_price:.2f} → ${fresh_current_price:.2f} (Ratio: {price_ratio:.3f})")
                        else:
                            print(f"⚠️ Price change too large for cached prediction update: {price_ratio:.3f}")
                    else:
                        print(f"🔧 Fresh current_price updated in cache: ${fresh_current_price:.6f}")
                else:
                    # Fallback to a default value
                    cached_prediction['current_price'] = cached_prediction.get('current_price', 0.0)
                    print("⚠️ Fresh price alınamadı, cache'deki fiyat korunuyor")
            except Exception as e:
                print(f"⚠️ Fresh price fetch hatası: {e}")
                # Keep existing price if available
                if 'current_price' not in cached_prediction:
                    cached_prediction['current_price'] = 0.0
            
            # **CRITICAL FIX: Generate missing analyses for cached results**
            print(f"🔄 Cache'den alınan sonuç için eksik analizler hesaplanıyor...")
            
            # Get fresh data for missing analyses
            try:
                df = data_fetcher.fetch_ohlcv_data(f"{symbol}/USDT", timeframe="4h", days=30)
                if df is not None:
                    # Prepare data for technical analysis
                    from data_preprocessor import CryptoDataPreprocessor
                    preprocessor = CryptoDataPreprocessor()
                    processed_df = preprocessor.prepare_data(df, use_technical_indicators=True)
                    
                    # Generate missing technical analysis
                    technical_analysis = {}
                    if processed_df is not None and len(processed_df) > 0:
                        technical_analysis = coin_monitor._generate_technical_analysis(processed_df, cached_prediction)
                        print("✅ Cache için teknik analiz hesaplandı")
                    
                    # Generate missing news analysis
                    news_analysis = cached_analysis.get('news_analysis', {})
                    if not news_analysis and coin_monitor.news_analyzer:
                        try:
                            news_data = coin_monitor.news_analyzer.fetch_all_news(symbol, days=7)
                            if news_data:
                                news_df = coin_monitor.news_analyzer.analyze_news_sentiment_batch(news_data)
                                if not news_df.empty:
                                    news_analysis = {
                                        'news_sentiment': news_df['overall_sentiment'].mean(),
                                        'news_count': len(news_df),
                                        'analysis_time': datetime.now().isoformat()
                                    }
                                    print("✅ Cache için haber analizi hesaplandı")
                        except Exception as e:
                            print(f"⚠️ Cache haber analizi hatası: {e}")
                    
                    # Generate missing whale analysis
                    whale_analysis = cached_analysis.get('whale_analysis', {})
                    if not whale_analysis and coin_monitor.whale_tracker:
                        try:
                            whale_txs = coin_monitor.whale_tracker.fetch_whale_alert_transactions(symbol, 24)
                            if whale_txs:
                                whale_data = coin_monitor.whale_tracker.analyze_whale_transactions(whale_txs)
                                whale_analysis = {
                                    'whale_activity_score': whale_data.get('whale_activity_score', 0),
                                    'total_volume': whale_data.get('total_volume', 0)
                                }
                                print("✅ Cache için whale analizi hesaplandı")
                        except Exception as e:
                            print(f"⚠️ Cache whale analizi hatası: {e}")
                    
                    # Generate missing yigit analysis
                    yigit_analysis = cached_analysis.get('yigit_analysis', {})
                    if not yigit_analysis and processed_df is not None:
                        try:
                            from predictor import CryptoPricePredictor
                            predictor = CryptoPricePredictor(None, preprocessor, coin_monitor.news_analyzer, coin_monitor.whale_tracker)
                            yigit_analysis = predictor.analyze_yigit_signals(processed_df)
                            print("✅ Cache için Yigit analizi hesaplandı")
                        except Exception as e:
                            print(f"⚠️ Cache Yigit analizi hatası: {e}")
                    
                    # Generate missing trading signal
                    trade_signal = cached_analysis.get('trade_signal', {})
                    if not trade_signal and cached_prediction:
                        price_change = cached_prediction.get('price_change_percent', 0)
                        confidence = cached_prediction.get('confidence', 50.0)
                        
                        if price_change > 2 and confidence > 60:
                            action = 'BUY'
                        elif price_change < -2 and confidence > 60:
                            action = 'SELL'
                        else:
                            action = 'HOLD'
                        
                        trade_signal = {
                            'action': action,
                            'confidence': confidence,
                            'reason': f'Cached prediction: {price_change:+.1f}% expected change',
                            'price_target': cached_prediction.get('predicted_price'),
                            'timestamp': datetime.now().isoformat()
                        }
                        print("✅ Cache için trading sinyali oluşturuldu")
                    
                else:
                    technical_analysis = {}
                    news_analysis = cached_analysis.get('news_analysis', {})
                    whale_analysis = cached_analysis.get('whale_analysis', {})
                    yigit_analysis = cached_analysis.get('yigit_analysis', {})
                    trade_signal = cached_analysis.get('trade_signal', {})
                    
            except Exception as e:
                print(f"⚠️ Cache analiz hesaplama hatası: {e}")
                technical_analysis = {}
                news_analysis = cached_analysis.get('news_analysis', {})
                whale_analysis = cached_analysis.get('whale_analysis', {})
                yigit_analysis = cached_analysis.get('yigit_analysis', {})
                trade_signal = cached_analysis.get('trade_signal', {})

            # **CRITICAL FIX: Extract multi-model results from cached prediction_data**
            prediction_data = cached_analysis.get('prediction', {})
            
            result = {
                'success': True,
                'prediction': cached_prediction,
                'technical_analysis': technical_analysis,
                'news_analysis': news_analysis,
                'whale_analysis': whale_analysis,
                'yigit_analysis': yigit_analysis,
                'trade_signal': trade_signal,
                'timestamp': cached_analysis.get('cache_timestamp', datetime.now()).isoformat() if isinstance(cached_analysis.get('cache_timestamp'), datetime) else cached_analysis.get('cache_timestamp', datetime.now().isoformat()),
                'analysis_id': f"cached_{symbol}",
                'is_cached': True,
                # **CRITICAL: Extract multi-model results from prediction_data**
                'model_type': prediction_data.get('model_type', cached_analysis.get('model_type', 'Cached_Analysis')),
                'lstm_analysis': prediction_data.get('lstm_analysis', {}),
                'dqn_analysis': prediction_data.get('dqn_analysis', {}),
                'hybrid_analysis': prediction_data.get('hybrid_analysis', {}),
                'ensemble_recommendation': prediction_data.get('ensemble_recommendation', {}),
                'model_comparison': prediction_data.get('model_comparison', {}),
                'multi_model_results': prediction_data.get('multi_model_results', {})
            }
        else:
            # **CRITICAL FIX: Always try multi-model analysis first**
            print(f"🔄 {symbol} için yeni multi-model analiz yapılıyor...")
            result = None
            
            # **IMPROVED: Try multi-model with better error handling**
            try:
                print(f"🚀 {symbol} Multi-Model Analysis başlatılıyor...")
                with analysis_timer(f"{symbol}/USDT"):
                    result = coin_monitor.analyze_coin_multi_model(symbol)
                
                if result and result.get('success', False):
                    result['model_type'] = 'Multi_Model_Analysis'
                    print(f"✅ {symbol} Multi-Model Analysis başarılı!")
                else:
                    error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                    print(f"❌ Multi-model başarısız: {error_msg}")
                    result = None
                    
            except Exception as e:
                print(f"❌ Multi-model exception: {str(e)}")
                import traceback
                print(f"🔍 Multi-model traceback: {traceback.format_exc()}")
                result = None
            
            # **Fallback to LSTM-only if multi-model failed**
            if result is None or not result.get('success', False):
                print(f"🔄 {symbol} için LSTM-only fallback analizi...")
                try:
                    result = coin_monitor.analyze_coin(symbol)
                    if result and result.get('success', False):
                        result['model_type'] = 'LSTM_Only'
                        print(f"✅ {symbol} LSTM-only fallback başarılı!")
                    else:
                        print(f"❌ LSTM-only de başarısız!")
                        result = {
                            'success': False,
                            'error': 'Both multi-model and LSTM-only analysis failed',
                            'model_type': 'Failed'
                        }
                except Exception as e:
                    print(f"❌ LSTM fallback exception: {str(e)}")
                    result = {
                        'success': False,
                        'error': f'Analysis failed: {str(e)}',
                        'model_type': 'Failed'
                    }
        
        # **NEW: Yeni analiz yapıldıysa cache'e kaydet**
        if result and result.get('success', False) and not result.get('is_cached', False):
            try:
                # **CRITICAL FIX: Multi-model sonuçlarını prediction_data'ya dahil et**
                enhanced_prediction_data = result.get('prediction', {}).copy()
                enhanced_prediction_data.update({
                    'model_type': result.get('model_type', 'Unknown'),
                    'lstm_analysis': result.get('lstm_analysis', {}),
                    'dqn_analysis': result.get('dqn_analysis', {}),
                    'hybrid_analysis': result.get('hybrid_analysis', {}),
                    'ensemble_recommendation': result.get('ensemble_recommendation', {}),
                    'model_comparison': result.get('model_comparison', {}),
                    'multi_model_results': result.get('multi_model_results', {}),
                    'timestamp': result.get('timestamp', datetime.now().isoformat())
                })
                
                db.save_prediction_cache(
                    coin_symbol=symbol,
                    model_type=result.get('model_type', 'Unknown'),
                    prediction_data=enhanced_prediction_data,
                    technical_analysis=result.get('technical_analysis', {}),
                    news_analysis=result.get('news_analysis', {}),
                    whale_analysis=result.get('whale_analysis', {}),
                    yigit_analysis=result.get('yigit_analysis', {}),
                    trade_signal=result.get('trade_signal', {}),
                    cache_duration_minutes=15
                )
                print(f"💾 {symbol} yeni analiz sonucu cache'e kaydedildi (Multi-model: {result.get('model_type', 'Unknown')})")
            except Exception as cache_error:
                print(f"⚠️ Cache kaydetme hatası: {cache_error}")
        
        if not result['success']:
            flash(f'{symbol} analizi başarısız: {result.get("error", "Bilinmeyen hata")}', 'error')
            return redirect(url_for('dashboard'))
        
        # **CRITICAL: Save analysis to database**
        try:
            # Analysis sonuçlarını database'e kaydet
            prediction_data = result.get('prediction', {})
            
            # Database'e kaydet (doğru parametre sırası ile)
            analysis_id = db.save_analysis_result(
                coin_symbol=symbol,
                prediction_result=prediction_data,
                news_analysis=result.get('news_analysis'),
                whale_analysis=result.get('whale_analysis'), 
                yigit_analysis=result.get('yigit_analysis')
            )
            
            print(f"💾 {symbol} analiz sonuçları database'e kaydedildi (ID: {analysis_id})")
            
        except Exception as db_error:
            print(f"⚠️ Database kaydetme hatası: {db_error}")
            # Database hatası analizi etkilemesin
            pass
        
        # **CRITICAL: Safe access to prediction data with fallbacks**
        prediction_data = result.get('prediction', {})
        current_price = prediction_data.get('current_price', 0.0)
        predicted_price = prediction_data.get('predicted_price', current_price)
        
        # **CRITICAL FIX: Always fetch fresh current_price for UI display**
        try:
            df = data_fetcher.fetch_ohlcv_data(f"{symbol}/USDT", timeframe="1h", days=1)
            if df is not None and len(df) > 0:
                fresh_current_price = df['close'].iloc[-1]
                
                # If we have a reasonable current_price from prediction, compare and update
                if current_price > 0:
                    old_current = current_price
                    current_price = fresh_current_price
                    
                    # Proportionally adjust predicted_price if needed
                    if predicted_price > 0 and old_current > 0:
                        price_ratio = fresh_current_price / old_current
                        # Only adjust if the change is reasonable (< 20%)
                        if 0.8 <= price_ratio <= 1.2:
                            predicted_price = predicted_price * price_ratio
                            print(f"🔧 UI Template updated: Current ${old_current:.2f} → ${fresh_current_price:.2f}, Predicted adjusted")
                        else:
                            print(f"⚠️ Large price change detected: ${old_current:.2f} → ${fresh_current_price:.2f} (No prediction adjustment)")
                else:
                    # No valid current_price from prediction, use fresh
                    current_price = fresh_current_price
                    predicted_price = current_price * 1.001  # Minimal prediction
                    print(f"🔧 UI Template: Fresh price used ${current_price:.6f}")
            else:
                print("⚠️ Fresh price fetch failed for UI template")
                if current_price <= 0:
                    current_price = 1.0
                    predicted_price = 1.001
        except Exception as e:
            print(f"⚠️ UI Template fresh price error: {e}")
            if current_price <= 0:
                current_price = 1.0
                predicted_price = 1.001
        
        # Prevent division by zero and calculate price change
        if current_price > 0:
            price_change = ((predicted_price - current_price) / current_price) * 100
        else:
            price_change = 0.0
        
        # **CRITICAL FIX: Handle cache vs fresh analysis for multi-model results**
        if result.get('is_cached', False):
            # Cache'den gelen sonuçlar için prediction_data'dan multi-model sonuçlarını al
            lstm_analysis = prediction_data.get('lstm_analysis', {})
            dqn_analysis = prediction_data.get('dqn_analysis', {})
            hybrid_analysis = prediction_data.get('hybrid_analysis', {})
            ensemble_recommendation = prediction_data.get('ensemble_recommendation', {})
            model_comparison = prediction_data.get('model_comparison', {})
            multi_model_results = prediction_data.get('multi_model_results', {})
            
            print(f"🔍 CACHE DEBUG - Extracted from prediction_data:")
            print(f"   LSTM: {bool(lstm_analysis)}")
            print(f"   DQN: {bool(dqn_analysis)}")
            print(f"   Hybrid: {bool(hybrid_analysis)}")
        else:
            # Fresh analysis için result'dan al
            lstm_analysis = result.get('lstm_analysis', {})
            dqn_analysis = result.get('dqn_analysis', {})
            hybrid_analysis = result.get('hybrid_analysis', {})
            ensemble_recommendation = result.get('ensemble_recommendation', {})
            model_comparison = result.get('model_comparison', {})
            multi_model_results = result.get('multi_model_results', {})
        
        # **CRITICAL: Comprehensive analysis data with multi-model support**
        analysis_data = {
            'symbol': symbol.upper(),
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change_percent': price_change,
            'confidence': prediction_data.get('confidence', 50.0) or 50.0,
            'prediction_timeframe': prediction_data.get('timeframe', '4 saat'),
            
            # **CRITICAL: Add model type for template conditionals**
            'model_type': result.get('model_type', 'LSTM_Only'),
            
            # LSTM ve Teknik Analiz with safe access
            'prediction_details': prediction_data,
            'technical_analysis': result.get('technical_analysis', {}) or {},
            
            # **CRITICAL: Multi-model results for template (from cache or fresh)**
            'lstm_analysis': lstm_analysis or {},
            'dqn_analysis': dqn_analysis or {},
            'hybrid_analysis': hybrid_analysis or {},
            'ensemble_recommendation': ensemble_recommendation or {},
            'model_comparison': model_comparison or {},
            'multi_model_results': multi_model_results or {},
            
            # **CRITICAL: Multi-model availability flags with corrected checks**
            'has_dqn_analysis': bool(dqn_analysis) and dqn_analysis != {},
            'has_hybrid_analysis': bool(hybrid_analysis) and hybrid_analysis != {},
            'has_ensemble': bool(ensemble_recommendation) and ensemble_recommendation != {},
            
            # **CRITICAL FIX: Always show analysis tabs with safe access**
            # Haber Analizi
            'news_analysis': result.get('news_analysis', {}) or {},
            'news_available': True,  # Always show, handle empty data in template
            
            # Whale Analizi
            'whale_analysis': result.get('whale_analysis', {}) or {},
            'whale_available': True,  # Always show, handle empty data in template
            
            # Yigit Analizi
            'yigit_analysis': result.get('yigit_analysis', {}) or {},
            'yigit_available': True,  # Always show, handle empty data in template
            
            # Trading Sinyali
            'trade_signal': result.get('trade_signal', {}) or {},
            'trade_available': True,  # Always show, handle empty data in template
            
            # Analiz zamanı with safe access
            'analysis_timestamp': result.get('timestamp', datetime.now().isoformat()),
            'analysis_id': result.get('analysis_id', f"unknown_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            'is_cached': result.get('is_cached', False)
        }
        
        # Risk değerlendirmesi
        risk_level = 'LOW'
        if price_change > 5 or price_change < -5:
            risk_level = 'HIGH'
        elif abs(price_change) > 2:
            risk_level = 'MEDIUM'
        
        analysis_data['risk_level'] = risk_level
        
        # Son 24 saat analiz geçmişi (basit mock data)
        try:
            # Mock data since get_coin_analysis_history method may not exist
            analysis_data['recent_analyses'] = []
        except:
            analysis_data['recent_analyses'] = []
        
        logger.info(f"✅ {symbol} analizi tamamlandı - Model: {analysis_data['model_type']}")
        
        # **DEBUG: Log template data for multi-model checking**
        print(f"🔍 DEBUG - Template Data:")
        print(f"   Model Type: {analysis_data['model_type']}")
        print(f"   Has DQN: {analysis_data['has_dqn_analysis']}")
        print(f"   Has Hybrid: {analysis_data['has_hybrid_analysis']}")
        print(f"   Has Ensemble: {analysis_data['has_ensemble']}")
        print(f"   DQN Status: {result.get('dqn_analysis', {}).get('status', 'not_present')}")
        print(f"   Hybrid Status: {result.get('hybrid_analysis', {}).get('status', 'not_present')}")
        
        return render_template('analyze_coin.html', analysis=analysis_data)
        
    except Exception as e:
        logger.error(f'Analiz sayfası hatası: {str(e)}')
        flash(f'Analiz sayfası yüklenirken hata: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/start_monitoring')
def start_monitoring():
    """İzleme başlatma - Persistence destekli"""
    try:
        if not coin_monitor.running:
            interval = request.args.get('interval', coin_monitor.monitoring_interval, type=int)
            
            # Aktif coinleri al
            active_coins = [coin['symbol'] for coin in db.get_active_coins()]
            
            # Monitoring başlat
            coin_monitor.start_monitoring(interval)
            
            # Persistence'a kaydet
            persistence.save_monitoring_state(
                is_active=True,
                interval_minutes=interval,
                active_coins=active_coins,
                session_info={
                    'database_type': DATABASE_TYPE,
                    'auto_trading_enabled': coin_monitor.auto_trader is not None,
                    'news_analysis_enabled': coin_monitor.news_analyzer is not None,
                    'whale_tracking_enabled': coin_monitor.whale_tracker is not None
                }
            )
            
            flash(f'Coin izleme başlatıldı! ({interval} dakika aralık)', 'success')
            logger.info(f"🚀 Monitoring başlatıldı: {len(active_coins)} coin, {interval}min interval")
        else:
            flash('İzleme zaten aktif!', 'warning')
    except Exception as e:
        flash(f'İzleme başlatma hatası: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/stop_monitoring')
def stop_monitoring():
    """İzleme durdurma - Persistence destekli"""
    try:
        # Monitoring durdur
        coin_monitor.stop_monitoring()
        
        # Persistence güncelle
        active_coins = [coin['symbol'] for coin in db.get_active_coins()]
        persistence.save_monitoring_state(
            is_active=False,
            interval_minutes=coin_monitor.monitoring_interval,
            active_coins=active_coins,
            session_info={
                'stopped_at': datetime.now().isoformat(),
                'stopped_manually': True
            }
        )
        
        flash('Coin izleme durduruldu!', 'success')
        logger.info("⏹️ Monitoring durduruldu")
    except Exception as e:
        flash(f'İzleme durdurma hatası: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/resume_monitoring')
def resume_monitoring():
    """Önceki session'ı resume etme"""
    try:
        monitoring_state = persistence.load_monitoring_state()
        
        if not monitoring_state.get('should_resume', False):
            flash('Resume edilecek önceki session bulunamadı!', 'warning')
            return redirect(url_for('dashboard'))
        
        if coin_monitor.running:
            flash('İzleme zaten aktif!', 'warning')
            return redirect(url_for('dashboard'))
        
        # Önceki session bilgilerini al
        active_coins = monitoring_state.get('active_coins', [])
        interval_minutes = monitoring_state.get('interval_minutes', 15)
        
        # Coinleri veritabanına ekle (eğer yoksa)
        for coin_symbol in active_coins:
            db.add_coin(coin_symbol)
        
        # Monitoring başlat
        coin_monitor.start_monitoring(interval_minutes)
        
        # Persistence güncelle
        persistence.save_monitoring_state(
            is_active=True,
            interval_minutes=interval_minutes,
            active_coins=active_coins,
            session_info={
                'resumed_at': datetime.now().isoformat(),
                'resumed_from_previous_session': True
            }
        )
        
        flash(f'Önceki session resume edildi! {len(active_coins)} coin, {interval_minutes} dakika aralık', 'success')
        logger.info(f"🔄 Previous session resumed: {len(active_coins)} coins")
        
    except Exception as e:
        flash(f'Session resume hatası: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/portfolio')
def portfolio():
    """Portfolio detay sayfası - Binance entegrasyonu ile"""
    try:
        # Portfolio özeti (database)
        summary = db.get_portfolio_summary()
        
        # Açık pozisyonlar (database) 
        positions = db.get_open_positions()
        
        # Son 30 günlük işlemler (database)
        trades = db.get_recent_trades(100)
        
        # Coin performansları (database)
        coins = db.get_active_coins()
        coin_performances = []
        for coin in coins:
            perf = db.get_coin_performance(coin['symbol'], 30)
            coin_performances.append(perf)
        
        # **NEW: Binance gerçek cüzdan verilerini çek**
        binance_data = {}
        try:
            from binance_history import BinanceHistoryFetcher
            
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_SECRET_KEY')
            testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
            
            if api_key and api_secret:
                fetcher = BinanceHistoryFetcher(api_key, api_secret, testnet)
                
                if fetcher.exchange:
                    print(f"💰 Binance Portfolio verileri çekiliyor ({'Testnet' if testnet else 'Mainnet'})...")
                    
                    # Gerçek hesap bilgileri ve bakiyeler
                    account_info = fetcher.fetch_account_info()
                    if 'error' not in account_info:
                        binance_data['account_info'] = account_info
                        binance_data['balances'] = account_info.get('balances', {})
                        
                        # Portfolio değeri hesapla
                        total_portfolio_value = 0
                        portfolio_items = []
                        
                        for currency, amounts in binance_data['balances'].items():
                            if amounts['total'] > 0:
                                # USD değeri hesapla (basit olarak USDT pairs kullan)
                                try:
                                    if currency == 'USDT':
                                        usd_value = amounts['total']
                                    elif currency == 'BTC':
                                        btc_price = data_fetcher.exchange.fetch_ticker('BTC/USDT')['last']
                                        usd_value = amounts['total'] * btc_price
                                    elif currency == 'ETH':
                                        eth_price = data_fetcher.exchange.fetch_ticker('ETH/USDT')['last']
                                        usd_value = amounts['total'] * eth_price
                                    else:
                                        # Diğer coinler için pair kontrol et
                                        try:
                                            ticker = data_fetcher.exchange.fetch_ticker(f'{currency}/USDT')
                                            usd_value = amounts['total'] * ticker['last']
                                        except:
                                            usd_value = 0
                                    
                                    portfolio_items.append({
                                        'symbol': currency,
                                        'amount': amounts['total'],
                                        'usd_value': usd_value
                                    })
                                    total_portfolio_value += usd_value
                                    
                                except Exception as price_error:
                                    print(f"⚠️ {currency} fiyat hesaplama hatası: {price_error}")
                        
                        binance_data['portfolio_items'] = portfolio_items
                        binance_data['total_portfolio_value'] = total_portfolio_value
                        
                        # Summary'yi Binance verileriyle güncelle
                        summary['binance_total_value'] = total_portfolio_value
                        summary['binance_balance_count'] = len(portfolio_items)
                    
                    # Son işlemler
                    real_trades = fetcher.fetch_trade_history(days=7, limit=10)
                    binance_data['real_trades'] = real_trades
                    
                    # Trading özeti
                    trading_summary = fetcher.get_trading_summary(days=30)
                    binance_data['trading_summary'] = trading_summary
                    
                    # 24h değişim hesapla (mevcut bakiyelerle)
                    try:
                        yesterday_value = total_portfolio_value  # Basit hesaplama
                        binance_data['portfolio_change_24h'] = 0  # Gerçek hesaplama için historical data gerekli
                    except:
                        binance_data['portfolio_change_24h'] = 0
                    
                    binance_data['connected'] = True
                    binance_data['testnet'] = testnet
                    
                    print(f"✅ Binance Portfolio: ${total_portfolio_value:.2f} ({len(portfolio_items)} coin)")
                else:
                    binance_data['connected'] = False
                    binance_data['error'] = 'Binance bağlantısı kurulamadı'
            else:
                binance_data['connected'] = False
                binance_data['error'] = 'Binance API anahtarları bulunamadı'
                
        except Exception as e:
            binance_data['connected'] = False
            binance_data['error'] = str(e)
            print(f"❌ Binance Portfolio hatası: {e}")
        
        # **Enhanced Summary with Binance data**
        if binance_data.get('connected', False):
            summary.update({
                'current_value': binance_data.get('total_portfolio_value', summary.get('current_value', 0)),
                'unrealized_pnl': binance_data.get('total_portfolio_value', 0) - summary.get('invested_amount', 0),
                'total_pnl_percent': ((binance_data.get('total_portfolio_value', 0) - summary.get('invested_amount', 0)) / max(summary.get('invested_amount', 1), 1)) * 100,
                'binance_connected': True
            })
        
        return render_template('portfolio.html',
                             summary=summary,
                             positions=positions,
                             trades=trades,
                             coin_performances=coin_performances,
                             binance_data=binance_data)
    except Exception as e:
        flash(f'Portfolio yükleme hatası: {str(e)}', 'error')
        return render_template('portfolio.html',
                             summary={}, positions=[], trades=[], coin_performances=[],
                             binance_data={'connected': False, 'error': str(e)})

@app.route('/settings')
@login_required
def settings():
    """Ayarlar sayfası"""
    return render_template('settings.html')


@app.route('/automation')
@login_required
def automation_dashboard():
    """Otomasyon motoru kokpiti (Market Radar) — keşif/skorlama/sinyal izleme."""
    return render_template('automation.html')


@app.route('/automation/coin/<symbol>')
@login_required
def automation_coin_detail(symbol):
    """Tek coin research detay sayfası."""
    return render_template('automation_coin.html', symbol=symbol.upper())


@app.route('/simulation')
@login_required
def simulation_list():
    """Simülasyon (paper trading) listesi."""
    return render_template('simulation.html')


@app.route('/simulation/new')
@login_required
def simulation_new():
    """Yeni simülasyon başlatma formu."""
    return render_template('simulation_new.html')


@app.route('/simulation/<int:sid>')
@login_required
def simulation_detail(sid):
    """Simülasyon detay sayfası (sekmeler + grafikler)."""
    return render_template('simulation_detail.html', sim_id=sid)


# ─── Nav kısayolları & geriye uyumluluk redirect'leri ────────────────────────────────────
# Hedef sayfalar zaten @login_required; bu alias'lar yalnız yönlendirir (404 yerine anlamlı gidiş).
# Watchlist/Signals/Reports bağımsız tam sayfaları sonraki turda; şimdilik mevcut yüzeylere bağlı.
@app.route('/dashboard')
def dashboard_alias():
    """/dashboard → ana dashboard (/) kısayolu."""
    return redirect(url_for('dashboard'))


@app.route('/automation.html')
def automation_html_alias():
    return redirect(url_for('automation_dashboard'))


@app.route('/simulation.html')
def simulation_html_alias():
    return redirect(url_for('simulation_list'))


@app.route('/signals')
def signals_alias():
    """Sinyaller şimdilik Automation kokpitinde gösteriliyor."""
    return redirect(url_for('automation_dashboard'))


@app.route('/watchlist')
def watchlist_alias():
    """Watchlist şimdilik Automation kokpitinde gösteriliyor."""
    return redirect(url_for('automation_dashboard'))


@app.route('/reports')
def reports_alias():
    """Raporlar şimdilik Simülasyon detay sekmelerinde gösteriliyor."""
    return redirect(url_for('simulation_list'))


@app.route('/api/train_coin', methods=['GET', 'POST'])
def api_train_coin():
    """Coin araştırma/eğitim API'si.

    GET  → kullanım (usage) bilgisi (public; sır içermez).
    POST → JSON gövde ile eğitimi arka planda başlatır (auth zorunlu). Geriye uyum için
           `symbol` | `coin` | `coin_symbol` alanlarından herhangi biri + opsiyonel `force` kabul edilir.
    """
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'endpoint': '/api/train_coin',
            'method': 'POST',
            'content_type': 'application/json',
            'required_body': {'symbol': 'BTC/USDT'},
            'example': {'symbol': 'BTC/USDT', 'force': False}
        })

    # POST — eğitim tetikler → auth zorunlu
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Yetkisiz. Lütfen giriş yapın.'}), 401

    # JSON gövde zorunlu (Content-Type application/json)
    if request.is_json:
        data = request.get_json(silent=True)
    else:
        data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({
            'success': False,
            'error': 'Request must be JSON. Content-Type should be application/json'
        }), 415

    # Geriye uyum: symbol | coin | coin_symbol; "BTC/USDT" | "BTCUSDT" → "BTC"
    raw = str(data.get('symbol') or data.get('coin') or data.get('coin_symbol') or '').strip().upper()
    symbol = raw.split('/')[0].strip()
    if symbol.endswith('USDT') and len(symbol) > 4:
        symbol = symbol[:-4]
    force = bool(data.get('force', False))

    if not symbol:
        return jsonify({'success': False, 'error': 'Coin sembolü gerekli (symbol)'}), 400

    try:
        if not data_fetcher.validate_symbol(symbol):
            return jsonify({'success': False, 'error': f'{symbol} geçerli bir sembol değil'}), 400

        # İzleme listesine ekle (idempotent) + haftalık schedule'a al
        try:
            db.add_coin(symbol, symbol)
            from training_scheduler import get_scheduler
            get_scheduler().add_coin_to_schedule(symbol)
        except Exception as _e:
            logger.warning(f"train_coin: coin ekleme/scheduler uyarısı ({symbol}): {_e}")

        # Eğitimi arka planda başlat (/add_coin ile aynı desen)
        def _bg_train():
            try:
                from comprehensive_trainer import ComprehensiveTrainer
                logger.info(f"🔥 API train_coin: {symbol} eğitimi başlıyor (force={force})...")
                result = ComprehensiveTrainer().train_coin_sync(symbol, is_fine_tune=force)
                socketio.emit('comprehensive_training_complete', {
                    'coin': symbol,
                    'success': bool(result.get('success')) if isinstance(result, dict) else False,
                    'message': f'{symbol} eğitimi tamamlandı',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as te:
                logger.error(f"API train_coin arka plan hatası ({symbol}): {te}")
                socketio.emit('analysis_error', {'coin': symbol, 'error': str(te)})

        threading.Thread(target=_bg_train, daemon=True).start()

        return jsonify({'success': True, 'message': 'Eğitim başlatıldı', 'symbol': symbol})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ─── Market Intelligence API (Faz 4 — minimal; tam API + dashboard Faz 7) ────────────────
@app.route('/api/intelligence/status')
@login_required
def api_intelligence_status():
    """Katman durumu: Ollama health + kaynak sayıları (graceful; Ollama yoksa ok=false)."""
    try:
        from intelligence.engine import get_status
        return jsonify({'success': True, 'status': get_status()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/intelligence/news/<path:symbol>')
@login_required
def api_intelligence_news(symbol):
    """Bir sembol için son snapshot + son haber öğeleri (DB'den; hesaplama yapmaz)."""
    try:
        from intelligence.repository import get_latest_news, get_snapshot
        sym = symbol.upper()
        return jsonify({'success': True, 'symbol': sym,
                        'snapshot': get_snapshot(sym), 'news': get_latest_news(sym, 30)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/intelligence/refresh/<path:symbol>', methods=['POST'])
@login_required
def api_intelligence_refresh(symbol):
    """Bir sembol için market zekâsını yeniden hesapla (collect+LLM+persist). Ağır → istek üzerine."""
    try:
        from intelligence.engine import build_snapshot
        sym = symbol.upper()
        snap = build_snapshot(sym)
        if snap is None:
            return jsonify({'success': True, 'symbol': sym, 'snapshot': None,
                            'message': 'Veri yok / katman kapalı (INTELLIGENCE_ENABLED?)'})
        return jsonify({'success': True, 'symbol': sym, 'snapshot': snap})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/market-intelligence')
@login_required
def market_intelligence():
    """Market Intelligence dashboard (haber kalitesi / hype / event / kaynak sağlığı / Ollama)."""
    return render_template('market_intelligence.html')


@app.route('/api/intelligence/sources')
@login_required
def api_intelligence_sources():
    """Kaynak registry'si (katmanlı source health)."""
    try:
        from intelligence.repository import get_sources
        return jsonify({'success': True, 'sources': get_sources()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/intelligence/overview')
@login_required
def api_intelligence_overview():
    """Piyasa geneli: durum + son snapshot'lar + türetilmiş listeler (yüksek hype / pozitif / negatif)."""
    try:
        from intelligence.engine import get_status
        from intelligence.repository import get_recent_snapshots
        snaps = get_recent_snapshots(60)
        high_hype = sorted([s for s in snaps if s.get('hype_risk') is not None],
                           key=lambda s: s['hype_risk'], reverse=True)[:8]
        positive = [s for s in snaps if s.get('news_sentiment') == 'positive'][:8]
        negative = [s for s in snaps if s.get('news_sentiment') == 'negative'][:8]
        return jsonify({'success': True, 'status': get_status(), 'snapshots': snaps,
                        'high_hype': high_hype, 'positive': positive, 'negative': negative})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/test_news_api')
@login_required
def test_news_api():
    """News API test endpoint'i"""
    try:
        if not coin_monitor.news_analyzer:
            return jsonify({
                'success': False,
                'error': 'News API aktif değil',
                'message': 'NEWSAPI_KEY environment variable\'ı ayarlanmamış'
            })
        
        # Bitcoin haberleri çek (kısa test)
        news_data = coin_monitor.news_analyzer.fetch_all_news("bitcoin", days=1)
        
        if news_data and len(news_data) > 0:
            # İlk 5 haberi göster
            sample_news = []
            for news in news_data[:5]:
                sample_news.append({
                    'title': news.get('title', 'Başlık yok'),
                    'description': news.get('description', 'Açıklama yok')[:100] + '...' if news.get('description') else 'Açıklama yok',
                    'published_at': news.get('publishedAt', 'Tarih yok'),
                    'source': news.get('source', {}).get('name', 'Bilinmeyen kaynak')
                })
            
            return jsonify({
                'success': True,
                'total_news': len(news_data),
                'sample_news': sample_news,
                'message': f'{len(news_data)} haber başarıyla çekildi'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Hiç haber bulunamadı',
                'message': 'News API\'den veri çekilemedi'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'News API test hatası'
        })

@app.route('/test_binance_api')
@login_required
def test_binance_api():
    """Binance API test endpoint'i"""
    try:
        from binance_history import BinanceHistoryFetcher
        
        # Environment'den API bilgilerini al
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_SECRET_KEY')
        testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        if not api_key or not api_secret:
            return jsonify({
                'success': False,
                'error': 'Binance API anahtarları bulunamadı',
                'message': 'BINANCE_API_KEY ve BINANCE_SECRET_KEY environment variable\'ları ayarlanmamış'
            })
        
        # Binance bağlantısı test et
        fetcher = BinanceHistoryFetcher(api_key, api_secret, testnet)
        
        if not fetcher.exchange:
            return jsonify({
                'success': False,
                'error': 'Binance bağlantısı kurulamadı',
                'message': 'API anahtarları geçersiz olabilir'
            })
        
        # Hesap bilgilerini çek
        account_info = fetcher.fetch_account_info()
        
        if 'error' in account_info:
            return jsonify({
                'success': False,
                'error': account_info['error'],
                'message': 'Hesap bilgileri çekilemedi'
            })
        
        # Trading özeti çek
        trading_summary = fetcher.get_trading_summary(days=7)
        
        return jsonify({
            'success': True,
            'account_info': {
                'total_balances': account_info.get('total_balances', 0),
                'account_type': account_info.get('account_type', 'UNKNOWN'),
                'testnet': testnet
            },
            'trading_summary': trading_summary,
            'message': f'Binance API başarıyla çalışıyor ({"Testnet" if testnet else "Mainnet"})'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Binance API test hatası'
        })

@app.route('/api/portfolio_summary')
def api_portfolio_summary():
    """Portfolio özeti API"""
    try:
        summary = db.get_portfolio_summary()
        return jsonify({'success': True, 'data': summary})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/coin_list')
def api_coin_list():
    """Coin listesi API"""
    try:
        coins = db.get_active_coins()
        return jsonify({'success': True, 'data': coins})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recent_trades')
def api_recent_trades():
    """Son işlemler API"""
    try:
        limit = request.args.get('limit', 20, type=int)
        trades = db.get_recent_trades(limit)
        return jsonify({'success': True, 'data': trades})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/portfolio')
def api_portfolio():
    """Portfolio detay API - Binance API entegrasyonu ile"""
    try:
        # Önce database'den verileri al
        db_summary = db.get_portfolio_summary()
        positions = db.get_open_positions()
        recent_trades = db.get_recent_trades(20)
        
        # Binance API'den gerçek cüzdan verilerini çek
        binance_data = {}
        try:
            from binance_history import BinanceHistoryFetcher
            
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_SECRET_KEY')
            testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
            
            if api_key and api_secret:
                fetcher = BinanceHistoryFetcher(api_key, api_secret, testnet)
                
                if fetcher.exchange:
                    # Gerçek hesap bilgileri
                    account_info = fetcher.fetch_account_info()
                    if 'error' not in account_info:
                        binance_data['account_info'] = account_info
                        binance_data['balances'] = account_info.get('balances', {})
                    
                    # Son işlemler
                    real_trades = fetcher.fetch_trade_history(days=7, limit=10)
                    binance_data['real_trades'] = real_trades
                    
                    # Trading özeti
                    trading_summary = fetcher.get_trading_summary(days=30)
                    binance_data['trading_summary'] = trading_summary
                    
                    binance_data['connected'] = True
                    binance_data['testnet'] = testnet
                else:
                    binance_data['connected'] = False
                    binance_data['error'] = 'Binance bağlantısı kurulamadı'
            else:
                binance_data['connected'] = False
                binance_data['error'] = 'Binance API anahtarları bulunamadı'
                
        except Exception as e:
            binance_data['connected'] = False
            binance_data['error'] = str(e)
        
        # Coin performansları
        coins = db.get_active_coins()
        coin_performances = []
        for coin in coins:
            try:
                perf = db.get_coin_performance(coin['symbol'], 7)
                coin_performances.append(perf)
            except:
                coin_performances.append({
                    'symbol': coin['symbol'],
                    'current_price': coin.get('current_price', 0),
                    'price_change_24h': coin.get('price_change_24h', 0),
                    'profit_loss': 0,
                    'profit_loss_percentage': 0
                })
        
        return jsonify({
            'success': True,
            'data': {
                'summary': db_summary,
                'positions': positions,
                'recent_trades': recent_trades,
                'coin_performances': coin_performances,
                'binance_real_data': binance_data,  # Gerçek Binance verileri
                'monitoring_active': coin_monitor.running,
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/close_position', methods=['POST'])
def api_close_position():
    """Pozisyon kapatma API"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        position_type = data.get('position_type')
        
        # Güncel fiyatı al
        df = data_fetcher.fetch_ohlcv_data(symbol)
        current_price = df['close'].iloc[-1] if df is not None else 0
        
        result = db.close_position(symbol, position_type, current_price, 'Manual close via web')
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/toggle_auto_trading', methods=['POST'])
@login_required
def api_toggle_auto_trading():
    """Otomatik trading açma/kapama API"""
    try:
        data = request.get_json()
        enabled = data.get('enabled', False)
        
        # Global konfigürasyonu güncelle
        TRADING_CONFIG['auto_trading_enabled'] = enabled
        
        # Persistence'a kaydet
        persistence.save_system_state('auto_trading_config', TRADING_CONFIG)
        
        status = "aktif" if enabled else "pasif"
        logger.info(f"🤖 Otomatik trading {status} edildi")
        
        return jsonify({
            'success': True,
            'auto_trading_enabled': enabled,
            'message': f'Otomatik trading {status} edildi',
            'config': {
                'minimum_profit_threshold': TRADING_CONFIG['minimum_profit_threshold'],
                'minimum_trade_amount': TRADING_CONFIG['minimum_trade_amount'],
                'max_concurrent_trades': TRADING_CONFIG['max_concurrent_trades'],
                'risk_percentages': TRADING_CONFIG['risk_percentages']
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading_config', methods=['GET', 'POST'])
@login_required
def api_trading_config():
    """Trading konfigürasyonu API"""
    try:
        if request.method == 'GET':
            return jsonify({
                'success': True,
                'config': TRADING_CONFIG
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            
            # Güvenli güncelleme
            if 'minimum_profit_threshold' in data:
                TRADING_CONFIG['minimum_profit_threshold'] = float(data['minimum_profit_threshold'])
            if 'minimum_trade_amount' in data:
                TRADING_CONFIG['minimum_trade_amount'] = float(data['minimum_trade_amount'])
            if 'max_concurrent_trades' in data:
                TRADING_CONFIG['max_concurrent_trades'] = int(data['max_concurrent_trades'])
            if 'stop_loss_percentage' in data:
                TRADING_CONFIG['stop_loss_percentage'] = float(data['stop_loss_percentage'])
            if 'take_profit_multiplier' in data:
                TRADING_CONFIG['take_profit_multiplier'] = float(data['take_profit_multiplier'])
            
            # Persistence'a kaydet
            persistence.save_system_state('auto_trading_config', TRADING_CONFIG)
            
            return jsonify({
                'success': True,
                'message': 'Trading konfigürasyonu güncellendi',
                'config': TRADING_CONFIG
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading_status')
@login_required
def api_trading_status():
    """Trading durumu API"""
    try:
        # USDT bakiyesi
        usdt_balance = coin_monitor._get_usdt_balance()
        
        # Aktif işlemler
        open_positions = db.get_open_positions()
        
        # Son 24 saatin işlemleri
        recent_trades = db.get_recent_trades(50)
        
        return jsonify({
            'success': True,
            'status': {
                'auto_trading_enabled': TRADING_CONFIG['auto_trading_enabled'],
                'usdt_balance': usdt_balance,
                'active_positions': len(open_positions),
                'open_positions': open_positions,
                'recent_trades_count': len(recent_trades),
                'monitoring_active': coin_monitor.running,
                'config': TRADING_CONFIG
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/scheduler_status')
@login_required
def api_scheduler_status():
    """Training Scheduler durumu API'si"""
    try:
        global training_scheduler
        
        if training_scheduler is None:
            return jsonify({
                'success': False,
                'error': 'Training scheduler mevcut değil'
            })
        
        status = training_scheduler.get_scheduler_status()
        
        return jsonify({
            'success': True,
            'scheduler_status': status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/scheduler_control', methods=['POST'])
@login_required  
def api_scheduler_control():
    """Training Scheduler kontrol API'si"""
    try:
        global training_scheduler
        
        if training_scheduler is None:
            return jsonify({
                'success': False,
                'error': 'Training scheduler mevcut değil'
            })
        
        action = request.json.get('action')
        coin_symbol = request.json.get('coin_symbol')
        
        if action == 'start':
            training_scheduler.start_scheduler()
            return jsonify({
                'success': True,
                'message': 'Training scheduler başlatıldı'
            })
            
        elif action == 'stop':
            training_scheduler.stop_scheduler()
            return jsonify({
                'success': True,
                'message': 'Training scheduler durduruldu'
            })
            
        elif action == 'force_run':
            if coin_symbol:
                # Tek coin için manual training
                result = training_scheduler.force_run_training(coin_symbol.upper())
                return jsonify({
                    'success': True,
                    'message': f'{coin_symbol} için manual training başlatıldı',
                    'training_result': result
                })
            else:
                # Tüm coinler için manual training
                training_scheduler.force_run_training()
                return jsonify({
                    'success': True,
                    'message': 'Tüm coinler için manual training başlatıldı'
                })
                
        elif action == 'add_coin':
            if coin_symbol:
                training_scheduler.add_coin_to_schedule(coin_symbol.upper())
                return jsonify({
                    'success': True,
                    'message': f'{coin_symbol} scheduler\'a eklendi'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Coin sembolü gerekli'
                })
                
        elif action == 'remove_coin':
            if coin_symbol:
                training_scheduler.remove_coin_from_schedule(coin_symbol.upper())
                return jsonify({
                    'success': True,
                    'message': f'{coin_symbol} scheduler\'dan çıkarıldı'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Coin sembolü gerekli'
                })
        else:
            return jsonify({
                'success': False,
                'error': 'Geçersiz action'
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# WebSocket Events
@socketio.on('connect')
def on_connect():
    """Client bağlantısı"""
    emit('connected', {'message': 'Trading Dashboard\'a bağlandınız!'})

@socketio.on('request_update')
def on_request_update():
    """Güncel veri talebi"""
    try:
        # Portfolio özeti
        portfolio = db.get_portfolio_summary()
        
        # Aktif coinler
        coins = db.get_active_coins()
        
        # Son işlemler
        recent_trades = db.get_recent_trades(5)
        
        emit('dashboard_update', {
            'portfolio': portfolio,
            'coins': coins,
            'recent_trades': recent_trades,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        emit('error', {'message': str(e)})

def main():
    """Ana fonksiyon - Environment variables ve persistence destekli"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║            🌐 KRİPTO TRADİNG DASHBOARD WEB UYGULAMASI 🌐          ║
║                                                                    ║
║  📊 Çoklu coin izleme                    🗄️ PostgreSQL DB        ║
║  💰 İşlem geçmişi takibi                 🔐 Environment Vars     ║
║  📈 Kar/zarar analizi                    💾 State Persistence    ║
║  🤖 Otomatik trading                     🔄 Auto Resume          ║
║  📱 Gerçek zamanlı güncelleme                                     ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    try:
        # Environment variables kontrol
        print(f"🗄️ Database: {DATABASE_TYPE}")
        _db_url = os.getenv('DATABASE_URL', '')
        if _db_url:
            # host bilgisini güvenli şekilde göster (kimlik bilgisi olmadan)
            try:
                print(f"   📍 DB Host: {_db_url.rsplit('@', 1)[-1]}")
            except Exception:
                pass
        
        # System startup summary
        startup_summary = persistence.get_startup_summary()
        print("📋 System Status:")
        print(f"   🔧 Session ID: {startup_summary['session_id']}")
        print(f"   📊 Monitoring coins: {startup_summary['monitoring']['active_coins_count']}")
        print(f"   💰 Trading enabled: {startup_summary['trading']['enabled']}")
        print(f"   🔑 APIs configured: {sum(startup_summary['apis'].values())}")
        
        # Auto-resume previous session check
        if startup_summary['monitoring']['should_resume']:
            print("🔄 Önceki session restore edilebilir!")
            print("   ➡️ Dashboard'da 'Resume Previous Session' butonunu kullanın")
        
        # Cache temizleme
        if CACHE_AVAILABLE:
            cache_manager = CachedModelManager()
            cache_manager.cleanup_old_models()
            print("🧹 Model cache temizlendi")
        
        # Test coinleri ekle (sadece ilk kez)
        existing_coins = db.get_active_coins()
        if len(existing_coins) == 0:
            print("🧪 Test coinleri ekleniyor...")
            db.add_coin('BTC', 'Bitcoin')
            db.add_coin('ETH', 'Ethereum')
            db.add_coin('BNB', 'Binance Coin')
        
        # Flask host ve port ayarları
        host = os.getenv('FLASK_HOST', '0.0.0.0')
        port = int(os.getenv('FLASK_PORT', '5002'))
        debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
        
        print("✅ Dashboard hazır!")
        print(f"🌐 http://localhost:{port} adresine gidin")
        print("📊 Dashboard: Ana sayfa")
        print(f"💰 Portfolio: http://localhost:{port}/portfolio") 
        print(f"⚙️ Settings: http://localhost:{port}/settings")
        print("🔴 Durdurmak için Ctrl+C")

        # Automation engine scheduler (AUTO_DISCOVERY_ENABLED ile; canlı trade default KAPALI)
        try:
            from automation.scheduler import get_scheduler
            get_scheduler().start()
        except Exception as _ae:
            print(f"⚠️ Automation scheduler başlatılamadı: {_ae}")

        # Flask uygulamasını başlat
        socketio.run(app, host=host, port=port, debug=debug)
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard kapatılıyor...")
        
        # Monitoring durdur ve state kaydet
        if coin_monitor.running:
            coin_monitor.stop_monitoring()
            active_coins = [coin['symbol'] for coin in db.get_active_coins()]
            persistence.save_monitoring_state(
                is_active=False,
                interval_minutes=coin_monitor.monitoring_interval,
                active_coins=active_coins,
                session_info={
                    'shutdown_reason': 'user_interrupt',
                    'shutdown_time': datetime.now().isoformat()
                }
            )
            print("💾 Session durumu kaydedildi")
        
        print("✅ Temiz kapatma tamamlandı")
        
    except Exception as e:
        print(f"❌ Başlatma hatası: {str(e)}")
        logger.error(f"Application startup error: {str(e)}")

if __name__ == '__main__':
    main() 