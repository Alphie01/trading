"""Sistem/oturum durumu kalıcılığı (tenant-aware, PostgreSQL).

Eski ``system_persistence.py`` public API'sini korur. State, aktif tenant'ın
``system_state`` tablosuna yazılır (repository üzerinden). Flask istek bağlamında
aktif tenant set olduğu için çalışır; bağlam yoksa güvenli varsayılanlar döner.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Dict, List

from .repository import TradingDatabase


class SystemPersistence:
    """Sistem durumu kalıcılığı yönetimi."""

    def __init__(self, use_mssql: bool = True):
        # use_mssql eski imza uyumu için tutulur (kullanılmaz).
        self.db = TradingDatabase()
        self.db_type = "PostgreSQL"
        self.session_id = str(uuid.uuid4())[:8]
        print(f"🔄 SystemPersistence başlatıldı ({self.db_type})")

        self.KEYS = {
            "active_monitoring": "dashboard_active_monitoring",
            "monitoring_interval": "dashboard_monitoring_interval",
            "active_coins": "dashboard_active_coins",
            "last_session": "dashboard_last_session",
            "trading_enabled": "trading_auto_enabled",
            "trading_settings": "trading_settings",
            "api_keys_configured": "api_keys_configured",
            "system_config": "system_configuration",
        }

    def save_monitoring_state(
        self, is_active: bool, interval_minutes: int, active_coins: List[str], session_info: Dict = None
    ):
        try:
            self.db.save_system_state(self.KEYS["active_monitoring"], is_active)
            self.db.save_system_state(self.KEYS["monitoring_interval"], interval_minutes)
            self.db.save_system_state(self.KEYS["active_coins"], active_coins)
            session_data = {
                "session_id": self.session_id,
                "start_time": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "coin_count": len(active_coins),
                "monitoring_active": is_active,
                "monitoring_interval": interval_minutes,
            }
            if session_info:
                session_data.update(session_info)
            self.db.save_system_state(self.KEYS["last_session"], session_data)
            print(f"💾 Monitoring durumu kaydedildi: {len(active_coins)} coin, {interval_minutes}min")
        except Exception as e:
            print(f"❌ Monitoring state kaydetme hatası: {str(e)}")

    def load_monitoring_state(self) -> Dict:
        try:
            state = {
                "is_active": self.db.load_system_state(self.KEYS["active_monitoring"], False),
                "interval_minutes": self.db.load_system_state(self.KEYS["monitoring_interval"], 15),
                "active_coins": self.db.load_system_state(self.KEYS["active_coins"], []),
                "last_session": self.db.load_system_state(self.KEYS["last_session"], {}),
                "should_resume": False,
            }
            last_session = state["last_session"]
            if last_session and "start_time" in last_session:
                try:
                    last_time = datetime.fromisoformat(last_session["start_time"])
                    if datetime.now() - last_time < timedelta(hours=24):
                        state["should_resume"] = True
                except Exception:
                    pass
            if state["should_resume"]:
                print(f"🔄 Önceki monitoring session bulundu: {len(state['active_coins'])} coin")
            return state
        except Exception as e:
            print(f"❌ Monitoring state yükleme hatası: {str(e)}")
            return {
                "is_active": False,
                "interval_minutes": 15,
                "active_coins": [],
                "last_session": {},
                "should_resume": False,
            }

    def save_trading_state(self, trading_enabled: bool, settings: Dict):
        try:
            self.db.save_system_state(self.KEYS["trading_enabled"], trading_enabled)
            self.db.save_system_state(self.KEYS["trading_settings"], settings)
            print(f"💾 Trading durumu kaydedildi: enabled={trading_enabled}")
        except Exception as e:
            print(f"❌ Trading state kaydetme hatası: {str(e)}")

    def load_trading_state(self) -> Dict:
        try:
            return {
                "trading_enabled": self.db.load_system_state(self.KEYS["trading_enabled"], False),
                "settings": self.db.load_system_state(self.KEYS["trading_settings"], {}),
            }
        except Exception as e:
            print(f"❌ Trading state yükleme hatası: {str(e)}")
            return {"trading_enabled": False, "settings": {}}

    def save_api_configuration(self, api_config: Dict):
        try:
            config_status = {
                "binance_configured": bool(api_config.get("binance_api_key")),
                "newsapi_configured": bool(api_config.get("newsapi_key")),
                "whale_alert_configured": bool(api_config.get("whale_alert_key")),
                "last_updated": datetime.now().isoformat(),
            }
            self.db.save_system_state(self.KEYS["api_keys_configured"], config_status)
            print("🔑 API konfigürasyonu kaydedildi")
        except Exception as e:
            print(f"❌ API config kaydetme hatası: {str(e)}")

    def load_api_configuration(self) -> Dict:
        try:
            return {"api_status": self.db.load_system_state(self.KEYS["api_keys_configured"], {})}
        except Exception as e:
            print(f"❌ API config yükleme hatası: {str(e)}")
            return {"api_status": {}}

    def get_startup_summary(self) -> Dict:
        try:
            monitoring_state = self.load_monitoring_state()
            trading_state = self.load_trading_state()
            api_config = self.load_api_configuration()
            summary = {
                "session_id": self.session_id,
                "database_type": self.db_type,
                "startup_time": datetime.now().isoformat(),
                "monitoring": {
                    "should_resume": monitoring_state["should_resume"],
                    "active_coins_count": len(monitoring_state["active_coins"]),
                    "interval_minutes": monitoring_state["interval_minutes"],
                    "was_active": monitoring_state["is_active"],
                },
                "trading": {"enabled": trading_state["trading_enabled"]},
                "apis": {
                    "binance": api_config["api_status"].get("binance_configured", False),
                    "newsapi": api_config["api_status"].get("newsapi_configured", False),
                    "whale_alert": api_config["api_status"].get("whale_alert_configured", False),
                },
            }
            print(f"📋 Startup summary hazırlandı: {self.db_type} database")
            return summary
        except Exception as e:
            print(f"❌ Startup summary hatası: {str(e)}")
            return {
                "session_id": self.session_id,
                "database_type": self.db_type,
                "startup_time": datetime.now().isoformat(),
                "error": str(e),
            }

    def restore_previous_session(self) -> bool:
        try:
            monitoring_state = self.load_monitoring_state()
            if not monitoring_state["should_resume"]:
                print("ℹ️ Resume edilecek önceki session bulunamadı")
                return False
            last_session = monitoring_state["last_session"]
            active_coins = monitoring_state["active_coins"]
            print("🔄 Önceki session restore ediliyor:")
            print(f"   - Session ID: {last_session.get('session_id', 'unknown')}")
            print(f"   - Coin sayısı: {len(active_coins)}")
            print(f"   - Monitoring interval: {monitoring_state['interval_minutes']} dakika")
            print(f"   - Son aktivite: {last_session.get('last_activity', 'unknown')}")
            return True
        except Exception as e:
            print(f"❌ Session restore hatası: {str(e)}")
            return False
