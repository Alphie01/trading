"""Birleşik veri erişim katmanı (repository).

``TradingDatabase`` sınıfı, eski SQLite ``TradingDatabase``'in public metod
adlarını, imzalarını ve DÖNÜŞ ŞEKİLLERİNİ birebir korur → ~71 çağrı noktası
değişmeden çalışır. İçeride PostgreSQL + SQLAlchemy + schema-per-tenant kullanır.

Sınır kuralı (API uyumu + Decimal/JSON riski):
- Numeric (Decimal) alanlar dış katmana ``float`` olarak döner (eski davranış).
- DateTime alanlar ISO string olarak döner (eski SQLite string davranışı).
- JSONB alanlar Python nesnesi olarak döner (eski json.loads davranışı).

Tenant izolasyonu:
- Tenant-özel tablolar (trades/positions/analysis_results/system_state/watchlist)
  aktif tenant contextvar'ına göre çalışır. Bağlam yoksa yazma güvenle atlanır,
  okuma boş döner (fail-safe, cross-tenant sızıntı olmaz).
- Ortak tablolar (coins kataloğu, prediction_cache) tenant bağlamı gerektirmez.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, text

from .models_shared import Coin, PredictionCache
from .models_tenant import (
    AnalysisResult,
    Position,
    SystemState,
    Trade,
    Watchlist,
)
from .session import get_session, get_tenant_session
from .tenancy import get_current_tenant


# --------------------------------------------------------------------------- #
# Sınır dönüşüm yardımcıları (API uyumu)
# --------------------------------------------------------------------------- #
def _f(v: Any) -> Any:
    """Decimal → float; None korunur."""
    if isinstance(v, Decimal):
        return float(v)
    return v


def _dt(v: Any) -> Any:
    """datetime → ISO string; None korunur."""
    if isinstance(v, datetime):
        return v.isoformat()
    return v


class TradingDatabase:
    """PostgreSQL + SQLAlchemy tabanlı, tenant-aware veri erişim katmanı."""

    def __init__(self, db_path: str = None):
        # db_path eski imza uyumu için tutulur (kullanılmaz).
        self.db_path = db_path

    # ------------------------------------------------------------------ #
    # Tenant bağlam yardımcıları
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tenant() -> Optional[str]:
        return get_current_tenant()

    # ================================================================== #
    # COINS — katalog (shared) + izleme (tenant.watchlist)
    # ================================================================== #
    def add_coin(self, symbol: str, name: str = None) -> bool:
        """Coin'i kataloğa (shared) ve tenant izleme listesine ekler (upsert)."""
        try:
            symbol = symbol.upper()
            # 1) Ortak katalog upsert
            with get_session() as s:
                coin = s.execute(
                    select(Coin).where(Coin.symbol == symbol)
                ).scalar_one_or_none()
                if coin is None:
                    s.add(Coin(symbol=symbol, name=name or symbol))
                elif name and not coin.name:
                    coin.name = name
            # 2) Tenant izleme upsert (bağlam varsa)
            schema = self._tenant()
            if schema:
                with get_tenant_session(schema) as s:
                    w = s.execute(
                        select(Watchlist).where(Watchlist.coin_symbol == symbol)
                    ).scalar_one_or_none()
                    if w is None:
                        s.add(Watchlist(coin_symbol=symbol, is_active=True))
                    else:
                        w.is_active = True
            print(f"✅ {symbol} coin listesine eklendi")
            return True
        except Exception as e:
            print(f"❌ Coin ekleme hatası: {str(e)}")
            return False

    def remove_coin(self, symbol: str) -> bool:
        """Coin'i tenant izleme listesinden çıkarır (soft, is_active=0)."""
        try:
            schema = self._tenant()
            if not schema:
                return False
            with get_tenant_session(schema) as s:
                w = s.execute(
                    select(Watchlist).where(Watchlist.coin_symbol == symbol.upper())
                ).scalar_one_or_none()
                if w:
                    w.is_active = False
            print(f"🗑️ {symbol} izleme listesinden çıkarıldı")
            return True
        except Exception as e:
            print(f"❌ Coin çıkarma hatası: {str(e)}")
            return False

    def get_active_coins(self) -> List[Dict]:
        """Tenant'ın aktif izlediği coinleri (katalog bilgisiyle) döndürür."""
        schema = self._tenant()
        if not schema:
            return []
        try:
            with get_tenant_session(schema) as s:
                rows = s.execute(
                    select(
                        Watchlist.coin_symbol,
                        Coin.name,
                        Watchlist.added_date,
                        Watchlist.last_analysis,
                        Coin.current_price,
                        Coin.price_change_24h,
                        Watchlist.analysis_count,
                    )
                    .join(Coin, Coin.symbol == Watchlist.coin_symbol, isouter=True)
                    .where(Watchlist.is_active.is_(True))
                    .order_by(Watchlist.added_date.desc())
                ).all()
                return [
                    {
                        "symbol": r[0],
                        "name": r[1],
                        "added_date": _dt(r[2]),
                        "last_analysis": _dt(r[3]),
                        "current_price": _f(r[4]),
                        "price_change_24h": _f(r[5]),
                        "analysis_count": r[6],
                    }
                    for r in rows
                ]
        except Exception as e:
            print(f"❌ Coin listesi alma hatası: {str(e)}")
            return []

    def get_catalog_coins(self) -> List[Dict]:
        """Ortak coin kataloğunun tamamı (eğitim/scheduler için, tenant'tan bağımsız)."""
        try:
            with get_session() as s:
                rows = s.execute(
                    select(Coin.symbol, Coin.name, Coin.current_price).order_by(Coin.symbol)
                ).all()
                return [
                    {"symbol": r[0], "name": r[1], "current_price": _f(r[2])} for r in rows
                ]
        except Exception as e:
            print(f"❌ Katalog alma hatası: {str(e)}")
            return []

    # Scheduler uyumu (eski buglı get_tracked_coins → ortak katalog)
    def get_tracked_coins(self) -> List[str]:
        return [c["symbol"] for c in self.get_catalog_coins()]

    def update_coin_price(self, symbol: str, current_price: float = None, price_change_24h: float = None) -> bool:
        """Ortak katalogdaki fiyatı ve (bağlam varsa) tenant izleme 'last_analysis'ini günceller."""
        try:
            sym = symbol.upper()
            with get_session() as s:
                coin = s.execute(select(Coin).where(Coin.symbol == sym)).scalar_one_or_none()
                if coin:
                    if current_price is not None:
                        coin.current_price = current_price
                    if price_change_24h is not None:
                        coin.price_change_24h = price_change_24h
            schema = self._tenant()
            if schema:
                with get_tenant_session(schema) as s:
                    w = s.execute(
                        select(Watchlist).where(Watchlist.coin_symbol == sym)
                    ).scalar_one_or_none()
                    if w:
                        w.last_analysis = datetime.now(timezone.utc)
            return True
        except Exception as e:
            print(f"❌ Coin fiyat güncelleme hatası: {str(e)}")
            return False

    # ================================================================== #
    # TRADES (tenant)
    # ================================================================== #
    def record_trade(
        self,
        coin_symbol: str,
        trade_type: str,
        price: float,
        quantity: float,
        confidence: float = None,
        news_sentiment: float = None,
        whale_activity: float = None,
        yigit_signal: str = None,
        trade_reason: str = None,
        fees: float = 0,
        is_simulated: bool = True,
    ) -> Optional[int]:
        schema = self._tenant()
        if not schema:
            print("⚠️ record_trade: aktif tenant yok, atlandı")
            return None
        try:
            total_value = price * quantity
            with get_tenant_session(schema) as s:
                t = Trade(
                    coin_symbol=coin_symbol.upper(),
                    trade_type=trade_type.upper(),
                    price=price,
                    quantity=quantity,
                    total_value=total_value,
                    confidence=confidence,
                    news_sentiment=news_sentiment,
                    whale_activity=whale_activity,
                    yigit_signal=yigit_signal,
                    trade_reason=trade_reason,
                    fees=fees,
                    is_simulated=is_simulated,
                )
                s.add(t)
                s.flush()
                trade_id = t.id
            print(f"💰 İşlem kaydedildi: {trade_type} {quantity} {coin_symbol} @ ${price:.6f}")
            return trade_id
        except Exception as e:
            print(f"❌ İşlem kaydetme hatası: {str(e)}")
            return None

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        schema = self._tenant()
        if not schema:
            return []
        try:
            with get_tenant_session(schema) as s:
                rows = s.execute(
                    select(
                        Trade.coin_symbol,
                        Trade.trade_type,
                        Trade.price,
                        Trade.quantity,
                        Trade.total_value,
                        Trade.timestamp,
                        Trade.confidence,
                        Trade.trade_reason,
                        Trade.fees,
                        Trade.is_simulated,
                    )
                    .order_by(Trade.timestamp.desc())
                    .limit(limit)
                ).all()
                return [
                    {
                        "coin_symbol": r[0],
                        "trade_type": r[1],
                        "price": _f(r[2]),
                        "quantity": _f(r[3]),
                        "total_value": _f(r[4]),
                        "timestamp": _dt(r[5]),
                        "confidence": _f(r[6]),
                        "trade_reason": r[7],
                        "fees": _f(r[8]),
                        "is_simulated": r[9],
                    }
                    for r in rows
                ]
        except Exception as e:
            print(f"❌ İşlem geçmişi alma hatası: {str(e)}")
            return []

    # ================================================================== #
    # POSITIONS (tenant)
    # ================================================================== #
    def update_position(
        self,
        coin_symbol: str,
        position_type: str,
        entry_price: float,
        quantity: float,
        current_price: float = None,
        leverage: float = 1,
        stop_loss: float = None,
        take_profit: float = None,
    ) -> Optional[int]:
        schema = self._tenant()
        if not schema:
            print("⚠️ update_position: aktif tenant yok, atlandı")
            return None
        try:
            entry_value = entry_price * quantity
            cur = current_price or entry_price
            current_value = cur * quantity
            if position_type.upper() == "SHORT":
                unrealized_pnl = (entry_price - cur) * quantity * leverage
            else:
                unrealized_pnl = (cur - entry_price) * quantity * leverage

            with get_tenant_session(schema) as s:
                existing = s.execute(
                    select(Position).where(
                        Position.coin_symbol == coin_symbol.upper(),
                        Position.position_type == position_type.upper(),
                        Position.is_open.is_(True),
                    )
                ).scalar_one_or_none()

                if existing:
                    existing.current_price = cur
                    existing.current_value = current_value
                    existing.unrealized_pnl = unrealized_pnl
                    existing.last_update = datetime.now(timezone.utc)
                    s.flush()
                    position_id = existing.id
                else:
                    p = Position(
                        coin_symbol=coin_symbol.upper(),
                        position_type=position_type.upper(),
                        entry_price=entry_price,
                        current_price=cur,
                        quantity=quantity,
                        entry_value=entry_value,
                        current_value=current_value,
                        unrealized_pnl=unrealized_pnl,
                        leverage=leverage,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    )
                    s.add(p)
                    s.flush()
                    position_id = p.id
            print(f"📈 Pozisyon güncellendi: {position_type} {coin_symbol} PnL: ${unrealized_pnl:.2f}")
            return position_id
        except Exception as e:
            print(f"❌ Pozisyon güncelleme hatası: {str(e)}")
            return None

    def close_position(
        self, coin_symbol: str, position_type: str, exit_price: float, exit_reason: str = None
    ) -> Dict:
        schema = self._tenant()
        if not schema:
            return {"success": False, "message": "Aktif tenant yok"}
        try:
            with get_tenant_session(schema) as s:
                pos = s.execute(
                    select(Position).where(
                        Position.coin_symbol == coin_symbol.upper(),
                        Position.position_type == position_type.upper(),
                        Position.is_open.is_(True),
                    )
                ).scalar_one_or_none()
                if not pos:
                    return {"success": False, "message": "Açık pozisyon bulunamadı"}

                entry_price = float(pos.entry_price)
                quantity = float(pos.quantity)
                leverage = float(pos.leverage or 1)
                entry_value = float(pos.entry_value or (entry_price * quantity))

                if position_type.upper() == "SHORT":
                    realized_pnl = (entry_price - exit_price) * quantity * leverage
                else:
                    realized_pnl = (exit_price - entry_price) * quantity * leverage
                pnl_percent = (realized_pnl / entry_value) * 100 if entry_value else 0

                pos.is_open = False
                pos.current_price = exit_price
                pos.unrealized_pnl = realized_pnl
                pos.last_update = datetime.now(timezone.utc)
                s.flush()

            # Kapanış işlemini kaydet (aynı tenant)
            self.record_trade(
                coin_symbol,
                f"CLOSE_{position_type}",
                exit_price,
                quantity,
                trade_reason=f'Position closed: {exit_reason or "Manual"}',
            )

            result = {
                "success": True,
                "realized_pnl": realized_pnl,
                "pnl_percent": pnl_percent,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "leverage": leverage,
            }
            print(
                f"🏁 Pozisyon kapatıldı: {position_type} {coin_symbol} "
                f"PnL: ${realized_pnl:.2f} ({pnl_percent:+.2f}%)"
            )
            return result
        except Exception as e:
            print(f"❌ Pozisyon kapatma hatası: {str(e)}")
            return {"success": False, "message": str(e)}

    def get_open_positions(self) -> List[Dict]:
        schema = self._tenant()
        if not schema:
            return []
        try:
            with get_tenant_session(schema) as s:
                rows = s.execute(
                    select(
                        Position.coin_symbol,
                        Position.position_type,
                        Position.entry_price,
                        Position.current_price,
                        Position.quantity,
                        Position.entry_value,
                        Position.current_value,
                        Position.unrealized_pnl,
                        Position.entry_timestamp,
                        Position.leverage,
                        Position.stop_loss,
                        Position.take_profit,
                    )
                    .where(Position.is_open.is_(True))
                    .order_by(Position.entry_timestamp.desc())
                ).all()
                positions = []
                for r in rows:
                    entry_value = float(r[5]) if r[5] is not None else 1
                    pnl = float(r[7]) if r[7] is not None else 0
                    pnl_percent = (pnl / (entry_value or 1)) * 100
                    positions.append(
                        {
                            "coin_symbol": r[0],
                            "position_type": r[1],
                            "entry_price": _f(r[2]),
                            "current_price": _f(r[3]),
                            "quantity": _f(r[4]),
                            "entry_value": _f(r[5]),
                            "current_value": _f(r[6]),
                            "unrealized_pnl": _f(r[7]),
                            "pnl_percent": pnl_percent,
                            "entry_timestamp": _dt(r[8]),
                            "leverage": _f(r[9]),
                            "stop_loss": _f(r[10]),
                            "take_profit": _f(r[11]),
                        }
                    )
                return positions
        except Exception as e:
            print(f"❌ Açık pozisyon alma hatası: {str(e)}")
            return []

    # web_service.py uyumu (eski MSSQL-only): kullanıcının pozisyonları = tenant açık pozisyonları
    def get_user_positions(self, user_id: str = None) -> List[Dict]:
        return self.get_open_positions()

    # ================================================================== #
    # ANALYSIS (tenant)
    # ================================================================== #
    def save_analysis_result(
        self,
        coin_symbol: str,
        prediction_result: Dict,
        news_analysis: Dict = None,
        whale_analysis: Dict = None,
        yigit_analysis: Dict = None,
        model_type: str = "LSTM",
    ) -> Optional[int]:
        schema = self._tenant()
        if not schema:
            print("⚠️ save_analysis_result: aktif tenant yok, atlandı")
            return None
        try:
            features_used = {
                "news": news_analysis is not None,
                "whale": whale_analysis is not None,
                "yigit": yigit_analysis is not None,
                "technical": True,
                "model_type": model_type,
            }
            # Model-özel ekstra feature'lar (dqn_features/hybrid_features/ensemble_features)
            for key in ("dqn_features", "hybrid_features", "ensemble_features"):
                if key in prediction_result:
                    features_used[key] = prediction_result[key]

            current_price = prediction_result.get("current_price", 0)
            with get_tenant_session(schema) as s:
                ar = AnalysisResult(
                    coin_symbol=coin_symbol.upper(),
                    predicted_price=prediction_result.get("predicted_price", 0),
                    current_price=current_price,
                    price_change_percent=prediction_result.get("price_change_percent", 0),
                    confidence=prediction_result.get("confidence", 0),
                    news_sentiment=(news_analysis.get("news_sentiment", 0) if news_analysis else None),
                    whale_activity_score=(
                        whale_analysis.get("whale_activity_score", 0) if whale_analysis else None
                    ),
                    yigit_position=(yigit_analysis.get("current_position", 0) if yigit_analysis else None),
                    yigit_signal=(yigit_analysis.get("current_signal", None) if yigit_analysis else None),
                    trend_strength=(yigit_analysis.get("trend_strength", 0) if yigit_analysis else None),
                    model_type=model_type,
                    features_used=features_used,
                )
                s.add(ar)
                s.flush()
                analysis_id = ar.id
                # Tenant izleme sayacını güncelle
                w = s.execute(
                    select(Watchlist).where(Watchlist.coin_symbol == coin_symbol.upper())
                ).scalar_one_or_none()
                if w:
                    w.last_analysis = datetime.now(timezone.utc)
                    w.analysis_count = (w.analysis_count or 0) + 1

            # Ortak katalogdaki güncel fiyatı güncelle
            try:
                with get_session() as cs:
                    coin = cs.execute(
                        select(Coin).where(Coin.symbol == coin_symbol.upper())
                    ).scalar_one_or_none()
                    if coin and current_price:
                        coin.current_price = current_price
            except Exception:
                pass

            print(f"📊 {model_type} analiz sonucu kaydedildi: {coin_symbol}")
            return analysis_id
        except Exception as e:
            print(f"❌ Analiz kaydetme hatası: {str(e)}")
            return None

    def save_dqn_analysis(
        self, coin_symbol: str, dqn_prediction: Dict, news_analysis: Dict = None, whale_analysis: Dict = None
    ) -> Optional[int]:
        try:
            action = dqn_prediction.get("action", 0)
            current_price = dqn_prediction.get("current_price", 0)
            if action == 0:
                price_change_percent, predicted_price = 0, current_price
            elif action in (1, 2, 3, 4):
                strength = action * 0.25
                price_change_percent = strength * 2
                predicted_price = current_price * (1 + price_change_percent / 100)
            else:
                strength = (action - 4) * 0.25
                price_change_percent = -strength * 2
                predicted_price = current_price * (1 + price_change_percent / 100)

            features_used = {
                "dqn_action": dqn_prediction.get("action_name", "UNKNOWN"),
                "q_values": dqn_prediction.get("q_values", []),
                "exploration_rate": dqn_prediction.get("epsilon", 0),
                "model_confidence": dqn_prediction.get("confidence", 0),
                "reasoning": dqn_prediction.get("reasoning", ""),
            }
            enhanced = {
                "current_price": current_price,
                "confidence": dqn_prediction.get("confidence", 0),
                "predicted_price": predicted_price,
                "price_change_percent": price_change_percent,
                "dqn_features": features_used,
            }
            return self.save_analysis_result(
                coin_symbol, enhanced, news_analysis, whale_analysis, model_type="DQN"
            )
        except Exception as e:
            print(f"❌ DQN analiz kaydetme hatası: {str(e)}")
            return None

    def save_hybrid_analysis(
        self, coin_symbol: str, hybrid_prediction: Dict, news_analysis: Dict = None, whale_analysis: Dict = None
    ) -> Optional[int]:
        try:
            ensemble_pred = hybrid_prediction.get("ensemble_prediction", {})
            lstm_pred = hybrid_prediction.get("lstm_prediction", {})
            dqn_pred = hybrid_prediction.get("dqn_prediction", {})
            features_used = {
                "ensemble_signal": ensemble_pred.get("ensemble_signal", 0),
                "recommendation": ensemble_pred.get("recommendation", "HOLD"),
                "model_weights": ensemble_pred.get("model_weights", {}),
                "component_signals": ensemble_pred.get("component_signals", {}),
                "lstm_confidence": lstm_pred.get("confidence", 0),
                "dqn_confidence": dqn_pred.get("confidence", 0),
                "dqn_action": dqn_pred.get("action_name", "UNKNOWN"),
                "reasoning": ensemble_pred.get("reasoning", ""),
            }
            enhanced = {
                "current_price": lstm_pred.get("current_price", 0) or dqn_pred.get("current_price", 0),
                "predicted_price": lstm_pred.get("predicted_price", 0),
                "price_change_percent": lstm_pred.get("price_change_pct", 0),
                "confidence": hybrid_prediction.get("confidence", 0),
                "hybrid_features": features_used,
            }
            return self.save_analysis_result(
                coin_symbol, enhanced, news_analysis, whale_analysis, model_type="HYBRID"
            )
        except Exception as e:
            print(f"❌ Hybrid analiz kaydetme hatası: {str(e)}")
            return None

    def save_ensemble_analysis(
        self,
        coin_symbol: str,
        ensemble_results: List[Dict],
        weights: Dict,
        news_analysis: Dict = None,
        whale_analysis: Dict = None,
    ) -> Optional[int]:
        try:
            weighted_price_change = 0
            weighted_confidence = 0
            current_price = 0
            model_predictions = {}
            for model_result in ensemble_results:
                model_type = model_result.get("model_type", "UNKNOWN")
                weight = weights.get(model_type.lower(), 0)
                if weight > 0:
                    price_change = model_result.get("price_change_percent", 0)
                    confidence = model_result.get("confidence", 0)
                    weighted_price_change += price_change * weight
                    weighted_confidence += confidence * weight
                    if not current_price:
                        current_price = model_result.get("current_price", 0)
                    model_predictions[model_type] = {
                        "prediction": price_change,
                        "confidence": confidence,
                        "weight": weight,
                    }
            predicted_price = current_price * (1 + weighted_price_change / 100) if current_price else 0
            features_used = {
                "model_weights": weights,
                "model_predictions": model_predictions,
                "ensemble_method": "weighted_average",
                "models_used": list(model_predictions.keys()),
                "individual_results": ensemble_results,
            }
            enhanced = {
                "current_price": current_price,
                "predicted_price": predicted_price,
                "price_change_percent": weighted_price_change,
                "confidence": weighted_confidence,
                "ensemble_features": features_used,
            }
            return self.save_analysis_result(
                coin_symbol, enhanced, news_analysis, whale_analysis, model_type="ENSEMBLE"
            )
        except Exception as e:
            print(f"❌ Ensemble analiz kaydetme hatası: {str(e)}")
            return None

    def get_analysis_history(
        self, coin_symbol: str = None, model_type: str = None, days: int = 7, limit: int = 100
    ) -> List[Dict]:
        schema = self._tenant()
        if not schema:
            return []
        try:
            with get_tenant_session(schema) as s:
                stmt = select(
                    AnalysisResult.id,
                    AnalysisResult.coin_symbol,
                    AnalysisResult.predicted_price,
                    AnalysisResult.current_price,
                    AnalysisResult.price_change_percent,
                    AnalysisResult.confidence,
                    AnalysisResult.news_sentiment,
                    AnalysisResult.whale_activity_score,
                    AnalysisResult.yigit_position,
                    AnalysisResult.yigit_signal,
                    AnalysisResult.trend_strength,
                    AnalysisResult.analysis_timestamp,
                    AnalysisResult.model_type,
                    AnalysisResult.features_used,
                )
                if coin_symbol:
                    stmt = stmt.where(AnalysisResult.coin_symbol == coin_symbol.upper())
                if model_type:
                    stmt = stmt.where(AnalysisResult.model_type == model_type.upper())
                if days:
                    since = datetime.now(timezone.utc) - timedelta(days=days)
                    stmt = stmt.where(AnalysisResult.analysis_timestamp >= since)
                stmt = stmt.order_by(AnalysisResult.analysis_timestamp.desc()).limit(limit)
                rows = s.execute(stmt).all()
                return [
                    {
                        "id": r[0],
                        "coin_symbol": r[1],
                        "predicted_price": _f(r[2]),
                        "current_price": _f(r[3]),
                        "price_change_percent": _f(r[4]),
                        "confidence": _f(r[5]),
                        "news_sentiment": _f(r[6]),
                        "whale_activity_score": _f(r[7]),
                        "yigit_position": r[8],
                        "yigit_signal": r[9],
                        "trend_strength": _f(r[10]),
                        "analysis_timestamp": _dt(r[11]),
                        "model_type": r[12],
                        "features_used": r[13] or {},
                    }
                    for r in rows
                ]
        except Exception as e:
            print(f"❌ Analiz geçmişi alma hatası: {str(e)}")
            return []

    def get_model_performance_comparison(self, coin_symbol: str = None, days: int = 7) -> Dict:
        try:
            analyses = self.get_analysis_history(coin_symbol, days=days, limit=1000)
            model_stats: Dict[str, Any] = {}
            for analysis in analyses:
                mt = analysis["model_type"]
                st = model_stats.setdefault(
                    mt,
                    {
                        "total_predictions": 0,
                        "avg_confidence": 0,
                        "confidence_sum": 0,
                        "price_changes": [],
                        "confidences": [],
                        "timestamps": [],
                    },
                )
                st["total_predictions"] += 1
                st["confidence_sum"] += analysis["confidence"] or 0
                st["price_changes"].append(analysis["price_change_percent"] or 0)
                st["confidences"].append(analysis["confidence"] or 0)
                st["timestamps"].append(analysis["analysis_timestamp"])

            for mt, st in model_stats.items():
                if st["total_predictions"] > 0:
                    st["avg_confidence"] = st["confidence_sum"] / st["total_predictions"]
                    st["avg_price_change"] = sum(st["price_changes"]) / len(st["price_changes"])
                    try:
                        import pandas as pd

                        st["confidence_std"] = (
                            pd.Series(st["confidences"]).std() if len(st["confidences"]) > 1 else 0
                        )
                        st["price_change_std"] = (
                            pd.Series(st["price_changes"]).std() if len(st["price_changes"]) > 1 else 0
                        )
                    except Exception:
                        st["confidence_std"] = 0
                        st["price_change_std"] = 0
                    recent = []
                    for t in st["timestamps"]:
                        try:
                            ts = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
                            if ts.replace(tzinfo=None) > datetime.now() - timedelta(hours=24):
                                recent.append(t)
                        except Exception:
                            pass
                    st["recent_predictions_24h"] = len(recent)
                for k in ("confidence_sum", "price_changes", "confidences", "timestamps"):
                    st.pop(k, None)

            return {
                "comparison_period_days": days,
                "coin_symbol": coin_symbol or "ALL",
                "model_performance": model_stats,
                "total_analyses": len(analyses),
                "analysis_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"❌ Model performans karşılaştırması hatası: {str(e)}")
            return {}

    # ================================================================== #
    # PORTFOLIO (tenant)
    # ================================================================== #
    def get_portfolio_summary(self) -> Dict:
        schema = self._tenant()
        if not schema:
            return {}
        try:
            with get_tenant_session(schema) as s:
                pos = s.execute(
                    select(
                        func.count(Position.id),
                        func.sum(Position.entry_value),
                        func.sum(Position.current_value),
                        func.sum(Position.unrealized_pnl),
                    ).where(Position.is_open.is_(True))
                ).one()
                active_positions = pos[0] or 0
                invested_amount = float(pos[1] or 0)
                current_value = float(pos[2] or 0)
                unrealized_pnl = float(pos[3] or 0)

                total_trades = (
                    s.execute(
                        select(func.count(Trade.id)).where(
                            Trade.trade_type.in_(["CLOSE_LONG", "CLOSE_SHORT", "SELL"])
                        )
                    ).scalar()
                    or 0
                )
                successful_trades = (
                    s.execute(
                        select(func.count(Position.id)).where(
                            Position.is_open.is_(False), Position.unrealized_pnl > 0
                        )
                    ).scalar()
                    or 0
                )
            win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl_percent = (unrealized_pnl / invested_amount * 100) if invested_amount > 0 else 0
            return {
                "total_balance": current_value,
                "invested_amount": invested_amount,
                "current_value": current_value,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl_percent": total_pnl_percent,
                "active_positions": active_positions,
                "successful_trades": successful_trades,
                "total_trades": total_trades,
                "win_rate": win_rate,
            }
        except Exception as e:
            print(f"❌ Portfolio özeti hatası: {str(e)}")
            return {}

    def get_coin_performance(self, coin_symbol: str, days: int = 30) -> Dict:
        schema = self._tenant()
        if not schema:
            return {}
        try:
            since = datetime.now(timezone.utc) - timedelta(days=days)
            sym = coin_symbol.upper()
            with get_tenant_session(schema) as s:
                analysis_count = (
                    s.execute(
                        select(func.count(AnalysisResult.id)).where(
                            AnalysisResult.coin_symbol == sym,
                            AnalysisResult.analysis_timestamp > since,
                        )
                    ).scalar()
                    or 0
                )
                trade_count = (
                    s.execute(
                        select(func.count(Trade.id)).where(
                            Trade.coin_symbol == sym, Trade.timestamp > since
                        )
                    ).scalar()
                    or 0
                )
                total_pnl = (
                    s.execute(
                        select(func.sum(Position.unrealized_pnl)).where(
                            Position.coin_symbol == sym, Position.entry_timestamp > since
                        )
                    ).scalar()
                    or 0
                )
                avg_confidence = (
                    s.execute(
                        select(func.avg(AnalysisResult.confidence)).where(
                            AnalysisResult.coin_symbol == sym,
                            AnalysisResult.analysis_timestamp > since,
                        )
                    ).scalar()
                    or 0
                )
            return {
                "coin_symbol": coin_symbol,
                "analysis_count": analysis_count,
                "trade_count": trade_count,
                "total_pnl": float(total_pnl),
                "avg_confidence": float(avg_confidence),
                "days": days,
            }
        except Exception as e:
            print(f"❌ Coin performans hatası: {str(e)}")
            return {}

    def get_all_coin_performances(self, days: int = 30) -> List[Dict]:
        return [self.get_coin_performance(c["symbol"], days) for c in self.get_active_coins()]

    # ================================================================== #
    # SYSTEM STATE (tenant)
    # ================================================================== #
    def save_system_state(self, state_key: str, state_value: Any):
        schema = self._tenant()
        if not schema:
            return
        try:
            with get_tenant_session(schema) as s:
                st = s.execute(
                    select(SystemState).where(SystemState.state_key == state_key)
                ).scalar_one_or_none()
                if st:
                    st.state_value = state_value
                    st.last_updated = datetime.now(timezone.utc)
                else:
                    s.add(SystemState(state_key=state_key, state_value=state_value))
            print(f"💾 System state kaydedildi: {state_key}")
        except Exception as e:
            print(f"❌ System state kaydetme hatası: {str(e)}")

    def load_system_state(self, state_key: str, default_value: Any = None) -> Any:
        schema = self._tenant()
        if not schema:
            return default_value
        try:
            with get_tenant_session(schema) as s:
                val = s.execute(
                    select(SystemState.state_value).where(SystemState.state_key == state_key)
                ).scalar_one_or_none()
                return val if val is not None else default_value
        except Exception as e:
            print(f"❌ System state yükleme hatası: {str(e)}")
            return default_value

    # ================================================================== #
    # PREDICTION CACHE (shared / ortak — eğitim tenant'lar arası paylaşılır)
    # ================================================================== #
    def save_prediction_cache(
        self,
        coin_symbol: str,
        model_type: str,
        prediction_data: Dict,
        technical_analysis: Dict = None,
        news_analysis: Dict = None,
        whale_analysis: Dict = None,
        yigit_analysis: Dict = None,
        trade_signal: Dict = None,
        ttl_hours: int = 4,
    ) -> bool:
        try:
            with get_session() as s:
                # Eski kayıtları geçersiz kıl
                for old in s.execute(
                    select(PredictionCache).where(
                        PredictionCache.coin_symbol == coin_symbol.upper(),
                        PredictionCache.model_type == model_type,
                        PredictionCache.is_valid.is_(True),
                    )
                ).scalars():
                    old.is_valid = False
                s.add(
                    PredictionCache(
                        coin_symbol=coin_symbol.upper(),
                        model_type=model_type,
                        prediction_data=prediction_data,
                        technical_analysis=technical_analysis,
                        news_analysis=news_analysis,
                        whale_analysis=whale_analysis,
                        yigit_analysis=yigit_analysis,
                        trade_signal=trade_signal,
                        expires_at=datetime.now(timezone.utc) + timedelta(hours=ttl_hours),
                    )
                )
            return True
        except Exception as e:
            print(f"❌ Prediction cache kaydetme hatası: {str(e)}")
            return False

    def get_prediction_cache(self, coin_symbol: str) -> Optional[Dict]:
        try:
            with get_session() as s:
                row = s.execute(
                    select(PredictionCache)
                    .where(
                        PredictionCache.coin_symbol == coin_symbol.upper(),
                        PredictionCache.is_valid.is_(True),
                        PredictionCache.expires_at > datetime.now(timezone.utc),
                    )
                    .order_by(PredictionCache.cache_timestamp.desc())
                    .limit(1)
                ).scalar_one_or_none()
                if not row:
                    return None
                return {
                    "model_type": row.model_type,
                    "prediction_data": row.prediction_data,
                    "technical_analysis": row.technical_analysis,
                    "news_analysis": row.news_analysis,
                    "whale_analysis": row.whale_analysis,
                    "yigit_analysis": row.yigit_analysis,
                    "trade_signal": row.trade_signal,
                    "cache_timestamp": _dt(row.cache_timestamp),
                    "expires_at": _dt(row.expires_at),
                }
        except Exception as e:
            print(f"❌ Prediction cache okuma hatası: {str(e)}")
            return None

    def cleanup_expired_cache(self) -> int:
        try:
            with get_session() as s:
                now = datetime.now(timezone.utc)
                invalidated = 0
                for row in s.execute(
                    select(PredictionCache).where(
                        PredictionCache.expires_at <= now, PredictionCache.is_valid.is_(True)
                    )
                ).scalars():
                    row.is_valid = False
                    invalidated += 1
                # 7 günden eski kayıtları sil
                old = s.execute(
                    select(PredictionCache).where(
                        PredictionCache.cache_timestamp < now - timedelta(days=7)
                    )
                ).scalars().all()
                for row in old:
                    s.delete(row)
                if invalidated:
                    print(f"🧹 {invalidated} adet süresi dolmuş cache temizlendi")
                return invalidated + len(old)
        except Exception as e:
            print(f"❌ Cache temizleme hatası: {str(e)}")
            return 0

    # ================================================================== #
    # Uyumluluk / yardımcı
    # ================================================================== #
    def test_connection(self) -> bool:
        try:
            with get_session() as s:
                s.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """Ham SQL kaçış-deliği (deprecated). Yalnız kalan birkaç inline sorgu için.

        Uyarı: PostgreSQL syntax'ı gerektirir; parametreler :p0, :p1 ... adlandırması.
        Yeni kodda kullanmayın; repository metodlarını tercih edin.
        """
        try:
            with get_session() as s:
                bind = {}
                if params:
                    q = query
                    for i, p in enumerate(params):
                        bind[f"p{i}"] = p
                    res = s.execute(text(q), bind)
                else:
                    res = s.execute(text(query))
                if fetch:
                    return [dict(r._mapping) for r in res]
                return res.rowcount
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
            raise
