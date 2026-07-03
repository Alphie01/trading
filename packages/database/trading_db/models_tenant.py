"""Tenant-özel şema modelleri.

Her tenant kendi PostgreSQL şemasında (``tenant_<slug>``) bu tabloların
birer kopyasına sahiptir. Sembolik ``tenant`` token'ı çalışma anında
``schema_translate_map`` ile gerçek şemaya çevrilir.

Not: ``coin_symbol`` kolonları shared ``coins`` kataloğuna MANTIKSAL referanstır
(orijinal SQLite'ta da FK enforcement kapalıydı) — cross-schema DB FK'sı kurulmaz.
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

from .base import TenantBase


class Watchlist(TenantBase):
    """Tenant'ın izleme listesi (coins kataloğuna per-tenant durum katmanı)."""

    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_symbol = Column(String(20), unique=True, nullable=False)
    added_date = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, nullable=False, server_default="true")
    last_analysis = Column(DateTime(timezone=True))
    analysis_count = Column(Integer, nullable=False, server_default="0")


class Trade(TenantBase):
    """İşlem geçmişi (append-only ledger)."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_symbol = Column(String(20), nullable=False)
    trade_type = Column(String(20), nullable=False)  # BUY/SELL/LONG/SHORT/CLOSE_*
    price = Column(Numeric(18, 8), nullable=False)
    quantity = Column(Numeric(18, 8), nullable=False)
    total_value = Column(Numeric(18, 8), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    confidence = Column(Numeric(10, 4))
    news_sentiment = Column(Numeric(10, 4))
    whale_activity = Column(Numeric(10, 4))
    yigit_signal = Column(String(50))
    trade_reason = Column(Text)
    exchange = Column(String(50), server_default="binance")
    fees = Column(Numeric(18, 8), server_default="0")
    is_simulated = Column(Boolean, nullable=False, server_default="true")

    __table_args__ = (Index("ix_trades_coin_symbol", "coin_symbol"),)


class Position(TenantBase):
    """Pozisyon takibi (kapatma = soft, is_open=0)."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_symbol = Column(String(20), nullable=False)
    position_type = Column(String(20), nullable=False)  # SPOT/LONG/SHORT
    entry_price = Column(Numeric(18, 8), nullable=False)
    current_price = Column(Numeric(18, 8))
    quantity = Column(Numeric(18, 8), nullable=False)
    entry_value = Column(Numeric(18, 8), nullable=False)
    current_value = Column(Numeric(18, 8))
    unrealized_pnl = Column(Numeric(18, 8))
    entry_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    last_update = Column(DateTime(timezone=True), server_default=func.now())
    is_open = Column(Boolean, nullable=False, server_default="true")
    stop_loss = Column(Numeric(18, 8))
    take_profit = Column(Numeric(18, 8))
    leverage = Column(Numeric(10, 4), server_default="1")

    __table_args__ = (Index("ix_positions_coin_symbol", "coin_symbol"),)


class AnalysisResult(TenantBase):
    """Tenant'ın analiz sonuçları."""

    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_symbol = Column(String(20), nullable=False)
    predicted_price = Column(Numeric(18, 8), nullable=False)
    current_price = Column(Numeric(18, 8), nullable=False)
    price_change_percent = Column(Numeric(18, 8), nullable=False)
    confidence = Column(Numeric(10, 4), nullable=False)
    news_sentiment = Column(Numeric(10, 4))
    whale_activity_score = Column(Numeric(10, 4))
    yigit_position = Column(Integer)
    yigit_signal = Column(String(50))
    trend_strength = Column(Numeric(10, 4))
    analysis_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    model_type = Column(String(50), server_default="LSTM")
    features_used = Column(JSONB)

    __table_args__ = (
        Index("ix_analysis_results_symbol_time", "coin_symbol", "analysis_timestamp"),
    )


class PortfolioSummary(TenantBase):
    """Portfolio snapshot (tarihsel özet)."""

    __tablename__ = "portfolio_summary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    total_balance = Column(Numeric(18, 8), nullable=False)
    invested_amount = Column(Numeric(18, 8), nullable=False)
    current_value = Column(Numeric(18, 8), nullable=False)
    total_pnl = Column(Numeric(18, 8), nullable=False)
    total_pnl_percent = Column(Numeric(18, 8), nullable=False)
    active_positions = Column(Integer, nullable=False)
    successful_trades = Column(Integer, nullable=False)
    total_trades = Column(Integer, nullable=False)
    win_rate = Column(Numeric(10, 4), nullable=False)
    summary_date = Column(DateTime(timezone=True), server_default=func.now())


class SystemState(TenantBase):
    """Key-value persistence (tenant başına sistem durumu)."""

    __tablename__ = "system_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    state_key = Column(String(150), unique=True, nullable=False)
    state_value = Column(JSONB)
    last_updated = Column(DateTime(timezone=True), server_default=func.now())


# ============================================================================ #
# Otomasyon motoru — TENANT (kullanıcıya özel sinyaller/uyarılar/kararlar)
# ============================================================================ #
class AutomationSignal(TenantBase):
    """Otomasyon motorunun ürettiği ticaret sinyalleri (tenant'a özel)."""

    __tablename__ = "automation_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(30), nullable=False)
    signal = Column(String(20), nullable=False)  # BUY / HOLD / SELL
    recommendation = Column(String(20))  # STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
    opportunity_score = Column(Numeric(6, 2))
    risk_score = Column(Numeric(6, 2))
    confidence = Column(Numeric(5, 4))
    reasons = Column(JSONB)
    warnings = Column(JSONB)
    source = Column(String(30), server_default="automation")
    status = Column(String(20), nullable=False, server_default="new")  # new/acknowledged/expired
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_automation_signals_symbol_time", "symbol", "created_at"),)


class AutomationAlert(TenantBase):
    """Dashboard/log uyarıları (tenant'a özel)."""

    __tablename__ = "automation_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(30))
    level = Column(String(20), nullable=False, server_default="info")  # info/warning/critical
    title = Column(String(200))
    message = Column(Text)
    is_read = Column(Boolean, nullable=False, server_default="false")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_automation_alerts_time", "created_at"),)


class TradeDecision(TenantBase):
    """Otomasyon trade kararları / audit log (Faz 7'de risk manager tarafından yazılır)."""

    __tablename__ = "trade_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(30), nullable=False)
    action = Column(String(20), nullable=False)  # long/short/close/skip
    decided = Column(Boolean, nullable=False, server_default="false")  # gerçekten işlem açıldı mı
    reason = Column(Text)
    confidence = Column(Numeric(5, 4))
    risk_score = Column(Numeric(6, 2))
    is_testnet = Column(Boolean, server_default="true")
    is_simulated = Column(Boolean, server_default="true")
    details = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_trade_decisions_symbol_time", "symbol", "created_at"),)


# ============================================================================ #
# Simulation / Paper Trading (TENANT) — SANAL; canlı Binance işlemiyle İLGİSİ YOK
# ============================================================================ #
class SimulationRun(TenantBase):
    """Bir simülasyon çalışması (kendi bakiyesi/config'i/portföyü)."""

    __tablename__ = "simulation_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False)
    status = Column(String(20), nullable=False, server_default="RUNNING")  # RUNNING/PAUSED/STOPPED/COMPLETED
    mode = Column(String(20), nullable=False, server_default="SPOT")  # SPOT/FUTURES
    quote_currency = Column(String(20), server_default="USDT")
    initial_balance = Column(Numeric(24, 8), nullable=False)
    current_balance = Column(Numeric(24, 8), nullable=False)
    equity = Column(Numeric(24, 8))
    config = Column(JSONB)  # risk/leverage/symbols/sl-tp/fee_profile/slippage/reentry/...
    created_by = Column(String(64))
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    stopped_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("ix_simulation_runs_status", "status"),)


class SimulationPosition(TenantBase):
    """Simülasyon pozisyonu (açık/kapalı)."""

    __tablename__ = "simulation_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, nullable=False)
    symbol = Column(String(30), nullable=False)
    mode = Column(String(20), nullable=False)  # SPOT/FUTURES
    side = Column(String(10), nullable=False)  # LONG/SHORT/SPOT
    status = Column(String(20), nullable=False, server_default="OPEN")  # OPEN/CLOSED/LIQUIDATED/CANCELLED
    entry_price = Column(Numeric(24, 8), nullable=False)
    current_price = Column(Numeric(24, 8))
    exit_price = Column(Numeric(24, 8))
    quantity = Column(Numeric(28, 12), nullable=False)
    notional_value = Column(Numeric(24, 8))
    margin_used = Column(Numeric(24, 8))
    leverage = Column(Numeric(10, 4))
    entry_fee = Column(Numeric(24, 8), server_default="0")
    exit_fee = Column(Numeric(24, 8), server_default="0")
    unrealized_pnl = Column(Numeric(24, 8), server_default="0")
    realized_pnl = Column(Numeric(24, 8), server_default="0")
    net_pnl = Column(Numeric(24, 8), server_default="0")
    roi_percent = Column(Numeric(12, 4))
    liquidation_price = Column(Numeric(24, 8))
    opened_at = Column(DateTime(timezone=True), server_default=func.now())
    closed_at = Column(DateTime(timezone=True))
    entry_signal_id = Column(String(64))
    exit_signal_id = Column(String(64))
    entry_reason = Column(Text)
    exit_reason = Column(Text)
    position_metadata = Column("metadata", JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_simulation_positions_sim_status", "simulation_id", "status"),
        Index("ix_simulation_positions_sim_symbol", "simulation_id", "symbol"),
    )


class SimulationTrade(TenantBase):
    """Simülasyon fill kaydı (entry/exit emirleri)."""

    __tablename__ = "simulation_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, nullable=False)
    position_id = Column(Integer)
    symbol = Column(String(30), nullable=False)
    side = Column(String(20), nullable=False)  # BUY/SELL/OPEN_LONG/CLOSE_LONG/OPEN_SHORT/CLOSE_SHORT
    order_type = Column(String(10), server_default="taker")
    price = Column(Numeric(24, 8), nullable=False)
    quantity = Column(Numeric(28, 12), nullable=False)
    notional_value = Column(Numeric(24, 8))
    fee = Column(Numeric(24, 8), server_default="0")
    fee_asset = Column(String(20), server_default="USDT")
    slippage = Column(Numeric(12, 8))
    signal_id = Column(String(64))
    reason = Column(Text)
    executed_at = Column(DateTime(timezone=True), server_default=func.now())
    trade_metadata = Column("metadata", JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_simulation_trades_sim_time", "simulation_id", "executed_at"),)


class SimulationReport(TenantBase):
    """Periyot (günlük/haftalık/aylık) performans snapshot'ı."""

    __tablename__ = "simulation_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, nullable=False)
    period_type = Column(String(10), nullable=False)  # DAILY/WEEKLY/MONTHLY
    period_start = Column(DateTime(timezone=True))
    period_end = Column(DateTime(timezone=True))
    starting_balance = Column(Numeric(24, 8))
    ending_balance = Column(Numeric(24, 8))
    net_profit = Column(Numeric(24, 8))
    gross_profit = Column(Numeric(24, 8))
    gross_loss = Column(Numeric(24, 8))
    total_fees = Column(Numeric(24, 8))
    roi_percent = Column(Numeric(12, 4))
    win_rate = Column(Numeric(6, 2))
    loss_rate = Column(Numeric(6, 2))
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    max_drawdown = Column(Numeric(12, 4))
    profit_factor = Column(Numeric(12, 4))
    best_trade = Column(Numeric(24, 8))
    worst_trade = Column(Numeric(24, 8))
    avg_win = Column(Numeric(24, 8))
    avg_loss = Column(Numeric(24, 8))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_simulation_reports_sim_period", "simulation_id", "period_type"),)
