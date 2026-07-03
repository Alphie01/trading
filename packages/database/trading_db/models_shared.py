"""Shared (ortak) şema modelleri.

Tenant'lar arası ORTAK veriler: tenant kayıt defteri, global kullanıcılar,
coin kataloğu, model tahmin cache'i ve model/eğitim metadata'sı.
"eğitimler ortak" ilkesi: coins/prediction_cache/model_registry burada durur.
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from .base import SharedBase


class Tenant(SharedBase):
    """Tenant kayıt defteri (her tenant bir PostgreSQL şemasına karşılık gelir)."""

    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    slug = Column(String(63), unique=True, nullable=False)
    schema_name = Column(String(63), unique=True, nullable=False)
    name = Column(String(200))
    # provisioning | active | failed | suspended
    status = Column(String(20), nullable=False, server_default="provisioning")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    users = relationship("User", back_populates="tenant")


class User(SharedBase):
    """Global kullanıcı. Her kullanıcı bir tenant'a bağlıdır (platform admin hariç)."""

    __tablename__ = "users"

    # Mevcut auth.py secrets.token_urlsafe(16) ürettiği için String PK korunur (uuid'ye çevrilmez).
    id = Column(String(64), primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(200))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
    is_active = Column(Boolean, nullable=False, server_default="true")
    role = Column(String(20), nullable=False, server_default="user")
    # NULL => platform/system admin (herhangi bir tenant'a bağlı değil)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="SET NULL"))

    tenant = relationship("Tenant", back_populates="users")


class Coin(SharedBase):
    """Coin KATALOĞU (ortak). Tenant'a özel izleme durumu tenant.watchlist'tedir."""

    __tablename__ = "coins"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False)
    name = Column(String(100))
    added_date = Column(DateTime(timezone=True), server_default=func.now())
    current_price = Column(Numeric(18, 8))
    price_change_24h = Column(Numeric(18, 8))


class PredictionCache(SharedBase):
    """Model tahmin cache'i (ortak) — eğitim/tahmin tenant'lar arası paylaşılır."""

    __tablename__ = "prediction_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_symbol = Column(String(20), nullable=False)
    model_type = Column(String(50), nullable=False)
    prediction_data = Column(JSONB, nullable=False)
    technical_analysis = Column(JSONB)
    news_analysis = Column(JSONB)
    whale_analysis = Column(JSONB)
    yigit_analysis = Column(JSONB)
    trade_signal = Column(JSONB)
    cache_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
    is_valid = Column(Boolean, nullable=False, server_default="true")

    __table_args__ = (
        Index("ix_prediction_cache_symbol_time", "coin_symbol", "cache_timestamp"),
        Index("ix_prediction_cache_expires", "expires_at", "is_valid"),
    )


class ModelRegistry(SharedBase):
    """Eğitilmiş model metadata'sı (ortak). model_cache/ dosyalarına referans."""

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    coin_symbol = Column(String(20), nullable=False)
    model_type = Column(String(50), nullable=False)  # lstm | dqn | hybrid
    model_id = Column(String(120))  # ör. BTC_USDT_a1b2c3d4
    file_path = Column(Text)
    config = Column(JSONB)
    metrics = Column(JSONB)
    feature_count = Column(Integer)
    data_hash = Column(String(64))
    version = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_trained = Column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_model_registry_symbol_type", "coin_symbol", "model_type"),
    )


# ============================================================================ #
# Otomasyon motoru — SHARED (evren-geneli market zekâsı; tenant'lardan bağımsız)
# ============================================================================ #
class DiscoveryCandidate(SharedBase):
    """Keşfedilen coin adayları (ccxt ticker taramasından)."""

    __tablename__ = "discovery_candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(30), unique=True, nullable=False)  # ör. SOL/USDT
    base_asset = Column(String(20))
    quote_asset = Column(String(20))
    exchange = Column(String(30), server_default="binance")
    volume_24h = Column(Numeric(24, 4))
    price_change_24h = Column(Numeric(18, 8))
    volume_change_score = Column(Numeric(10, 4))
    volatility_score = Column(Numeric(10, 4))
    liquidity_score = Column(Numeric(10, 4))
    is_new_listing_candidate = Column(Boolean, server_default="false")
    discovery_reason = Column(Text)
    # discovered | research_queue | watchlist | hot_candidate | trade_candidate | rejected | cooldown
    status = Column(String(30), nullable=False, server_default="discovered")
    discovered_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("ix_discovery_candidates_status", "status"),)


class CoinScore(SharedBase):
    """Coin fırsat/risk skorları (evren-geneli; tüm tenant'lara ortak)."""

    __tablename__ = "coin_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(30), nullable=False)
    opportunity_score = Column(Numeric(6, 2))  # 0-100
    risk_score = Column(Numeric(6, 2))  # 0-100
    confidence = Column(Numeric(5, 4))  # 0-1
    technical_score = Column(Numeric(6, 2))
    volume_liquidity_score = Column(Numeric(6, 2))
    ai_prediction_score = Column(Numeric(6, 2))
    sentiment_score = Column(Numeric(6, 2))
    whale_score = Column(Numeric(6, 2))
    recommendation = Column(String(20))  # STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
    reasons = Column(JSONB)
    warnings = Column(JSONB)
    scored_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_coin_scores_symbol_time", "symbol", "scored_at"),)


class AutomationRun(SharedBase):
    """Otomasyon çalışma kayıtları (discovery/scan/research turları)."""

    __tablename__ = "automation_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_type = Column(String(30), nullable=False)  # discovery | scan | research
    status = Column(String(20), nullable=False, server_default="running")  # running|success|failed
    scanned_count = Column(Integer, server_default="0")
    passed_count = Column(Integer, server_default="0")
    watchlisted_count = Column(Integer, server_default="0")
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    finished_at = Column(DateTime(timezone=True))
    details = Column(JSONB)
