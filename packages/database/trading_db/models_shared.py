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
    # Market Intelligence bileşenleri (haber kalitesi / sosyal momentum / hype riski)
    news_quality_score = Column(Numeric(6, 2))
    social_momentum_score = Column(Numeric(6, 2))
    hype_risk = Column(Numeric(6, 2))
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


# ============================================================================ #
# Market Intelligence — SHARED (haber/sosyal/LLM zekâsı; evren-geneli, tenant'lardan bağımsız)
# ============================================================================ #
class IntelligenceSource(SharedBase):
    """Haber/sosyal kaynak kayıt defteri (katmanlı: tier + reputation + impact_weight)."""

    __tablename__ = "intelligence_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), unique=True, nullable=False)
    # exchange_official | regulator | crypto_news | security | social | regional | event
    category = Column(String(50))
    tier = Column(Integer)  # 1 (en güvenilir) .. 4 (sosyal/blog)
    reputation_score = Column(Numeric(6, 2))  # 0-100
    impact_weight = Column(Numeric(5, 3))  # 0-1
    allow_trade_signal_boost = Column(Boolean, server_default="true")
    regional = Column(Boolean, server_default="false")
    enabled = Column(Boolean, server_default="true")
    last_collected_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class NewsItem(SharedBase):
    """Toplanan (deduplike) haber öğesi. Prod'da sahte veri YOK (data_source ile etiketli)."""

    __tablename__ = "news_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(120))
    source_tier = Column(Integer)
    symbol = Column(String(30))  # ilişkili sembol (varsa; global haber için None)
    title = Column(Text, nullable=False)
    url = Column(Text)
    published_at = Column(DateTime(timezone=True))
    content_hash = Column(String(64), unique=True, nullable=False)
    duplicate_group_id = Column(String(64))
    raw_sentiment = Column(Numeric(6, 3))  # FinBERT/ensemble baseline (-1..1)
    data_source = Column(String(40))  # 'rss' | 'api' | 'reddit' ... (demo_mock yalnız DEMO_MODE)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_news_items_symbol_time", "symbol", "published_at"),)


class NewsAnalysisResult(SharedBase):
    """Bir haber öğesinin LLM (Ollama) + kural tabanlı analiz sonucu."""

    __tablename__ = "news_analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    news_item_id = Column(Integer, ForeignKey("news_items.id"))
    symbol = Column(String(30))
    sentiment = Column(String(20))  # positive | neutral | negative
    sentiment_score = Column(Numeric(6, 2))  # 0-100
    quality_score = Column(Numeric(6, 2))
    relevance_score = Column(Numeric(6, 2))
    impact_score = Column(Numeric(6, 2))
    freshness_score = Column(Numeric(6, 2))
    hype_score = Column(Numeric(6, 2))
    risk_score = Column(Numeric(6, 2))
    event_type = Column(String(40))  # listing|delisting|regulatory|hack|partnership|token_unlock|upgrade|general|unknown
    ollama_model = Column(String(80))
    ollama_confidence = Column(Numeric(5, 4))
    analysis_json = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_news_analysis_symbol_time", "symbol", "created_at"),)


class MarketIntelligenceSnapshot(SharedBase):
    """Sembol başına toplu market zekâsı özeti — scoring + dashboard bunu okur."""

    __tablename__ = "market_intelligence_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(30), nullable=False)
    news_quality_score = Column(Numeric(6, 2))  # 0-100
    social_momentum_score = Column(Numeric(6, 2))  # 0-100 (Faz 5'e kadar None)
    hype_risk = Column(Numeric(6, 2))  # 0-100
    news_sentiment = Column(String(20))
    news_count = Column(Integer, server_default="0")
    independent_news_count = Column(Integer, server_default="0")  # dedup sonrası
    duplicate_count = Column(Integer, server_default="0")
    event_summary = Column(JSONB)
    top_news = Column(JSONB)
    source_breakdown = Column(JSONB)
    ollama_used = Column(Boolean, server_default="false")
    warnings = Column(JSONB)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_mis_symbol_time", "symbol", "computed_at"),)


# ============================================================================ #
# ML Evaluation & Feature Store — SHARED (evren-geneli ölçüm/feature metadata)
# Faz 1: walk-forward evaluation + feature snapshot (etiket geç doldurulur).
# "eğitimler ortak" → ölçüm/ağırlık/feature metadata shared'da durur (coin_scores gibi).
# ============================================================================ #
class FeatureSnapshot(SharedBase):
    """Bir sembol+feature_set için hesaplanmış feature vektörü (eğitim/backtest kaynağı).

    `label` ileriye dönük pencere kapanınca (evaluation/feedback job) geri-doldurulur.
    """

    __tablename__ = "feature_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(30), nullable=False)
    feature_set_version = Column(String(40), nullable=False)  # v1_lstm_25 | v2_ensemble_advanced | technical_baseline_v1
    timeframe = Column(String(10))
    features = Column(JSONB)
    feature_hash = Column(String(64))
    horizon = Column(Integer)                 # etiket için ileri bar sayısı
    label = Column(Numeric(12, 6))            # forward return / yön (geç doldurulur)
    label_type = Column(String(20))           # forward_return | direction
    resolved = Column(Boolean, nullable=False, server_default="false")
    bar_time = Column(DateTime(timezone=True))  # snapshot'ın ait olduğu mum zamanı
    computed_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_feature_snapshots_symbol_ver_time", "symbol", "feature_set_version", "computed_at"),
    )


class ModelEvaluation(SharedBase):
    """Model/sinyal değerlendirme kaydı (walk-forward / holdout / forward-test).

    `metrics`: directional_accuracy/mae/rmse/hit_rate/win_rate/profit_factor/... ;
    `folds`: per-fold detay (walk-forward). `model_id` → model_registry.model_id (baseline'da None).
    """

    __tablename__ = "model_evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(120))            # model_registry.model_id'ye mantıksal referans
    symbol = Column(String(30), nullable=False)
    model_type = Column(String(50))           # baseline_technical | lstm | random_forest | ensemble ...
    feature_set_version = Column(String(40))
    eval_type = Column(String(30), nullable=False)  # walk_forward | holdout | forward_test
    timeframe = Column(String(10))
    horizon = Column(Integer)
    sample_count = Column(Integer)
    metrics = Column(JSONB)
    folds = Column(JSONB)
    window_start = Column(DateTime(timezone=True))
    window_end = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_model_evaluations_symbol_type_time", "symbol", "model_type", "created_at"),
    )


class ModelWeight(SharedBase):
    """Ensemble model ağırlıkları (symbol × model_type × feature_set × regime × timeframe).

    Faz 4'te simülasyon performansından EWMA ile güncellenir; EnsembleVoter karar anında okur.
    regime/timeframe = 'all' → segment-bağımsız varsayılan ağırlık.
    """

    __tablename__ = "model_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(30), nullable=False)
    model_type = Column(String(50), nullable=False)
    feature_set_version = Column(String(40))
    regime = Column(String(20), nullable=False, server_default="all")
    timeframe = Column(String(10), nullable=False, server_default="all")
    weight = Column(Numeric(10, 4), nullable=False, server_default="0")
    sample_count = Column(Integer, server_default="0")
    win_rate = Column(Numeric(6, 2))
    profit_factor = Column(Numeric(12, 4))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_model_weights_symbol_regime_tf", "symbol", "regime", "timeframe"),
    )


class MarketRegimeSnapshot(SharedBase):
    """Market rejimi + anomali skorları (evren-geneli). Faz 5: kural-tabanlı regime + pump/dump risk."""

    __tablename__ = "market_regime_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(30), nullable=False)
    timeframe = Column(String(10), nullable=False, server_default="4h")
    regime = Column(String(20))           # BULL_TREND/BEAR_TREND/SIDEWAYS/HIGH_VOLATILITY/...
    regime_confidence = Column(Numeric(5, 4))
    method = Column(String(20), server_default="rule")
    adx = Column(Numeric(8, 2))
    volatility = Column(Numeric(12, 6))
    anomaly_score = Column(Numeric(6, 2))
    pump_risk_score = Column(Numeric(6, 2))
    dump_risk_score = Column(Numeric(6, 2))
    features = Column(JSONB)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_market_regime_symbol_tf_time", "symbol", "timeframe", "computed_at"),
    )
