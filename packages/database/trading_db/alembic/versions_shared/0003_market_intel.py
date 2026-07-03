"""shared market intelligence tables + coin_scores intel columns

Revision ID: shared_0003
Revises: shared_0002
Create Date: 2026-07-04

Market Intelligence katmanı — haber/sosyal/LLM zekâsı (evren-geneli, shared).
search_path env.py'de 'shared' olarak set edilir (schema= verilmez).
Prod-güvenli: yalnız CREATE/ADD; mevcut tabloları kırmaz.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "shared_0003"
down_revision = "shared_0002"
branch_labels = None
depends_on = None


def upgrade():
    # coin_scores'a Market Intelligence bileşen kolonları (nullable → mevcut satırları kırmaz)
    op.add_column("coin_scores", sa.Column("news_quality_score", sa.Numeric(6, 2), nullable=True))
    op.add_column("coin_scores", sa.Column("social_momentum_score", sa.Numeric(6, 2), nullable=True))
    op.add_column("coin_scores", sa.Column("hype_risk", sa.Numeric(6, 2), nullable=True))

    op.create_table(
        "intelligence_sources",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=True),
        sa.Column("tier", sa.Integer(), nullable=True),
        sa.Column("reputation_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("impact_weight", sa.Numeric(5, 3), nullable=True),
        sa.Column("allow_trade_signal_boost", sa.Boolean(), server_default="true"),
        sa.Column("regional", sa.Boolean(), server_default="false"),
        sa.Column("enabled", sa.Boolean(), server_default="true"),
        sa.Column("last_collected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_intelligence_sources"),
        sa.UniqueConstraint("name", name="uq_intelligence_sources_name"),
    )

    op.create_table(
        "news_items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(length=120), nullable=True),
        sa.Column("source_tier", sa.Integer(), nullable=True),
        sa.Column("symbol", sa.String(length=30), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("duplicate_group_id", sa.String(length=64), nullable=True),
        sa.Column("raw_sentiment", sa.Numeric(6, 3), nullable=True),
        sa.Column("data_source", sa.String(length=40), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_news_items"),
        sa.UniqueConstraint("content_hash", name="uq_news_items_content_hash"),
    )
    op.create_index("ix_news_items_symbol_time", "news_items", ["symbol", "published_at"])

    op.create_table(
        "news_analysis_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("news_item_id", sa.Integer(), nullable=True),
        sa.Column("symbol", sa.String(length=30), nullable=True),
        sa.Column("sentiment", sa.String(length=20), nullable=True),
        sa.Column("sentiment_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("quality_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("relevance_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("impact_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("freshness_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("hype_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("risk_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("event_type", sa.String(length=40), nullable=True),
        sa.Column("ollama_model", sa.String(length=80), nullable=True),
        sa.Column("ollama_confidence", sa.Numeric(5, 4), nullable=True),
        sa.Column("analysis_json", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_news_analysis_results"),
        sa.ForeignKeyConstraint(
            ["news_item_id"], ["news_items.id"],
            name="fk_news_analysis_results_news_item_id_news_items",
        ),
    )
    op.create_index("ix_news_analysis_symbol_time", "news_analysis_results", ["symbol", "created_at"])

    op.create_table(
        "market_intelligence_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("news_quality_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("social_momentum_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("hype_risk", sa.Numeric(6, 2), nullable=True),
        sa.Column("news_sentiment", sa.String(length=20), nullable=True),
        sa.Column("news_count", sa.Integer(), server_default="0"),
        sa.Column("independent_news_count", sa.Integer(), server_default="0"),
        sa.Column("duplicate_count", sa.Integer(), server_default="0"),
        sa.Column("event_summary", postgresql.JSONB(), nullable=True),
        sa.Column("top_news", postgresql.JSONB(), nullable=True),
        sa.Column("source_breakdown", postgresql.JSONB(), nullable=True),
        sa.Column("ollama_used", sa.Boolean(), server_default="false"),
        sa.Column("warnings", postgresql.JSONB(), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_market_intelligence_snapshots"),
    )
    op.create_index("ix_mis_symbol_time", "market_intelligence_snapshots", ["symbol", "computed_at"])


def downgrade():
    op.drop_table("market_intelligence_snapshots")
    op.drop_table("news_analysis_results")
    op.drop_table("news_items")
    op.drop_table("intelligence_sources")
    op.drop_column("coin_scores", "hype_risk")
    op.drop_column("coin_scores", "social_momentum_score")
    op.drop_column("coin_scores", "news_quality_score")
