"""shared automation tables (discovery_candidates, coin_scores, automation_runs)

Revision ID: shared_0002
Revises: shared_0001
Create Date: 2026-07-03

Otomasyon motoru — evren-geneli (shared) market zekâsı tabloları.
search_path env.py'de 'shared' olarak set edilir.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "shared_0002"
down_revision = "shared_0001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "discovery_candidates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("base_asset", sa.String(length=20), nullable=True),
        sa.Column("quote_asset", sa.String(length=20), nullable=True),
        sa.Column("exchange", sa.String(length=30), server_default="binance"),
        sa.Column("volume_24h", sa.Numeric(24, 4), nullable=True),
        sa.Column("price_change_24h", sa.Numeric(18, 8), nullable=True),
        sa.Column("volume_change_score", sa.Numeric(10, 4), nullable=True),
        sa.Column("volatility_score", sa.Numeric(10, 4), nullable=True),
        sa.Column("liquidity_score", sa.Numeric(10, 4), nullable=True),
        sa.Column("is_new_listing_candidate", sa.Boolean(), server_default="false"),
        sa.Column("discovery_reason", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=30), server_default="discovered", nullable=False),
        sa.Column("discovered_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_discovery_candidates"),
        sa.UniqueConstraint("symbol", name="uq_discovery_candidates_symbol"),
    )
    op.create_index("ix_discovery_candidates_status", "discovery_candidates", ["status"])

    op.create_table(
        "coin_scores",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("opportunity_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("risk_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=True),
        sa.Column("technical_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("volume_liquidity_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("ai_prediction_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("sentiment_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("whale_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("recommendation", sa.String(length=20), nullable=True),
        sa.Column("reasons", postgresql.JSONB(), nullable=True),
        sa.Column("warnings", postgresql.JSONB(), nullable=True),
        sa.Column("scored_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_coin_scores"),
    )
    op.create_index("ix_coin_scores_symbol_time", "coin_scores", ["symbol", "scored_at"])

    op.create_table(
        "automation_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_type", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="running", nullable=False),
        sa.Column("scanned_count", sa.Integer(), server_default="0"),
        sa.Column("passed_count", sa.Integer(), server_default="0"),
        sa.Column("watchlisted_count", sa.Integer(), server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_automation_runs"),
    )


def downgrade():
    op.drop_table("automation_runs")
    op.drop_table("coin_scores")
    op.drop_table("discovery_candidates")
