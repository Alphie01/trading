"""tenant initial schema (watchlist, trades, positions, analysis_results, portfolio_summary, system_state)

Revision ID: tenant_0001
Revises:
Create Date: 2026-07-03

Not: search_path env.py'de hedef tenant şemasına set edilir → tablolar o şemada oluşur.
coin_symbol kolonları shared coins kataloğuna MANTIKSAL referanstır (DB FK'sı kurulmaz).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "tenant_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "watchlist",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("coin_symbol", sa.String(length=20), nullable=False),
        sa.Column("added_date", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("last_analysis", sa.DateTime(timezone=True), nullable=True),
        sa.Column("analysis_count", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_watchlist"),
        sa.UniqueConstraint("coin_symbol", name="uq_watchlist_coin_symbol"),
    )

    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("coin_symbol", sa.String(length=20), nullable=False),
        sa.Column("trade_type", sa.String(length=20), nullable=False),
        sa.Column("price", sa.Numeric(18, 8), nullable=False),
        sa.Column("quantity", sa.Numeric(18, 8), nullable=False),
        sa.Column("total_value", sa.Numeric(18, 8), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("confidence", sa.Numeric(10, 4), nullable=True),
        sa.Column("news_sentiment", sa.Numeric(10, 4), nullable=True),
        sa.Column("whale_activity", sa.Numeric(10, 4), nullable=True),
        sa.Column("yigit_signal", sa.String(length=50), nullable=True),
        sa.Column("trade_reason", sa.Text(), nullable=True),
        sa.Column("exchange", sa.String(length=50), server_default="binance"),
        sa.Column("fees", sa.Numeric(18, 8), server_default="0"),
        sa.Column("is_simulated", sa.Boolean(), server_default="true", nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_trades"),
    )
    op.create_index("ix_trades_coin_symbol", "trades", ["coin_symbol"])

    op.create_table(
        "positions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("coin_symbol", sa.String(length=20), nullable=False),
        sa.Column("position_type", sa.String(length=20), nullable=False),
        sa.Column("entry_price", sa.Numeric(18, 8), nullable=False),
        sa.Column("current_price", sa.Numeric(18, 8), nullable=True),
        sa.Column("quantity", sa.Numeric(18, 8), nullable=False),
        sa.Column("entry_value", sa.Numeric(18, 8), nullable=False),
        sa.Column("current_value", sa.Numeric(18, 8), nullable=True),
        sa.Column("unrealized_pnl", sa.Numeric(18, 8), nullable=True),
        sa.Column("entry_timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("last_update", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("is_open", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("stop_loss", sa.Numeric(18, 8), nullable=True),
        sa.Column("take_profit", sa.Numeric(18, 8), nullable=True),
        sa.Column("leverage", sa.Numeric(10, 4), server_default="1"),
        sa.PrimaryKeyConstraint("id", name="pk_positions"),
    )
    op.create_index("ix_positions_coin_symbol", "positions", ["coin_symbol"])

    op.create_table(
        "analysis_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("coin_symbol", sa.String(length=20), nullable=False),
        sa.Column("predicted_price", sa.Numeric(18, 8), nullable=False),
        sa.Column("current_price", sa.Numeric(18, 8), nullable=False),
        sa.Column("price_change_percent", sa.Numeric(18, 8), nullable=False),
        sa.Column("confidence", sa.Numeric(10, 4), nullable=False),
        sa.Column("news_sentiment", sa.Numeric(10, 4), nullable=True),
        sa.Column("whale_activity_score", sa.Numeric(10, 4), nullable=True),
        sa.Column("yigit_position", sa.Integer(), nullable=True),
        sa.Column("yigit_signal", sa.String(length=50), nullable=True),
        sa.Column("trend_strength", sa.Numeric(10, 4), nullable=True),
        sa.Column("analysis_timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("model_type", sa.String(length=50), server_default="LSTM"),
        sa.Column("features_used", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_analysis_results"),
    )
    op.create_index(
        "ix_analysis_results_symbol_time", "analysis_results", ["coin_symbol", "analysis_timestamp"]
    )

    op.create_table(
        "portfolio_summary",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("total_balance", sa.Numeric(18, 8), nullable=False),
        sa.Column("invested_amount", sa.Numeric(18, 8), nullable=False),
        sa.Column("current_value", sa.Numeric(18, 8), nullable=False),
        sa.Column("total_pnl", sa.Numeric(18, 8), nullable=False),
        sa.Column("total_pnl_percent", sa.Numeric(18, 8), nullable=False),
        sa.Column("active_positions", sa.Integer(), nullable=False),
        sa.Column("successful_trades", sa.Integer(), nullable=False),
        sa.Column("total_trades", sa.Integer(), nullable=False),
        sa.Column("win_rate", sa.Numeric(10, 4), nullable=False),
        sa.Column("summary_date", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_portfolio_summary"),
    )

    op.create_table(
        "system_state",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("state_key", sa.String(length=150), nullable=False),
        sa.Column("state_value", postgresql.JSONB(), nullable=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_system_state"),
        sa.UniqueConstraint("state_key", name="uq_system_state_state_key"),
    )


def downgrade():
    op.drop_table("system_state")
    op.drop_table("portfolio_summary")
    op.drop_table("analysis_results")
    op.drop_table("positions")
    op.drop_table("trades")
    op.drop_table("watchlist")
