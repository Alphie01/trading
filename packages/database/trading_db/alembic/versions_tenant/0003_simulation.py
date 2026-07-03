"""tenant simulation tables (paper trading)

Revision ID: tenant_0003
Revises: tenant_0002
Create Date: 2026-07-03

Simulation / Paper Trading — SANAL; canlı Binance işlemiyle ilgisi yok.
Her tenant şemasına uygulanır (run_all_tenant_migrations).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "tenant_0003"
down_revision = "tenant_0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "simulation_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="RUNNING", nullable=False),
        sa.Column("mode", sa.String(length=20), server_default="SPOT", nullable=False),
        sa.Column("quote_currency", sa.String(length=20), server_default="USDT"),
        sa.Column("initial_balance", sa.Numeric(24, 8), nullable=False),
        sa.Column("current_balance", sa.Numeric(24, 8), nullable=False),
        sa.Column("equity", sa.Numeric(24, 8), nullable=True),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column("created_by", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_simulation_runs"),
    )
    op.create_index("ix_simulation_runs_status", "simulation_runs", ["status"])

    op.create_table(
        "simulation_positions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("simulation_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("mode", sa.String(length=20), nullable=False),
        sa.Column("side", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="OPEN", nullable=False),
        sa.Column("entry_price", sa.Numeric(24, 8), nullable=False),
        sa.Column("current_price", sa.Numeric(24, 8), nullable=True),
        sa.Column("exit_price", sa.Numeric(24, 8), nullable=True),
        sa.Column("quantity", sa.Numeric(28, 12), nullable=False),
        sa.Column("notional_value", sa.Numeric(24, 8), nullable=True),
        sa.Column("margin_used", sa.Numeric(24, 8), nullable=True),
        sa.Column("leverage", sa.Numeric(10, 4), nullable=True),
        sa.Column("entry_fee", sa.Numeric(24, 8), server_default="0"),
        sa.Column("exit_fee", sa.Numeric(24, 8), server_default="0"),
        sa.Column("unrealized_pnl", sa.Numeric(24, 8), server_default="0"),
        sa.Column("realized_pnl", sa.Numeric(24, 8), server_default="0"),
        sa.Column("net_pnl", sa.Numeric(24, 8), server_default="0"),
        sa.Column("roi_percent", sa.Numeric(12, 4), nullable=True),
        sa.Column("liquidation_price", sa.Numeric(24, 8), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("entry_signal_id", sa.String(length=64), nullable=True),
        sa.Column("exit_signal_id", sa.String(length=64), nullable=True),
        sa.Column("entry_reason", sa.Text(), nullable=True),
        sa.Column("exit_reason", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_simulation_positions"),
    )
    op.create_index("ix_simulation_positions_sim_status", "simulation_positions", ["simulation_id", "status"])
    op.create_index("ix_simulation_positions_sim_symbol", "simulation_positions", ["simulation_id", "symbol"])

    op.create_table(
        "simulation_trades",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("simulation_id", sa.Integer(), nullable=False),
        sa.Column("position_id", sa.Integer(), nullable=True),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("side", sa.String(length=20), nullable=False),
        sa.Column("order_type", sa.String(length=10), server_default="taker"),
        sa.Column("price", sa.Numeric(24, 8), nullable=False),
        sa.Column("quantity", sa.Numeric(28, 12), nullable=False),
        sa.Column("notional_value", sa.Numeric(24, 8), nullable=True),
        sa.Column("fee", sa.Numeric(24, 8), server_default="0"),
        sa.Column("fee_asset", sa.String(length=20), server_default="USDT"),
        sa.Column("slippage", sa.Numeric(12, 8), nullable=True),
        sa.Column("signal_id", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_simulation_trades"),
    )
    op.create_index("ix_simulation_trades_sim_time", "simulation_trades", ["simulation_id", "executed_at"])

    op.create_table(
        "simulation_reports",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("simulation_id", sa.Integer(), nullable=False),
        sa.Column("period_type", sa.String(length=10), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("starting_balance", sa.Numeric(24, 8), nullable=True),
        sa.Column("ending_balance", sa.Numeric(24, 8), nullable=True),
        sa.Column("net_profit", sa.Numeric(24, 8), nullable=True),
        sa.Column("gross_profit", sa.Numeric(24, 8), nullable=True),
        sa.Column("gross_loss", sa.Numeric(24, 8), nullable=True),
        sa.Column("total_fees", sa.Numeric(24, 8), nullable=True),
        sa.Column("roi_percent", sa.Numeric(12, 4), nullable=True),
        sa.Column("win_rate", sa.Numeric(6, 2), nullable=True),
        sa.Column("loss_rate", sa.Numeric(6, 2), nullable=True),
        sa.Column("total_trades", sa.Integer(), nullable=True),
        sa.Column("winning_trades", sa.Integer(), nullable=True),
        sa.Column("losing_trades", sa.Integer(), nullable=True),
        sa.Column("max_drawdown", sa.Numeric(12, 4), nullable=True),
        sa.Column("profit_factor", sa.Numeric(12, 4), nullable=True),
        sa.Column("best_trade", sa.Numeric(24, 8), nullable=True),
        sa.Column("worst_trade", sa.Numeric(24, 8), nullable=True),
        sa.Column("avg_win", sa.Numeric(24, 8), nullable=True),
        sa.Column("avg_loss", sa.Numeric(24, 8), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_simulation_reports"),
    )
    op.create_index("ix_simulation_reports_sim_period", "simulation_reports", ["simulation_id", "period_type"])


def downgrade():
    op.drop_table("simulation_reports")
    op.drop_table("simulation_trades")
    op.drop_table("simulation_positions")
    op.drop_table("simulation_runs")
