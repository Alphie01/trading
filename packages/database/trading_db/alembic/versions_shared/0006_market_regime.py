"""shared market_regime_snapshots table (regime + anomaly)

Revision ID: shared_0006
Revises: shared_0005
Create Date: 2026-07-04

Faz 5 — Market Regime + Anomaly: kural-tabanlı rejim + pump/dump/anomaly risk skorları.
search_path env.py'de 'shared' olarak set edilir (schema= verilmez).
Prod-güvenli: yalnız CREATE; mevcut tabloları kırmaz.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "shared_0006"
down_revision = "shared_0005"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "market_regime_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("timeframe", sa.String(length=10), server_default="4h", nullable=False),
        sa.Column("regime", sa.String(length=20), nullable=True),
        sa.Column("regime_confidence", sa.Numeric(5, 4), nullable=True),
        sa.Column("method", sa.String(length=20), server_default="rule"),
        sa.Column("adx", sa.Numeric(8, 2), nullable=True),
        sa.Column("volatility", sa.Numeric(12, 6), nullable=True),
        sa.Column("anomaly_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("pump_risk_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("dump_risk_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("features", postgresql.JSONB(), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_market_regime_snapshots"),
    )
    op.create_index(
        "ix_market_regime_symbol_tf_time",
        "market_regime_snapshots",
        ["symbol", "timeframe", "computed_at"],
    )


def downgrade():
    op.drop_table("market_regime_snapshots")
