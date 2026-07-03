"""tenant automation tables (automation_signals, automation_alerts, trade_decisions)

Revision ID: tenant_0002
Revises: tenant_0001
Create Date: 2026-07-03

Otomasyon motorunun kullanıcıya-özel çıktıları (sinyal/uyarı/trade kararı).
Her tenant şemasına uygulanır (run_all_tenant_migrations).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "tenant_0002"
down_revision = "tenant_0001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "automation_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("signal", sa.String(length=20), nullable=False),
        sa.Column("recommendation", sa.String(length=20), nullable=True),
        sa.Column("opportunity_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("risk_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=True),
        sa.Column("reasons", postgresql.JSONB(), nullable=True),
        sa.Column("warnings", postgresql.JSONB(), nullable=True),
        sa.Column("source", sa.String(length=30), server_default="automation"),
        sa.Column("status", sa.String(length=20), server_default="new", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_automation_signals"),
    )
    op.create_index("ix_automation_signals_symbol_time", "automation_signals", ["symbol", "created_at"])

    op.create_table(
        "automation_alerts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=True),
        sa.Column("level", sa.String(length=20), server_default="info", nullable=False),
        sa.Column("title", sa.String(length=200), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("is_read", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_automation_alerts"),
    )
    op.create_index("ix_automation_alerts_time", "automation_alerts", ["created_at"])

    op.create_table(
        "trade_decisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("action", sa.String(length=20), nullable=False),
        sa.Column("decided", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=True),
        sa.Column("risk_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("is_testnet", sa.Boolean(), server_default="true"),
        sa.Column("is_simulated", sa.Boolean(), server_default="true"),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_trade_decisions"),
    )
    op.create_index("ix_trade_decisions_symbol_time", "trade_decisions", ["symbol", "created_at"])


def downgrade():
    op.drop_table("trade_decisions")
    op.drop_table("automation_alerts")
    op.drop_table("automation_signals")
