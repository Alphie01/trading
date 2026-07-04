"""shared signal_feedback table (simulation → weights feedback)

Revision ID: shared_0007
Revises: shared_0006
Create Date: 2026-07-04

Faz 8 — Feedback kapanışı: kapalı simülasyon sonuçlarının (symbol × regime × bucket) agregatı;
model_weights'i regime-spesifik besler + false_signal_reasons.
search_path env.py'de 'shared' olarak set edilir (schema= verilmez).
Prod-güvenli: yalnız CREATE; mevcut tabloları kırmaz.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "shared_0007"
down_revision = "shared_0006"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "signal_feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("feature_set_version", sa.String(length=40), nullable=True),
        sa.Column("regime", sa.String(length=20), server_default="all", nullable=False),
        sa.Column("timeframe", sa.String(length=10), server_default="all", nullable=False),
        sa.Column("signal_bucket", sa.String(length=10), server_default="unknown", nullable=False),
        sa.Column("sample_count", sa.Integer(), server_default="0"),
        sa.Column("win_count", sa.Integer(), server_default="0"),
        sa.Column("win_rate", sa.Numeric(6, 2), nullable=True),
        sa.Column("avg_pnl", sa.Numeric(18, 8), nullable=True),
        sa.Column("profit_factor", sa.Numeric(12, 4), nullable=True),
        sa.Column("quality_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("false_signal_reasons", postgresql.JSONB(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_signal_feedback"),
    )
    op.create_index("ix_signal_feedback_symbol_regime_bucket", "signal_feedback",
                    ["symbol", "regime", "signal_bucket"])


def downgrade():
    op.drop_table("signal_feedback")
