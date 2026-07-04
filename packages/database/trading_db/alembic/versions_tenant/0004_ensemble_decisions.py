"""tenant ensemble_decisions table (Decision Layer audit)

Revision ID: tenant_0004
Revises: tenant_0003
Create Date: 2026-07-04

Faz 7 — Decision Layer: nihai karar (score_full_v2 superset) audit'i, tenant başına.
search_path env.py'de tenant şeması olarak set edilir (schema= verilmez).
Prod-güvenli: yalnız CREATE; mevcut tabloları kırmaz. Tüm tenant'lara `migrate all-tenants` ile uygulanır.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "tenant_0004"
down_revision = "tenant_0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ensemble_decisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("timeframe", sa.String(length=10), server_default="4h"),
        sa.Column("regime", sa.String(length=20), nullable=True),
        sa.Column("recommendation", sa.String(length=20), nullable=True),
        sa.Column("final_action", sa.String(length=20), nullable=True),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=True),
        sa.Column("opportunity_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("risk_score", sa.Numeric(6, 2), nullable=True),
        sa.Column("data_quality", sa.Numeric(5, 4), nullable=True),
        sa.Column("ensemble", postgresql.JSONB(), nullable=True),
        sa.Column("multi_timeframe", postgresql.JSONB(), nullable=True),
        sa.Column("blocked_reasons", postgresql.JSONB(), nullable=True),
        sa.Column("decision", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_ensemble_decisions"),
    )
    op.create_index("ix_ensemble_decisions_symbol_time", "ensemble_decisions",
                    ["symbol", "created_at"])


def downgrade():
    op.drop_table("ensemble_decisions")
