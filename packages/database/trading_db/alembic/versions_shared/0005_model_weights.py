"""shared model_weights table (ensemble dynamic weighting)

Revision ID: shared_0005
Revises: shared_0004
Create Date: 2026-07-04

Faz 2 — Model Registry + Feature Versioning: ensemble ağırlıkları için model_weights.
Ağırlıklar Faz 4'te simülasyon performansından güncellenir (tablo şimdi boş başlar).
search_path env.py'de 'shared' olarak set edilir (schema= verilmez).
Prod-güvenli: yalnız CREATE; mevcut tabloları kırmaz.
"""
from alembic import op
import sqlalchemy as sa

revision = "shared_0005"
down_revision = "shared_0004"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "model_weights",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("model_type", sa.String(length=50), nullable=False),
        sa.Column("feature_set_version", sa.String(length=40), nullable=True),
        sa.Column("regime", sa.String(length=20), server_default="all", nullable=False),
        sa.Column("timeframe", sa.String(length=10), server_default="all", nullable=False),
        sa.Column("weight", sa.Numeric(10, 4), server_default="0", nullable=False),
        sa.Column("sample_count", sa.Integer(), server_default="0"),
        sa.Column("win_rate", sa.Numeric(6, 2), nullable=True),
        sa.Column("profit_factor", sa.Numeric(12, 4), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_model_weights"),
    )
    op.create_index(
        "ix_model_weights_symbol_regime_tf",
        "model_weights",
        ["symbol", "regime", "timeframe"],
    )


def downgrade():
    op.drop_table("model_weights")
