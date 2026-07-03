"""shared ML evaluation + feature snapshot tables

Revision ID: shared_0004
Revises: shared_0003
Create Date: 2026-07-04

Faz 1 — Ölçüm altyapısı: walk-forward evaluation (model_evaluations) +
feature snapshot store (feature_snapshots; etiket ileriye dönük pencere kapanınca
geç doldurulur). search_path env.py'de 'shared' olarak set edilir (schema= verilmez).
Prod-güvenli: yalnız CREATE; mevcut tabloları kırmaz.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "shared_0004"
down_revision = "shared_0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "feature_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("feature_set_version", sa.String(length=40), nullable=False),
        sa.Column("timeframe", sa.String(length=10), nullable=True),
        sa.Column("features", postgresql.JSONB(), nullable=True),
        sa.Column("feature_hash", sa.String(length=64), nullable=True),
        sa.Column("horizon", sa.Integer(), nullable=True),
        sa.Column("label", sa.Numeric(12, 6), nullable=True),
        sa.Column("label_type", sa.String(length=20), nullable=True),
        sa.Column("resolved", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("bar_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_feature_snapshots"),
    )
    op.create_index(
        "ix_feature_snapshots_symbol_ver_time",
        "feature_snapshots",
        ["symbol", "feature_set_version", "computed_at"],
    )

    op.create_table(
        "model_evaluations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_id", sa.String(length=120), nullable=True),
        sa.Column("symbol", sa.String(length=30), nullable=False),
        sa.Column("model_type", sa.String(length=50), nullable=True),
        sa.Column("feature_set_version", sa.String(length=40), nullable=True),
        sa.Column("eval_type", sa.String(length=30), nullable=False),
        sa.Column("timeframe", sa.String(length=10), nullable=True),
        sa.Column("horizon", sa.Integer(), nullable=True),
        sa.Column("sample_count", sa.Integer(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("folds", postgresql.JSONB(), nullable=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_model_evaluations"),
    )
    op.create_index(
        "ix_model_evaluations_symbol_type_time",
        "model_evaluations",
        ["symbol", "model_type", "created_at"],
    )


def downgrade():
    op.drop_table("model_evaluations")
    op.drop_table("feature_snapshots")
