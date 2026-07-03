"""shared initial schema (tenants, users, coins, prediction_cache, model_registry)

Revision ID: shared_0001
Revises:
Create Date: 2026-07-03

Not: search_path env.py'de ``shared`` olarak set edilir → tablolar shared şemada oluşur.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "shared_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "tenants",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("slug", sa.String(length=63), nullable=False),
        sa.Column("schema_name", sa.String(length=63), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=True),
        sa.Column("status", sa.String(length=20), server_default="provisioning", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_tenants"),
        sa.UniqueConstraint("slug", name="uq_tenants_slug"),
        sa.UniqueConstraint("schema_name", name="uq_tenants_schema_name"),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("role", sa.String(length=20), server_default="user", nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("username", name="uq_users_username"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_users_tenant_id_tenants", ondelete="SET NULL"
        ),
    )

    op.create_table(
        "coins",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=True),
        sa.Column("added_date", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("current_price", sa.Numeric(18, 8), nullable=True),
        sa.Column("price_change_24h", sa.Numeric(18, 8), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_coins"),
        sa.UniqueConstraint("symbol", name="uq_coins_symbol"),
    )

    op.create_table(
        "prediction_cache",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("coin_symbol", sa.String(length=20), nullable=False),
        sa.Column("model_type", sa.String(length=50), nullable=False),
        sa.Column("prediction_data", postgresql.JSONB(), nullable=False),
        sa.Column("technical_analysis", postgresql.JSONB(), nullable=True),
        sa.Column("news_analysis", postgresql.JSONB(), nullable=True),
        sa.Column("whale_analysis", postgresql.JSONB(), nullable=True),
        sa.Column("yigit_analysis", postgresql.JSONB(), nullable=True),
        sa.Column("trade_signal", postgresql.JSONB(), nullable=True),
        sa.Column("cache_timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_valid", sa.Boolean(), server_default="true", nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_prediction_cache"),
    )
    op.create_index(
        "ix_prediction_cache_symbol_time", "prediction_cache", ["coin_symbol", "cache_timestamp"]
    )
    op.create_index("ix_prediction_cache_expires", "prediction_cache", ["expires_at", "is_valid"])

    op.create_table(
        "model_registry",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("coin_symbol", sa.String(length=20), nullable=False),
        sa.Column("model_type", sa.String(length=50), nullable=False),
        sa.Column("model_id", sa.String(length=120), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("feature_count", sa.Integer(), nullable=True),
        sa.Column("data_hash", sa.String(length=64), nullable=True),
        sa.Column("version", sa.String(length=20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("last_trained", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_model_registry"),
    )
    op.create_index("ix_model_registry_symbol_type", "model_registry", ["coin_symbol", "model_type"])


def downgrade():
    op.drop_table("model_registry")
    op.drop_table("prediction_cache")
    op.drop_table("coins")
    op.drop_table("users")
    op.drop_table("tenants")
