"""shared jobs table (async iş kuyruğu — RQ worker)

Revision ID: shared_0008
Revises: shared_0007
Create Date: 2026-07-04

Web/AI ayrıştırma Faz 2: ağır işler (eğitim/analiz/intelligence) web'i bloklamasın →
RQ worker'ın işlediği job kayıtları. search_path env.py'de 'shared'. Prod-güvenli: yalnız CREATE.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "shared_0008"
down_revision = "shared_0007"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=80), nullable=False),
        sa.Column("job_type", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="queued", nullable=False),
        sa.Column("symbol", sa.String(length=30), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
        sa.Column("progress_percent", sa.Integer(), server_default="0"),
        sa.Column("current_step", sa.String(length=160), nullable=True),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=80), nullable=True),
        sa.Column("tenant_schema", sa.String(length=63), nullable=True),
        sa.Column("worker_id", sa.String(length=80), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
        sa.UniqueConstraint("job_id", name="uq_jobs_job_id"),
    )
    op.create_index("ix_jobs_status_created", "jobs", ["status", "created_at"])


def downgrade():
    op.drop_table("jobs")
