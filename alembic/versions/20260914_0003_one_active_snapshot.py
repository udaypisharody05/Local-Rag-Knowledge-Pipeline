"""Enforce one authoritative active retrieval snapshot.

Revision ID: 20260914_0003
Revises: 20260914_0002
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "20260914_0003"
down_revision: str | None = "20260914_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "uq_retrieval_index_one_active",
        "retrieval_index_versions",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVATED'"),
    )


def downgrade() -> None:
    op.drop_index("uq_retrieval_index_one_active", table_name="retrieval_index_versions")
