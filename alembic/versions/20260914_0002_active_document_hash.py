"""Enforce active document content uniqueness.

Revision ID: 20260914_0002
Revises: 20260910_0001
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "20260914_0002"
down_revision: str | None = "20260910_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "uq_documents_active_content_hash",
        "documents",
        ["content_hash"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL AND content_hash IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_documents_active_content_hash", table_name="documents")
