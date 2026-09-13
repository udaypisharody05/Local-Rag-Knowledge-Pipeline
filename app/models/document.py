"""Source document model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.chunk import DocumentChunk
    from app.models.job import IngestionJob


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_name: Mapped[str] = mapped_column(String(512), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_identifier: Mapped[str | None] = mapped_column(String(1024))
    original_filename: Mapped[str | None] = mapped_column(String(512))
    storage_path: Mapped[str | None] = mapped_column(String(1024))
    mime_type: Mapped[str | None] = mapped_column(String(255))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    content_hash: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(50), default="PENDING", server_default="PENDING")
    parser_version: Mapped[str | None] = mapped_column(String(100))
    chunking_config_hash: Mapped[str | None] = mapped_column(String(128))
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )
    jobs: Mapped[list["IngestionJob"]] = relationship(back_populates="document")
