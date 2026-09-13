"""Document API response schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: UUID
    source_name: str
    source_type: str
    status: str
    chunk_count: int


class DocumentSummary(BaseModel):
    id: UUID
    source_name: str
    source_type: str
    status: str
    size_bytes: int | None
    chunk_count: int
    created_at: datetime
    indexed_at: datetime | None


class DocumentDetail(DocumentSummary):
    original_filename: str | None
    mime_type: str | None
    content_hash: str | None
    parser_version: str | None
    chunking_config_hash: str | None
