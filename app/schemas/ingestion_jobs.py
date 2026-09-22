"""Asynchronous ingestion job API schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class AsyncUploadResponse(BaseModel):
    job_id: UUID
    status: str
    source_name: str


class IngestionJobResponse(BaseModel):
    job_id: UUID
    status: str
    source_name: str | None
    document_id: UUID | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error: str | None
