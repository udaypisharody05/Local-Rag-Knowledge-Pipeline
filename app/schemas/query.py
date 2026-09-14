"""Grounded query API schemas."""

from uuid import UUID

from pydantic import BaseModel

from app.schemas.retrieval import HybridSearchRequest


class QueryRequest(HybridSearchRequest):
    pass


class CitationResponse(BaseModel):
    source_id: str
    document_id: UUID
    chunk_id: UUID
    source_name: str
    source_type: str
    page_number: int | None
    section_title: str | None
    repo_relative_path: str | None


class QueryRetrievalMetadata(BaseModel):
    snapshot_version: int
    retrieved_chunk_ids: list[UUID]
    context_chunk_ids: list[UUID]
    fusion_strategy: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationResponse]
    retrieval: QueryRetrievalMetadata
    model: str
