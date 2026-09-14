"""Dense retrieval API schemas."""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class RebuildResponse(BaseModel):
    snapshot_version: int
    status: str
    chunk_count: int
    embedding_model: str
    embedding_dimension: int


class DenseSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    k: int | None = Field(default=None, ge=1)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain text")
        return value


class DenseSearchResult(BaseModel):
    rank: int
    score: float
    chunk_id: UUID
    document_id: UUID
    text: str
    source_name: str
    source_type: str
    page_number: int | None
    section_title: str | None


class DenseSearchResponse(BaseModel):
    query: str
    snapshot_version: int
    results: list[DenseSearchResult]


class RetrievalStatusResponse(BaseModel):
    available: bool
    snapshot_version: int | None
    chunk_count: int | None = None
    embedding_model: str | None = None
