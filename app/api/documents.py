"""Synchronous document ingestion and metadata endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.core.config import settings
from app.core.security import require_api_key
from app.ingestion import IngestionError, IngestionService
from app.models import Document, DocumentChunk
from app.schemas.documents import DocumentDetail, DocumentSummary, UploadResponse

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/upload", response_model=UploadResponse, status_code=201)
def upload_document(
    file: UploadFile = File(...), db: Session = Depends(get_db)
) -> UploadResponse | JSONResponse:
    try:
        return IngestionService(db, settings).ingest(file)
    except IngestionError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.response_body())


@router.get("", response_model=list[DocumentSummary])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentSummary]:
    rows = db.execute(
        select(Document, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk)
        .where(Document.deleted_at.is_(None))
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    ).all()
    return [_summary(document, chunk_count) for document, chunk_count in rows]


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(document_id: UUID, db: Session = Depends(get_db)) -> DocumentDetail | JSONResponse:
    row = db.execute(
        select(Document, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk)
        .where(Document.id == document_id, Document.deleted_at.is_(None))
        .group_by(Document.id)
    ).one_or_none()
    if row is None:
        return JSONResponse(status_code=404, content={"detail": "Document not found"})
    document, chunk_count = row
    summary = _summary(document, chunk_count)
    return DocumentDetail(
        **summary.model_dump(),
        original_filename=document.original_filename,
        mime_type=document.mime_type,
        content_hash=document.content_hash,
        parser_version=document.parser_version,
        chunking_config_hash=document.chunking_config_hash,
    )


def _summary(document: Document, chunk_count: int) -> DocumentSummary:
    return DocumentSummary(
        id=document.id,
        source_name=document.source_name,
        source_type=document.source_type,
        status=document.status,
        size_bytes=document.size_bytes,
        chunk_count=chunk_count,
        created_at=document.created_at,
        indexed_at=document.indexed_at,
    )
