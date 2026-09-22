"""Synchronous/asynchronous document ingestion and metadata endpoints."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_ingestion_enqueuer
from app.core.config import settings
from app.core.security import require_api_key
from app.ingestion import IngestionError, IngestionService
from app.ingestion.storage import JobStagingStorage, UploadTooLargeError
from app.models import Document, DocumentChunk, IngestionJob
from app.schemas.documents import DocumentDetail, DocumentSummary, UploadResponse
from app.schemas.ingestion_jobs import AsyncUploadResponse

logger = logging.getLogger(__name__)

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


@router.post("/upload/async", response_model=AsyncUploadResponse, status_code=202)
def upload_document_async(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    enqueue: Callable[[str], object] = Depends(get_ingestion_enqueuer),
) -> AsyncUploadResponse | JSONResponse:
    service = IngestionService(db, settings)
    job_id = uuid4()
    staging = JobStagingStorage(
        settings.staging_root, settings.max_upload_size_mb * 1024 * 1024
    )
    job_created = False
    try:
        filename, extension = service.validate_filename(file.filename)
        service.validate_declared_mime(extension, file.content_type)
        try:
            staged = staging.stage(file.file, job_id, extension)
        except UploadTooLargeError:
            raise IngestionError(
                413, f"File exceeds the {settings.max_upload_size_mb} MB upload limit"
            ) from None
        service.validate_staged(extension, staged)
        duplicate = service.find_duplicate(staged.content_hash)
        if duplicate is not None:
            raise service.duplicate_error(duplicate)

        job = IngestionJob(
            id=job_id,
            job_type="DOCUMENT_UPLOAD",
            status="QUEUED",
            source_name=filename,
        )
        db.add(job)
        db.commit()
        job_created = True

        try:
            queued_task = enqueue(str(job_id))
        except Exception:
            db.rollback()
            job = db.get(IngestionJob, job_id)
            if job is not None and job.status == "QUEUED":
                job.status = "FAILED"
                job.completed_at = datetime.now(UTC)
                job.error_message = "Ingestion queue is unavailable"
                db.commit()
            staging.remove(job_id)
            logger.warning("ingestion_job_enqueue_failed job_id=%s", job_id)
            return JSONResponse(
                status_code=503, content={"detail": "Ingestion queue is unavailable"}
            )

        task_id = getattr(queued_task, "id", None)
        if task_id:
            job.celery_task_id = str(task_id)
            db.commit()
        logger.info("ingestion_job_queued job_id=%s", job_id)
        return AsyncUploadResponse(job_id=job_id, status="QUEUED", source_name=filename)
    except IngestionError as exc:
        db.rollback()
        if not job_created:
            staging.remove(job_id)
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
