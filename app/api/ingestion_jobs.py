"""Persistent asynchronous ingestion job status endpoint."""

from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.core.security import require_api_key
from app.models import IngestionJob
from app.schemas.ingestion_jobs import IngestionJobResponse

router = APIRouter(
    prefix="/ingestion/jobs",
    tags=["ingestion"],
    dependencies=[Depends(require_api_key)],
)


@router.get("/{job_id}", response_model=IngestionJobResponse)
def get_ingestion_job(
    job_id: UUID, db: Session = Depends(get_db)
) -> IngestionJobResponse | JSONResponse:
    job = db.get(IngestionJob, job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "Ingestion job not found"})
    return IngestionJobResponse(
        job_id=job.id,
        status=job.status,
        source_name=job.source_name,
        document_id=job.document_id,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error_message,
    )
