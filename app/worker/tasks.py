"""Celery tasks and directly testable asynchronous ingestion processor."""

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, settings
from app.db.session import SessionLocal
from app.ingestion import IngestionError, IngestionService
from app.ingestion.storage import JobStagingStorage
from app.models import IngestionJob
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)
TERMINAL_STATUSES = {"COMPLETED", "FAILED"}


class IngestionJobProcessor:
    def __init__(self, db: Session, app_settings: Settings) -> None:
        self.db = db
        self.settings = app_settings
        self.staging = JobStagingStorage(
            app_settings.staging_root,
            app_settings.max_upload_size_mb * 1024 * 1024,
        )

    def process(self, job_id: UUID) -> None:
        response = None
        job = self.db.scalar(
            select(IngestionJob).where(IngestionJob.id == job_id).with_for_update()
        )
        if job is None:
            logger.warning("ingestion_job_missing job_id=%s", job_id)
            return
        if job.status in TERMINAL_STATUSES:
            self.staging.remove(job_id)
            logger.info("ingestion_job_terminal job_id=%s status=%s", job_id, job.status)
            return

        try:
            filename, extension = IngestionService.validate_filename(job.source_name)
            job.status = "PROCESSING"
            job.started_at = job.started_at or datetime.now(UTC)
            job.attempt_count += 1
            job.error_message = None
            self.db.commit()
            logger.info("ingestion_job_started job_id=%s", job_id)

            staged = self.staging.load(job_id, extension)
            service = IngestionService(self.db, self.settings)
            # Keep the authoritative staged file intact until the database outcome is known,
            # so a worker-loss redelivery can still process a PROCESSING job.
            with staged.path.open("rb") as source:
                processing_copy = service.storage.stage(source)
            response = service.ingest_staged(
                processing_copy, filename, extension, commit=False
            )
            job = self.db.get(IngestionJob, job_id)
            if job is None:
                raise RuntimeError("Ingestion job disappeared during processing")
            job.document_id = response.document_id
            job.status = "COMPLETED"
            job.progress_current = job.progress_total
            job.completed_at = datetime.now(UTC)
            job.error_message = None
            self.db.commit()
            logger.info("ingestion_job_completed job_id=%s", job_id)
        except IngestionError as exc:
            self.db.rollback()
            self._mark_failed(job_id, exc.detail)
        except Exception:
            self.db.rollback()
            if response is not None:
                IngestionService(self.db, self.settings).storage.remove_document(
                    response.document_id
                )
            self._mark_failed(job_id, "Document ingestion failed")
            logger.error("ingestion_job_failed job_id=%s", job_id)
        finally:
            self.staging.remove(job_id)

    def _mark_failed(self, job_id: UUID, message: str) -> None:
        job = self.db.get(IngestionJob, job_id)
        if job is None or job.status in TERMINAL_STATUSES:
            return
        job.status = "FAILED"
        job.completed_at = datetime.now(UTC)
        job.error_message = message
        self.db.commit()
        logger.warning("ingestion_job_failed job_id=%s", job_id)


@celery_app.task(name="app.worker.tasks.process_ingestion_job")
def process_ingestion_job(job_id: str) -> None:
    with SessionLocal() as session:
        IngestionJobProcessor(session, settings).process(UUID(job_id))
