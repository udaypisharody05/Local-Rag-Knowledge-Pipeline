"""Phase 7 unit tests that do not require PostgreSQL or Redis."""

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

from app.ingestion.storage import JobStagingStorage, UploadTooLargeError
from app.worker.celery_app import celery_app


def test_celery_accepts_json_only_and_uses_safe_delivery_settings() -> None:
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.accept_content == ["json"]
    assert celery_app.conf.result_serializer == "json"
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_job_staging_uses_generated_contained_path_and_can_reload(tmp_path: Path) -> None:
    job_id = uuid4()
    storage = JobStagingStorage(tmp_path / "staging", max_size_bytes=100)
    staged = storage.stage(BytesIO(b"safe data"), job_id, "txt")
    assert staged.path == (tmp_path / "staging" / str(job_id) / "original.txt").resolve()
    assert storage.load(job_id, "txt") == staged
    storage.remove(job_id)
    assert not staged.path.parent.exists()


def test_job_staging_removes_partial_oversized_upload(tmp_path: Path) -> None:
    job_id = uuid4()
    storage = JobStagingStorage(tmp_path / "staging", max_size_bytes=3)
    with pytest.raises(UploadTooLargeError):
        storage.stage(BytesIO(b"too large"), job_id, "txt")
    assert not (tmp_path / "staging" / str(job_id)).exists()
