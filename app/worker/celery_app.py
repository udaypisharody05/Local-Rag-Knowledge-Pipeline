"""Celery application configured for JSON-only task messages."""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "local_rag",
    broker=settings.celery_broker_url,
    include=["app.worker.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_ignore_result=True,
    task_acks_late=settings.celery_task_acks_late,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
)
