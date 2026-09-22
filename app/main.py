"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.ingestion_jobs import router as ingestion_jobs_router
from app.api.query import router as query_router
from app.api.retrieval import router as retrieval_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.session import SessionLocal
from app.retrieval import snapshot_manager
from app.retrieval.service import load_active_snapshot

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("application_started env=%s", settings.app_env)
    try:
        with SessionLocal() as session:
            snapshot = load_active_snapshot(
                session, settings.index_storage_root, snapshot_manager
            )
            if snapshot is not None:
                logger.info(
                    "retrieval_snapshot_loaded version=%d chunk_count=%d model=%s",
                    snapshot.version,
                    snapshot.chunk_count,
                    snapshot.embedding_model,
                )
    except Exception:
        snapshot_manager.clear()
        logger.exception("active_retrieval_snapshot_load_failed")
    yield
    logger.info("application_stopped")


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(ingestion_jobs_router)
app.include_router(retrieval_router)
app.include_router(query_router)
