"""Reusable FastAPI dependencies."""

from collections.abc import Callable, Generator

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from app.generation import GenerationProvider, OllamaGenerationProvider


def get_db() -> Generator[Session, None, None]:
    """Provide one SQLAlchemy session per request."""
    with SessionLocal() as session:
        yield session


def get_ingestion_enqueuer() -> Callable[[str], object]:
    """Return the queue boundary so API tests never require a live broker."""
    from app.worker.tasks import process_ingestion_job

    return process_ingestion_job.delay


def get_embedding_provider() -> Generator[EmbeddingProvider, None, None]:
    """Provide a short-lived local Ollama client for retrieval operations."""
    with OllamaEmbeddingProvider(
        base_url=str(settings.ollama_base_url),
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        timeout_seconds=settings.ollama_timeout_seconds,
    ) as provider:
        yield provider


def get_generation_provider() -> Generator[GenerationProvider, None, None]:
    """Provide a short-lived local Ollama generation client."""
    with OllamaGenerationProvider(
        base_url=str(settings.ollama_base_url),
        model=settings.generation_model,
        temperature=settings.generation_temperature,
        timeout_seconds=settings.generation_timeout_seconds,
    ) as provider:
        yield provider
