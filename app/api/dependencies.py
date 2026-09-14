"""Reusable FastAPI dependencies."""

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.embeddings import EmbeddingProvider, OllamaEmbeddingProvider


def get_db() -> Generator[Session, None, None]:
    """Provide one SQLAlchemy session per request."""
    with SessionLocal() as session:
        yield session


def get_embedding_provider() -> Generator[EmbeddingProvider, None, None]:
    """Provide a short-lived local Ollama client for retrieval operations."""
    with OllamaEmbeddingProvider(
        base_url=str(settings.ollama_base_url),
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        timeout_seconds=settings.ollama_timeout_seconds,
    ) as provider:
        yield provider
