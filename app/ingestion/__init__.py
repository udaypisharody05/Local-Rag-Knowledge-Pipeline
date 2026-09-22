"""Shared synchronous and asynchronous document ingestion core."""

from app.ingestion.service import IngestionError, IngestionService

__all__ = ["IngestionError", "IngestionService"]
