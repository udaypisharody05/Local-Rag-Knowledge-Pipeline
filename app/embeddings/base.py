"""Minimal embedding provider contract."""

from typing import Protocol


class EmbeddingError(RuntimeError):
    """A sanitized embedding provider failure."""


class EmbeddingProvider(Protocol):
    model: str

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
