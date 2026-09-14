"""Local embedding providers."""

from app.embeddings.base import EmbeddingError, EmbeddingProvider
from app.embeddings.ollama import OllamaEmbeddingProvider

__all__ = ["EmbeddingError", "EmbeddingProvider", "OllamaEmbeddingProvider"]
