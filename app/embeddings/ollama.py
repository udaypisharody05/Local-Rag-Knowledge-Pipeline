"""Direct local Ollama embedding API client."""

import logging
import math
from collections.abc import Sequence

import httpx

from app.embeddings.base import EmbeddingError

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        batch_size: int,
        timeout_seconds: float,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds
        )

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingError("Embedding input cannot be empty")
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise EmbeddingError("Embedding input cannot be empty")

        vectors: list[list[float]] = []
        expected_dimension: int | None = None
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batch_vectors = self._embed_batch(batch)
            for vector in batch_vectors:
                if expected_dimension is None:
                    expected_dimension = len(vector)
                elif len(vector) != expected_dimension:
                    raise EmbeddingError("Ollama returned inconsistent embedding dimensions")
            vectors.extend(batch_vectors)
        return vectors

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = self._client.post(
                "/api/embed", json={"model": self.model, "input": list(texts)}
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException:
            logger.warning("ollama_embedding_timeout model=%s input_count=%d", self.model, len(texts))
            raise EmbeddingError("Local embedding service timed out") from None
        except (httpx.HTTPError, ValueError):
            logger.exception(
                "ollama_embedding_request_failed model=%s input_count=%d",
                self.model,
                len(texts),
            )
            raise EmbeddingError("Local embedding service is unavailable") from None

        raw_vectors = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(raw_vectors, list) or len(raw_vectors) != len(texts):
            raise EmbeddingError("Ollama returned an unexpected embedding count")

        vectors: list[list[float]] = []
        dimension: int | None = None
        for raw_vector in raw_vectors:
            if not isinstance(raw_vector, list) or not raw_vector:
                raise EmbeddingError("Ollama returned an invalid embedding vector")
            try:
                vector = [float(value) for value in raw_vector]
            except (TypeError, ValueError):
                raise EmbeddingError("Ollama returned an invalid embedding vector") from None
            if any(not math.isfinite(value) for value in vector):
                raise EmbeddingError("Ollama returned an invalid embedding vector")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise EmbeddingError("Ollama returned inconsistent embedding dimensions")
            vectors.append(vector)
        return vectors

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaEmbeddingProvider":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
