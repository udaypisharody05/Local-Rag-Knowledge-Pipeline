"""Exact cosine search using normalized vectors and FAISS IndexFlatIP."""

from pathlib import Path

import faiss
import numpy as np

from app.retrieval.dense.base import DenseMatch


class FaissStoreError(ValueError):
    pass


class FaissStore:
    index_type = "IndexFlatIP"
    similarity_metric = "cosine"

    def __init__(self, index: faiss.Index) -> None:
        self.index = index

    @classmethod
    def build(cls, vectors: list[list[float]]) -> "FaissStore":
        matrix = cls._matrix(vectors)
        cls._normalize(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index)

    @property
    def count(self) -> int:
        return int(self.index.ntotal)

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    def search(self, query_vector: list[float], k: int) -> list[DenseMatch]:
        if len(query_vector) != self.dimension:
            raise FaissStoreError("Query embedding dimension does not match the active snapshot")
        matrix = self._matrix([query_vector])
        self._normalize(matrix)
        scores, positions = self.index.search(matrix, min(k, self.count))
        return [
            DenseMatch(position=int(position), score=float(score))
            for position, score in zip(positions[0], scores[0], strict=True)
            if position >= 0
        ]

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "FaissStore":
        try:
            return cls(faiss.read_index(str(path)))
        except RuntimeError as exc:
            raise FaissStoreError("FAISS index could not be loaded") from exc

    @staticmethod
    def _matrix(vectors: list[list[float]]) -> np.ndarray:
        if not vectors or not vectors[0]:
            raise FaissStoreError("At least one non-empty embedding is required")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise FaissStoreError("Embedding dimensions are inconsistent")
        matrix = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise FaissStoreError("Embeddings must contain only finite values")
        return np.ascontiguousarray(matrix)

    @staticmethod
    def _normalize(matrix: np.ndarray) -> None:
        if np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise FaissStoreError("Zero-length embeddings cannot be indexed")
        faiss.normalize_L2(matrix)
