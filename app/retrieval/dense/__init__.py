"""Dense vector stores."""

from app.retrieval.dense.faiss_store import FaissStore, FaissStoreError

__all__ = ["FaissStore", "FaissStoreError"]
