# Dense retrieval

FAISS performs semantic vector search. The system uses `IndexFlatIP` with L2-normalized embeddings, so inner product corresponds to cosine similarity. Snapshot positions map to chunk UUIDs whose current metadata is loaded from PostgreSQL.
