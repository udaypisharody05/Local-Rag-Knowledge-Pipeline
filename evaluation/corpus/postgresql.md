# PostgreSQL source of truth

PostgreSQL stores document metadata, chunks, ingestion jobs, and retrieval snapshot lifecycle records. Redis is not authoritative for application job state. FAISS and BM25 are derived indexes that can be rebuilt from active PostgreSQL chunks.
