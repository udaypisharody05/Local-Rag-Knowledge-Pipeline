# Logical deletion

Deleting a document immediately marks it deleted in PostgreSQL. Existing immutable FAISS and BM25 snapshots are not changed in place; retrieval hydrates candidates through PostgreSQL and filters deleted content. An explicit rebuild creates a new snapshot without deleted chunks.
