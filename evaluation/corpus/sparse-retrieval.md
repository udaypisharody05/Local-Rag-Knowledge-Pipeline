# Sparse retrieval

BM25 supplies keyword retrieval for exact terms and technical identifiers. Its deterministic tokenizer preserves identifiers such as `document_id`, `nomic-embed-text`, and `/api/embed`. Zero-evidence BM25 candidates are excluded.
