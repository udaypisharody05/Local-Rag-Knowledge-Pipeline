# Project Summary

## End-to-end system

The Local RAG Knowledge Pipeline is a self-hosted knowledge system with direct, inspectable components:

```text
upload
  -> allowlist and size validation
  -> TXT / Markdown / PDF parsing
  -> recursive chunking with metadata
  -> PostgreSQL documents and chunks
  -> Ollama embeddings
  -> immutable FAISS + BM25 snapshot
  -> Reciprocal Rank Fusion
  -> bounded context selection and injection sanitization
  -> local Ollama generation
  -> application-verified citations
  -> JSON response or SSE token stream
```

Uploads may run synchronously or through Redis and a single Celery worker. Asynchronous job state remains durable in PostgreSQL. Snapshot artifacts are immutable and rebuildable; PostgreSQL filters every candidate against current logical-deletion state. The included evaluation tool calls the real hybrid-search and query endpoints and calculates deterministic retrieval, citation, refusal, exclusion, and latency metrics.

## Engineering highlights

- PostgreSQL-backed data and lifecycle state with Alembic migrations and active-content uniqueness.
- Safe persistent source/staging storage with generated UUID paths and cleanup on handled failure.
- Exact normalized FAISS retrieval, deterministic BM25 tokenization, and rank-based RRF fusion.
- Atomic versioned snapshot publication with dense/sparse mapping validation and restart loading.
- Grounded generation with bounded context, injection mitigation, citation repair, and application-owned metadata.
- SSE streaming with sanitized failures and disconnect-aware upstream cancellation.
- Celery/Redis decoupling with JSON-only task payloads, late acknowledgement, and terminal-state idempotency.
- Logical deletion that immediately excludes stale indexed chunks while preserving audit history.

## Resume-ready bullets

- Built a fully local RAG pipeline using FastAPI, PostgreSQL, Ollama, FAISS, and BM25 for hybrid semantic and keyword retrieval.
- Implemented Reciprocal Rank Fusion, immutable versioned retrieval snapshots, grounded generation, prompt-injection mitigation, and application-verified citations.
- Added SSE token streaming and Celery/Redis asynchronous ingestion with persistent PostgreSQL job tracking, safe staging, and idempotent task handling.
- Developed reproducible retrieval/grounding evaluation tooling with Hit@K, MRR, Recall@K, citation, refusal, exclusion, and latency metrics plus automated unit and PostgreSQL integration tests.
