# Local RAG Knowledge Pipeline

A fully local Retrieval-Augmented Generation system built with FastAPI, PostgreSQL, Ollama, FAISS, BM25, Celery, and Redis. It ingests local documents, builds immutable hybrid-retrieval snapshots, generates grounded answers with application-verified citations, and keeps document/job state durable in PostgreSQL.

## What it does

- Ingests TXT, Markdown, and text-based PDF documents synchronously or asynchronously.
- Parses and recursively chunks content while preserving page, heading, and offset metadata.
- Generates local embeddings through Ollama.
- Searches normalized vectors with FAISS `IndexFlatIP` and keywords with BM25.
- Combines dense and sparse rankings using Reciprocal Rank Fusion (RRF).
- Generates bounded-context answers with a local Ollama model.
- Returns deterministic citations whose metadata is verified against PostgreSQL.
- Streams answer tokens and final citations through Server-Sent Events (SSE).
- Tracks Celery ingestion jobs durably in PostgreSQL with Redis used only as broker.
- Logically deletes documents while immediately filtering stale snapshot candidates.
- Evaluates retrieval, grounding, citations, refusals, and exclusion behavior reproducibly.

## Architecture

```text
Documents
   |
   v
Ingestion ---------> Sync API
   |
   +---------------> Redis broker ---> Celery worker
   |
   v
PostgreSQL (source of truth)
   |
   +--------------------+
   |                    |
   v                    v
FAISS dense         BM25 sparse
   |                    |
   +---------+----------+
             |
             v
             RRF
             |
             v
      Hybrid retrieval
             |
             v
      Context selection
             |
             v
 Prompt-injection mitigation
             |
             v
      Ollama generation
             |
       +-----+------+
       |            |
    /query    /query/stream
       |            |
 answer +       SSE tokens +
 citations       citations
```

## Project status

| Capability | Status |
|---|---|
| API-key authentication, health, readiness | Implemented |
| TXT / Markdown / PDF ingestion | Implemented |
| PostgreSQL document and chunk persistence | Implemented |
| Celery / Redis asynchronous ingestion | Implemented |
| Ollama embeddings and generation | Implemented |
| FAISS dense + BM25 sparse retrieval | Implemented |
| RRF hybrid search | Implemented |
| Immutable versioned snapshots | Implemented |
| Grounded answers and verified citations | Implemented |
| SSE answer streaming | Implemented |
| Logical deletion and stale-index filtering | Implemented |
| Deterministic evaluation tooling | Implemented |

## Key design decisions

- **PostgreSQL is authoritative.** Documents, chunks, logical deletion, snapshot lifecycle, and async-job state live in PostgreSQL.
- **Indexes are derived.** FAISS and BM25 snapshots are rebuildable from active PostgreSQL chunks.
- **Cosine similarity is exact.** L2-normalized vectors plus `IndexFlatIP` make inner product equivalent to cosine similarity.
- **RRF combines ranks.** Dense and sparse raw scores are exposed but never added because their scales differ.
- **Snapshots are immutable.** Every rebuild publishes a new validated version; ingestion and deletion never mutate an existing snapshot.
- **Rebuilds are explicit.** New or deleted content reaches the next snapshot only through `POST /retrieval/index/rebuild`.
- **Redis is broker-only.** Celery messages contain a JSON-safe job UUID; PostgreSQL stores job status.
- **Citations are application-owned.** Document IDs, chunk IDs, filenames, pages, and headings come from PostgreSQL, not model output.
- **Context is untrusted.** Obvious prompt-injection lines are sanitized and the system prompt treats retrieved text as evidence, never instructions.
- **Local consistency is intentional.** Docker runs one Uvicorn process and one single-concurrency Celery worker with prefetch one.

## API summary

All application data and operational endpoints except `/health` require `X-API-Key`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Public liveness check |
| GET | `/ready` | Protected PostgreSQL readiness |
| POST | `/documents/upload` | Synchronous document ingestion |
| POST | `/documents/upload/async` | Stage and enqueue asynchronous ingestion |
| GET | `/documents` | List active documents |
| GET | `/documents/{id}` | Read active document metadata |
| DELETE | `/documents/{id}` | Idempotent logical deletion |
| GET | `/ingestion/jobs/{id}` | Read durable ingestion-job state |
| POST | `/retrieval/index/rebuild` | Build and activate a new immutable snapshot |
| GET | `/retrieval/status` | Inspect the loaded snapshot |
| POST | `/search/dense` | FAISS semantic search |
| POST | `/search/sparse` | BM25 keyword search |
| POST | `/search/hybrid` | RRF hybrid search |
| POST | `/query` | Grounded non-streaming answer |
| POST | `/query/stream` | Grounded SSE answer stream |

## Quick start

Requirements:

- Docker with Docker Compose
- Ollama running on the host
- `nomic-embed-text` and `llama3.2:3b` (or compatible configured models)

Create local configuration without committing it:

```powershell
Copy-Item .env.example .env
```

Set secure `API_KEY` and `POSTGRES_PASSWORD` values in `.env`, then start the local stack:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:3b
docker compose up --build -d
```

Compose runs `api`, `postgres`, `redis`, and `worker`. PostgreSQL, Redis, source files, and retrieval snapshots use persistent volumes. The API applies Alembic migrations before Uvicorn starts.

Docker-backed PostgreSQL, Redis, Celery, shared-volume, and Ollama connectivity checks are deployment validation. They are not fabricated when those services are unavailable.

## Core workflows

Synchronous ingestion returns after parsing, chunking, and persistence. Asynchronous ingestion safely stages the file, commits a `QUEUED` PostgreSQL job, and publishes only the job UUID. The worker advances it through `PROCESSING` to `COMPLETED` or `FAILED` and removes staging data after handled outcomes.

Exact duplicate active content returns `409`. After the original is logically deleted, identical content may be ingested as a new active document.

Deletion sets `status=DELETED` and `deleted_at` but retains metadata, chunks, source files, ingestion history, and historical snapshots. Current retrieval candidates are always hydrated through PostgreSQL, so deleted chunks cannot reach search results, prompts, or citations even before rebuilding.

An explicit rebuild embeds all active chunks, builds matching FAISS and BM25 mappings, validates and atomically publishes the version, deprecates the prior database version, and swaps the in-memory snapshot.

`POST /query` performs one hybrid retrieval, selects bounded context, sanitizes obvious injected instructions, assigns deterministic source labels, calls Ollama, and validates model citations. A substantive uncited answer gets one repair attempt and then fails closed. `/query/stream` uses the same preparation path; it emits `start`, `token`, `citations`, `metadata`, and `done` events, or a sanitized `error` event.

## Configuration

Configuration is environment-driven through `.env`. Important settings include:

- `API_KEY`, `DATABASE_URL`, `POSTGRES_PASSWORD`
- `MAX_UPLOAD_SIZE_MB`, `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`, `GENERATION_MODEL`
- dense, sparse, hybrid, and context-size limits
- `CELERY_BROKER_URL`, late acknowledgement, and worker prefetch
- document, staging, index, and snapshot storage roots

See [.env.example](.env.example) for the complete non-secret template.

## Evaluation

The [`evaluation`](evaluation/README.md) package includes:

- an 18-case human-readable dataset;
- a compact reproducible corpus;
- live `/search/hybrid` and `/query` evaluation;
- Hit@K, MRR, Recall@K, exclusion, citation, expected-term, and refusal checks;
- optional request-latency reporting;
- a console summary and ignored machine-readable JSON report.

The evaluator uses `RAG_API_BASE_URL` and `RAG_API_KEY`; credentials are never stored in the dataset. Evaluation metrics are deterministic structural checks, not proof of factual correctness and not an LLM-as-a-judge.

## Testing

```powershell
pytest -v
```

Unit tests do not require Redis or Ollama. PostgreSQL integration tests exercise migrations, ingestion, async processing logic, snapshot publication, stale-content filtering, generation, streaming, and deletion when the configured database is available.

## Security and data safety

- Constant-time API-key comparison and protected operational endpoints.
- Streaming upload limits, extension/MIME allowlists, PDF signature validation, and server-generated storage paths.
- JSON-only Celery serialization; no pickle-based BM25 persistence.
- Sanitized Ollama, Redis, retrieval, and queue errors.
- No document contents, prompts, API keys, or database URLs in application lifecycle logs.
- Current PostgreSQL state gates every snapshot candidate before retrieval or generation.
- Structured citation metadata cannot be invented by the LLM.
- Logical deletion preserves auditability; physical deletion is intentionally separate.

## Limitations and future work

This v1.0 release intentionally does not include:

- repository ingestion;
- CSV/JSON ingestion;
- OCR for scanned PDFs;
- reranking;
- automatic/coalesced snapshot rebuilding;
- physical garbage collection for deleted source files;
- restore/undelete workflows;
- larger domain-specific evaluation datasets;
- multi-user authorization or distributed deployment;
- a frontend or Kubernetes manifests.

See [PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) for the end-to-end technical summary and resume-ready bullets.
