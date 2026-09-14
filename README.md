# Local RAG Knowledge Pipeline

A self-hosted knowledge pipeline for a future single-server Retrieval-Augmented Generation system. Phase 3 adds local Ollama embeddings and persistent exact dense retrieval over ingested TXT, Markdown, and text-based PDF documents. Answer generation is **not** implemented yet.

## Phase 1 architecture

- FastAPI serves a public liveness endpoint and an API-key-protected readiness endpoint.
- SQLAlchemy 2.x provides typed synchronous PostgreSQL models and request-scoped sessions.
- Alembic is the sole schema authority; application startup never calls `create_all()`.
- PostgreSQL stores documents, chunks, ingestion job state, and future retrieval snapshot versions.
- Docker Compose starts only the API and PostgreSQL, with persistent database storage and health-based startup ordering.
- Logs are JSON on standard output and never include API-key values or database URLs.

The `/ready` endpoint is intentionally protected because it reports infrastructure readiness. `/health` remains public for container/orchestrator liveness checks.

## Implemented

- Typed settings from environment variables or `.env`
- Constant-time `X-API-Key` validation
- `GET /health` and database-backed `GET /ready`
- Four-table PostgreSQL schema with UUID keys, JSONB chunk metadata, timezone-aware timestamps, and cascading chunk deletion
- Initial Alembic migration and autogeneration configuration
- JSON structured logging
- Python 3.12 container with one Uvicorn worker
- Unit tests plus PostgreSQL migration integration tests
- Authenticated synchronous `POST /documents/upload`
- TXT, Markdown, and text-based PDF extraction with page/heading metadata
- Configurable recursive character chunking and deterministic SHA-256 hashes
- Content-based duplicate detection with `409 Conflict`
- Safe UUID-based source storage and cleanup on failed ingestion
- Authenticated document list and detail endpoints
- Local, configurable Ollama embeddings through the direct `/api/embed` HTTP API
- Exact FAISS `IndexFlatIP` search using L2-normalized vectors for cosine similarity
- Immutable, persistent, versioned retrieval snapshots with PostgreSQL-controlled activation
- Protected snapshot rebuild, dense-search, and retrieval-status endpoints

## Not implemented yet

OCR, repository/CSV/JSON ingestion, semantic chunking, BM25, rank fusion, hybrid retrieval, reranking, grounded LLM answer generation, citations, streaming, asynchronous indexing, Celery, Redis, and evaluation are planned for later phases.

## Prerequisites

- Docker with Docker Compose (recommended), or Python 3.12 and PostgreSQL 14+

## Configure and start with Docker

Create your local environment file and replace both secrets:

```bash
cp .env.example .env
```

On PowerShell, use `Copy-Item .env.example .env`.

Then start the stack:

```bash
docker compose up --build
```

The API container runs `alembic upgrade head` before starting Uvicorn. PostgreSQL data, source documents, and FAISS snapshots are retained in separate Docker volumes.

## Ingestion configuration

The defaults are a 10 MB upload limit, 1,000-character chunks, and 150-character overlap. Override `MAX_UPLOAD_SIZE_MB`, `CHUNK_SIZE`, and `CHUNK_OVERLAP` in `.env`; overlap must be smaller than chunk size. Only `.txt`, `.md`, and `.pdf` files are accepted. PDFs must contain extractable text because OCR is intentionally out of scope.

Upload a document with curl:

```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "X-API-Key: your-api-key" \
  -F "file=@example.pdf"
```

PowerShell equivalent:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/documents/upload `
  -Headers @{"X-API-Key"="your-api-key"} -Form @{file=Get-Item .\example.pdf}
```

List indexed documents without returning chunk text:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/documents
```

## Local Ollama and dense retrieval

Install Ollama separately on the host, start it, and pull the configured local embedding model:

```bash
ollama serve
ollama pull nomic-embed-text
ollama list
```

The model is controlled by `EMBEDDING_MODEL`. Docker defaults `OLLAMA_BASE_URL` to `http://host.docker.internal:11434`; native development defaults to `http://localhost:11434`. Change the URL in `.env` for another local networking arrangement. The API still starts and document ingestion remains available when Ollama is offline.

Build a complete immutable snapshot from all active PostgreSQL chunks:

```bash
curl -X POST http://localhost:8000/retrieval/index/rebuild \
  -H "X-API-Key: your-api-key"
```

Every rebuild creates a new directory under `storage/indexes/versions/` containing `faiss.index`, `faiss_mapping.json`, and `manifest.json`. A temporary snapshot is validated and atomically published before PostgreSQL deprecates the previous active version and activates the new one. Existing active snapshots remain usable if a rebuild fails.

Run exact dense semantic search:

```bash
curl -X POST http://localhost:8000/search/dense \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is retrieval augmented generation?","k":5}'
```

FAISS positions map only to chunk UUIDs; returned text and source metadata are reloaded from PostgreSQL. `DENSE_TOP_K` defaults to 5 and `DENSE_MAX_K` defaults to 20. Check whether a snapshot is loaded with `GET /retrieval/status`.

Manual verification sequence:

```bash
docker compose up --build -d
docker compose exec api alembic upgrade head
docker compose exec api pytest -v
ollama pull nomic-embed-text
curl -X POST http://localhost:8000/retrieval/index/rebuild -H "X-API-Key: your-api-key"
curl -X POST http://localhost:8000/search/dense -H "X-API-Key: your-api-key" -H "Content-Type: application/json" -d '{"query":"your question","k":5}'
```

## Local Python setup

Use a local PostgreSQL URL (usually host `localhost`, rather than Compose host `postgres`) in `.env`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

PowerShell activation is `.\.venv\Scripts\Activate.ps1`.

## Migrations

Apply migrations:

```bash
alembic upgrade head
```

After changing models in a later phase, generate and review a migration:

```bash
alembic revision --autogenerate -m "describe schema change"
```

## Tests

The health and authentication tests do not require a database. Database tests connect to `DATABASE_URL`, apply Alembic migrations, and skip with an explicit reason if PostgreSQL is unavailable. Use a dedicated test database in CI.

```bash
pytest -v
```

Inside the running API container:

```bash
docker compose exec api pytest -v
```

## Health checks

Public liveness:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Protected readiness:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/ready
# {"status":"ready","database":"ok"}
```

Missing or invalid keys return `401`. A database failure returns `503` with a sanitized message.

## Current limitations

Ingestion and index rebuilds are synchronous. PDFs are not OCR-processed, and there is no delete endpoint. Phase 3 provides dense retrieval only: BM25, hybrid fusion, reranking, LLM answer generation, citations, streaming, and asynchronous indexing are not implemented.

## Next phase

Add BM25 and Reciprocal Rank Fusion as a separate hybrid-retrieval phase without changing the immutable dense snapshot contract.
