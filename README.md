# Local RAG Knowledge Pipeline

A self-hosted knowledge pipeline for a future single-server Retrieval-Augmented Generation system. Phase 4 combines local semantic and keyword retrieval over ingested TXT, Markdown, and text-based PDF documents. Answer generation is **not** implemented yet.

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
- Deterministic BM25 keyword retrieval with technical-identifier-aware tokenization
- Reciprocal Rank Fusion (RRF) hybrid search over dense and sparse rankings
- One immutable snapshot version containing both FAISS and safely persisted BM25 data
- Protected snapshot rebuild, dense-, sparse-, hybrid-search, and retrieval-status endpoints

## Not implemented yet

OCR, repository/CSV/JSON ingestion, semantic chunking, reranking, grounded LLM answer generation, citations in generated answers, streaming, asynchronous ingestion/indexing, Celery, Redis, and evaluation are planned for later phases.

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

## Local Ollama and hybrid retrieval

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

Every rebuild creates one combined version directory under `storage/indexes/versions/`. It contains `faiss.index`, `faiss_mapping.json`, `bm25_corpus.jsonl`, `bm25_mapping.json`, and `manifest.json`. The BM25 corpus is deterministic JSONL and is reconstructed in memory at load time; Python pickle is never used. A temporary snapshot is validated and atomically published before PostgreSQL deprecates the previous active version and activates the new one. Existing active snapshots remain usable if either index build fails.

Run exact dense semantic search:

```bash
curl -X POST http://localhost:8000/search/dense \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is retrieval augmented generation?","k":5}'
```

FAISS positions map only to chunk UUIDs; returned text and source metadata are reloaded from PostgreSQL. `DENSE_TOP_K` defaults to 5 and `DENSE_MAX_K` defaults to 20. Check whether a snapshot is loaded with `GET /retrieval/status`.

BM25 complements semantic retrieval when exact keywords, API paths, model names, or identifiers matter. The tokenizer lowercases text and preserves compounds such as `nomic-embed-text`, `document_id`, and `/api/embed`; it intentionally does not stem words. Run keyword search with:

```bash
curl -X POST http://localhost:8000/search/sparse \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query":"IndexFlatIP PostgreSQL","k":5}'
```

Hybrid search retrieves independent dense and sparse candidate lists, then combines their 1-based ranks using RRF. It does not add cosine and BM25 scores because those raw scales are incompatible. Raw component scores and ranks remain in the response for evaluation and debugging.

```bash
curl -X POST http://localhost:8000/search/hybrid \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query":"How are vectors searched?","k":5}'
```

PowerShell examples:

```powershell
$headers = @{"X-API-Key"="your-api-key"}
Invoke-RestMethod -Method Post -Uri http://localhost:8000/search/sparse -Headers $headers -ContentType "application/json" -Body '{"query":"IndexFlatIP PostgreSQL","k":5}'
Invoke-RestMethod -Method Post -Uri http://localhost:8000/search/hybrid -Headers $headers -ContentType "application/json" -Body '{"query":"How are vectors searched?","k":5}'
```

Configuration defaults are `SPARSE_TOP_K=5`, `SPARSE_MAX_K=20`, `DENSE_CANDIDATE_K=20`, `SPARSE_CANDIDATE_K=20`, `HYBRID_TOP_K=5`, `HYBRID_MAX_K=20`, and `RRF_K=60`. All three search endpoints report the same active snapshot version.

Manual verification sequence:

```bash
docker compose up --build -d
docker compose exec api alembic upgrade head
docker compose exec api pytest -v
ollama pull nomic-embed-text
curl -X POST http://localhost:8000/retrieval/index/rebuild -H "X-API-Key: your-api-key"
curl -X POST http://localhost:8000/search/dense -H "X-API-Key: your-api-key" -H "Content-Type: application/json" -d '{"query":"your question","k":5}'
curl -X POST http://localhost:8000/search/sparse -H "X-API-Key: your-api-key" -H "Content-Type: application/json" -d '{"query":"IndexFlatIP PostgreSQL","k":5}'
curl -X POST http://localhost:8000/search/hybrid -H "X-API-Key: your-api-key" -H "Content-Type: application/json" -d '{"query":"your question","k":5}'
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

Ingestion and full snapshot rebuilds are synchronous. PDFs are not OCR-processed, and there is no delete endpoint. Reranking, generated answers, citations in generated answers, streaming, and asynchronous ingestion/indexing are not implemented.

## Next phase

Evaluate dense, sparse, and hybrid retrieval before selecting and implementing a reranking approach in a later phase.
