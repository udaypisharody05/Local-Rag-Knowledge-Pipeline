# Local RAG Knowledge Pipeline

A self-hosted, local Retrieval-Augmented Generation pipeline. Phase 7 adds durable asynchronous document ingestion with Celery, Redis, and PostgreSQL-backed job state while preserving synchronous upload and the explicit retrieval-snapshot lifecycle.

## Architecture

- FastAPI serves a public liveness endpoint and an API-key-protected readiness endpoint.
- SQLAlchemy 2.x provides typed synchronous PostgreSQL models and request-scoped sessions.
- Alembic is the sole schema authority; application startup never calls `create_all()`.
- PostgreSQL stores documents, chunks, ingestion job state, and future retrieval snapshot versions.
- Redis is only the Celery broker; PostgreSQL remains authoritative for ingestion job state.
- One Celery worker parses, chunks, and persists staged uploads using the same ingestion core as synchronous requests.
- Docker Compose starts API, PostgreSQL, Redis, and one intentionally single-concurrency worker with shared document storage.
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
- Non-streaming grounded answers from a configurable local Ollama generation model
- Deterministic source labels and structured citations verified against PostgreSQL metadata
- Bounded prompt context and explicit handling for an empty usable context
- Authenticated asynchronous upload and persistent job-status endpoints
- Safe UUID-based shared staging storage with cleanup after success or handled failure
- JSON-only Celery messages containing job UUIDs, late acknowledgement, and prefetch of one

## Not implemented yet

OCR, repository/CSV/JSON ingestion, semantic chunking, reranking, automatic/coalesced reindexing, a frontend, and a formal evaluation framework are not implemented.

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

The API container runs `alembic upgrade head` before starting Uvicorn. PostgreSQL data, Redis broker data, source documents/staging files, and FAISS snapshots are retained in Docker volumes. The worker uses the same application image and shared document-storage volume.

## Ingestion configuration

The defaults are a 10 MB upload limit, 1,000-character chunks, and 150-character overlap. Override `MAX_UPLOAD_SIZE_MB`, `CHUNK_SIZE`, and `CHUNK_OVERLAP` in `.env`; overlap must be smaller than chunk size. Only `.txt`, `.md`, `.markdown`, and `.pdf` files are accepted. PDFs must contain extractable text because OCR is intentionally out of scope.

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

## Asynchronous ingestion

`POST /documents/upload` remains synchronous. `POST /documents/upload/async` validates and safely stages the upload, commits a `QUEUED` PostgreSQL job, enqueues only its UUID, and returns `202 Accepted` without waiting for parsing or chunking. The worker transitions jobs through `QUEUED -> PROCESSING -> COMPLETED`, or to `FAILED` with a sanitized error. `GET /ingestion/jobs/{job_id}` reads PostgreSQL directly; Celery result metadata is not application state.

Successfully ingested documents are not added to the active FAISS/BM25 snapshot automatically. Run `POST /retrieval/index/rebuild` explicitly when newly completed documents should become searchable.

Exact PowerShell verification:

```powershell
docker compose up --build -d
docker compose ps
docker compose exec api alembic upgrade head
docker compose exec api pytest -v
$base = "http://localhost:8000"
$headers = @{"X-API-Key"="your-api-key"}
$unique = "phase7-$([guid]::NewGuid())"
Set-Content -Path .\phase7-async.txt -Value "Unique asynchronous knowledge token: $unique"
$snapshotBefore = Invoke-RestMethod -Uri "$base/retrieval/status" -Headers $headers
$accepted = Invoke-RestMethod -Method Post -Uri "$base/documents/upload/async" -Headers $headers -Form @{file=Get-Item .\phase7-async.txt}
$accepted
$job = do {
  Start-Sleep -Milliseconds 250
  $current = Invoke-RestMethod -Uri "$base/ingestion/jobs/$($accepted.job_id)" -Headers $headers
  $current
} until ($current.status -in @("COMPLETED", "FAILED"))
$documentId = $current.document_id
Invoke-RestMethod -Uri "$base/documents/$documentId" -Headers $headers
$snapshotAfterIngestion = Invoke-RestMethod -Uri "$base/retrieval/status" -Headers $headers
$snapshotBefore
$snapshotAfterIngestion
if ($snapshotBefore.snapshot_version -ne $snapshotAfterIngestion.snapshot_version) { throw "Ingestion unexpectedly changed the retrieval snapshot" }
Invoke-RestMethod -Method Post -Uri "$base/retrieval/index/rebuild" -Headers $headers
$body = @{query=$unique; k=5} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$base/search/hybrid" -Headers $headers -ContentType "application/json" -Body $body
docker compose logs worker --tail=100
```

The two status responses should report the same active snapshot before the explicit rebuild. Fast jobs may move directly from `QUEUED` to `COMPLETED` between polls.

Optional decoupling demonstration:

```powershell
docker compose stop worker
$unique2 = "phase7-queued-$([guid]::NewGuid())"
Set-Content -Path .\phase7-queued.txt -Value "Queued worker demonstration: $unique2"
$queued = Invoke-RestMethod -Method Post -Uri "$base/documents/upload/async" -Headers $headers -Form @{file=Get-Item .\phase7-queued.txt}
Invoke-RestMethod -Uri "$base/ingestion/jobs/$($queued.job_id)" -Headers $headers
docker compose start worker
Invoke-RestMethod -Uri "$base/ingestion/jobs/$($queued.job_id)" -Headers $headers
```

Use a new file body for this optional demonstration if the first document has already completed, because exact active-content duplicates are rejected before enqueueing.

## Local Ollama and hybrid retrieval

Install Ollama separately on the host, start it, and pull the configured local embedding and generation models:

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2:3b
ollama list
```

The models are controlled independently by `EMBEDDING_MODEL` and `GENERATION_MODEL`. Docker defaults `OLLAMA_BASE_URL` to `http://host.docker.internal:11434`; native development defaults to `http://localhost:11434`. Change the URL in `.env` for another local networking arrangement. The API still starts and document ingestion remains available when Ollama is offline; generation failures return a sanitized service error and do not affect `/health`.

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

## Grounded query flow

`POST /query` runs the existing hybrid search once, keeps results in rank order, selects at most `GENERATION_MAX_CONTEXT_CHUNKS` within `GENERATION_MAX_CONTEXT_CHARS`, assigns `[SOURCE_1]`, `[SOURCE_2]`, and so on, then sends the delimited context to Ollama. The default generation settings are:

```dotenv
GENERATION_MODEL=llama3.2:3b
GENERATION_TEMPERATURE=0.1
GENERATION_TIMEOUT_SECONDS=120
GENERATION_MAX_CONTEXT_CHUNKS=5
GENERATION_MAX_CONTEXT_CHARS=12000
```

Retrieved documents are treated as untrusted data. Before prompt rendering, a small deterministic sanitizer removes lines matching obvious instruction-injection phrases while preserving surrounding factual text; stored documents and retrieval indexes are unchanged. The system prompt separately tells the model that context is evidence only, to ignore document-borne commands or role changes, use only explicit supplied facts, avoid speculative conclusions, and cite every factual claim with supplied labels. This is lightweight defense-in-depth: prompt-injection risk is mitigated, not eliminated.

Citation labels in the answer are parsed and checked against the application-owned label mapping; filenames, page numbers, document IDs, and chunk IDs always come from PostgreSQL rather than model output. If a substantive answer contains no valid citation, the service makes exactly one repair call using the same sanitized, verified context and asks for a concise cited rewrite without unsupported claims. If that repair remains uncited, the request fails safely instead of returning the factual draft as grounded. Genuine insufficient-context refusals may contain no citations and do not trigger repair.

If retrieval produces no usable context, Ollama is not called and the API returns a static insufficient-context answer with no citations. If context exists but lacks the answer, the model is instructed to refuse concisely. Hallucinations are mitigated, not guaranteed to be eliminated.

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query":"How does the project compare vectors?","k":5}'
```

PowerShell equivalent:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/query `
  -Headers @{"X-API-Key"="your-api-key"} -ContentType "application/json" `
  -Body '{"query":"How does the project compare vectors?","k":5}'
```

Example response shape:

```json
{
  "query": "How does the project compare vectors?",
  "answer": "The project uses normalized vectors with IndexFlatIP [SOURCE_1].",
  "citations": [{
    "source_id": "SOURCE_1",
    "document_id": "verified-document-uuid",
    "chunk_id": "verified-chunk-uuid",
    "source_name": "retrieval.md",
    "source_type": "md",
    "page_number": null,
    "section_title": "Dense retrieval",
    "repo_relative_path": null
  }],
  "retrieval": {
    "snapshot_version": 2,
    "retrieved_chunk_ids": ["verified-chunk-uuid"],
    "context_chunk_ids": ["verified-chunk-uuid"],
    "fusion_strategy": "rrf"
  },
  "model": "llama3.2:3b"
}
```

## Streaming grounded queries

`POST /query` remains the non-streaming endpoint with its single citation-repair attempt. `POST /query/stream` uses the same one-time hybrid retrieval, context limits, source labeling, and prompt-injection sanitization, then streams the local Ollama response using Server-Sent Events (SSE).

Successful event order:

```text
event: start
data: {"query":"...","snapshot_version":2,"model":"llama3.2:3b"}

event: token
data: {"text":"incremental text"}

event: citations
data: {"citations":[...]}

event: metadata
data: {"snapshot_version":2,"retrieved_chunk_ids":[],"context_chunk_ids":[],"fusion_strategy":"rrf","model":"llama3.2:3b"}

event: done
data: {}
```

Tokens are accumulated only for end-of-stream citation validation. Structured citations still come exclusively from the application-owned source mapping. Invalid labels are excluded. Unlike `/query`, streaming cannot retract an already-sent uncited answer, so a substantive response with no valid citation ends with a sanitized `error` event and no `citations`, `metadata`, or successful `done` event. A legitimate insufficient-context refusal completes normally with an empty citation list.

If Ollama fails before or during token output, the stream sends one sanitized `error` event and terminates. Client disconnects stop consumption and close the upstream Ollama stream; no background generation job is retained. Streaming remains fully local through the configured Ollama host.

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query":"How does vector search work?","k":5}'
```

Windows PowerShell using the real curl executable:

```powershell
curl.exe --no-buffer -X POST http://localhost:8000/query/stream `
  -H "X-API-Key: your-api-key" `
  -H "Content-Type: application/json" `
  -d '{"query":"How does vector search work?","k":5}'
```

Manual verification sequence:

```bash
docker compose up --build -d
docker compose exec api alembic upgrade head
docker compose exec api pytest -v
ollama pull nomic-embed-text
ollama pull llama3.2:3b
curl -X POST http://localhost:8000/retrieval/index/rebuild -H "X-API-Key: your-api-key"
curl -X POST http://localhost:8000/search/dense -H "X-API-Key: your-api-key" -H "Content-Type: application/json" -d '{"query":"your question","k":5}'
curl -X POST http://localhost:8000/search/sparse -H "X-API-Key: your-api-key" -H "Content-Type: application/json" -d '{"query":"IndexFlatIP PostgreSQL","k":5}'
curl -X POST http://localhost:8000/search/hybrid -H "X-API-Key: your-api-key" -H "Content-Type: application/json" -d '{"query":"your question","k":5}'
curl -X POST http://localhost:8000/query -H "X-API-Key: your-api-key" -H "Content-Type: application/json" -d '{"query":"your question","k":5}'
curl -X POST http://localhost:8000/query -H "X-API-Key: your-api-key" -H "Content-Type: application/json" -d '{"query":"an unrelated question","k":5}'
```

For each real answer, compare `citations` with the referenced documents and confirm `context_chunk_ids` belongs to the reported snapshot. An unrelated question may take the static no-context path or produce the model-level insufficient-context response because Phase 5 intentionally adds no uncalibrated relevance threshold.

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

Docker-backed PostgreSQL, Redis, and Celery runtime checks remain part of deployment validation.

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

Retrieval snapshot rebuilds remain synchronous and explicit; asynchronous ingestion does not automatically reindex. PDFs are not OCR-processed, and there is no delete/retry endpoint. The local deployment intentionally uses one ingestion worker. Reranking, repository ingestion, CSV/JSON ingestion, a frontend, and a formal evaluation framework are not implemented.

## Next phase

Add automatic/coalesced snapshot rebuilding only as a separately designed later phase; do not couple a full rebuild to every ingestion job.
