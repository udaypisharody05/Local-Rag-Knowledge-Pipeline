# Local RAG Knowledge Pipeline

A self-hosted backend foundation for a future single-server Retrieval-Augmented Generation system. Phase 1 establishes the API, PostgreSQL schema, migrations, authentication, observability, container workflow, and tests. It does **not** ingest or retrieve documents yet.

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

## Not implemented yet

Document parsing/ingestion, semantic chunking, Ollama embeddings, FAISS, BM25, rank fusion, reranking, grounded generation, citations, streaming, Celery, Redis, and evaluation are planned for later phases.

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

The API container runs `alembic upgrade head` before starting Uvicorn. PostgreSQL data is retained in the `postgres_data` volume.

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

## Next phase

Build document ingestion and parsing on this schema, keeping retrieval and generation concerns out of the ingestion path until their dedicated phases.
