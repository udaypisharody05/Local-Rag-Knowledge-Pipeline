# Asynchronous ingestion

`POST /documents/upload/async` safely stages a file, stores a `QUEUED` job in PostgreSQL, and sends only the job UUID through Redis to Celery. The worker advances the persistent state through `PROCESSING` to `COMPLETED` or `FAILED`. Redis is the broker; PostgreSQL is authoritative.
