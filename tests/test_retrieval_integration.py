"""PostgreSQL-backed snapshot lifecycle and dense-search tests."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_embedding_provider
from app.core.config import settings
from app.db.session import engine
from app.embeddings import EmbeddingError
from app.main import app
from app.models import Document, DocumentChunk, RetrievalIndexVersion
from app.retrieval import snapshot_manager
from app.retrieval.dense import FaissStore
from app.retrieval.service import RetrievalService, load_active_snapshot
from app.retrieval.snapshot import DenseSnapshot, SnapshotError


@dataclass
class DeterministicEmbeddingProvider:
    model: str = "deterministic-test-model"

    @staticmethod
    def _vector(value: str) -> list[float]:
        lowered = value.lower()
        if "apple" in lowered or "fruit" in lowered:
            return [1.0, 0.0]
        if "ocean" in lowered or "water" in lowered:
            return [0.0, 1.0]
        return [0.5, 0.5]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


class FailingEmbeddingProvider(DeterministicEmbeddingProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("Deterministic embedding failure")


@pytest.fixture
def retrieval_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    try:
        connection = engine.connect()
        outer_transaction = connection.begin()
        connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        pytest.skip(f"PostgreSQL integration database unavailable: {type(exc).__name__}")

    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    session.execute(
        update(Document).values(status="DELETED", deleted_at=datetime.now(UTC))
    )
    session.execute(
        update(RetrievalIndexVersion)
        .where(RetrievalIndexVersion.status == "ACTIVATED")
        .values(status="DEPRECATED")
    )
    session.commit()

    previous_snapshot = snapshot_manager.get()
    snapshot_manager.clear()
    provider = DeterministicEmbeddingProvider()
    root = tmp_path / "indexes" / "versions"
    monkeypatch.setattr(settings, "index_storage_root", root)

    def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_embedding_provider] = lambda: provider
    try:
        yield session, provider, root
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        snapshot_manager.clear()
        if previous_snapshot is not None:
            snapshot_manager.replace(previous_snapshot)
        session.close()
        if outer_transaction.is_active:
            outer_transaction.rollback()
        connection.close()


def _add_document(session: Session, source_name: str, texts: list[str]) -> Document:
    document = Document(
        source_name=source_name,
        source_type="txt",
        original_filename=source_name,
        mime_type="text/plain",
        size_bytes=sum(len(value) for value in texts),
        content_hash=uuid4().hex,
        status="INDEXED",
        parser_version="txt-v1",
        chunking_config_hash="test-chunking-hash",
        indexed_at=datetime.now(UTC),
    )
    document.chunks = [
        DocumentChunk(
            text_content=value,
            chunk_index=index,
            content_hash=uuid4().hex,
            chunk_metadata={"source_filename": source_name},
        )
        for index, value in enumerate(texts)
    ]
    session.add(document)
    session.commit()
    return document


def _rebuild(client: TestClient, valid_api_key: str):
    return client.post(
        "/retrieval/index/rebuild", headers={"X-API-Key": valid_api_key}
    )


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/retrieval/index/rebuild", {}),
        ("post", "/search/dense", {"json": {"query": "query"}}),
        ("get", "/retrieval/status", {}),
    ],
)
def test_retrieval_endpoints_require_authentication(
    retrieval_environment,
    client: TestClient,
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 401


def test_rebuild_activates_snapshot_and_creates_version_row(
    retrieval_environment, client: TestClient, valid_api_key: str
) -> None:
    session, _, root = retrieval_environment
    _add_document(session, "knowledge.txt", ["apple fruit", "ocean water"])
    response = _rebuild(client, valid_api_key)
    assert response.status_code == 200
    body = response.json()
    row = session.scalar(
        select(RetrievalIndexVersion).where(
            RetrievalIndexVersion.version == body["snapshot_version"]
        )
    )
    assert row is not None and row.status == "ACTIVATED"
    assert body["chunk_count"] == 2
    assert body["embedding_dimension"] == 2
    assert (root / f"{row.version:06d}" / "faiss.index").is_file()


def test_second_rebuild_deprecates_previous_snapshot(
    retrieval_environment, client: TestClient, valid_api_key: str
) -> None:
    session, _, _ = retrieval_environment
    _add_document(session, "knowledge.txt", ["apple fruit"])
    first = _rebuild(client, valid_api_key).json()
    second = _rebuild(client, valid_api_key).json()
    rows = session.scalars(
        select(RetrievalIndexVersion).where(
            RetrievalIndexVersion.version.in_(
                [first["snapshot_version"], second["snapshot_version"]]
            )
        )
    ).all()
    statuses = {row.version: row.status for row in rows}
    assert statuses[first["snapshot_version"]] == "DEPRECATED"
    assert statuses[second["snapshot_version"]] == "ACTIVATED"
    assert snapshot_manager.get().version == second["snapshot_version"]


def test_failed_rebuild_preserves_active_snapshot(
    retrieval_environment, client: TestClient, valid_api_key: str
) -> None:
    session, _, root = retrieval_environment
    _add_document(session, "knowledge.txt", ["apple fruit"])
    active_version = _rebuild(client, valid_api_key).json()["snapshot_version"]
    app.dependency_overrides[get_embedding_provider] = lambda: FailingEmbeddingProvider()
    failed = _rebuild(client, valid_api_key)
    assert failed.status_code == 503
    assert snapshot_manager.get().version == active_version
    rows = session.scalars(
        select(RetrievalIndexVersion).order_by(RetrievalIndexVersion.version.desc())
    ).all()
    assert rows[0].status == "FAILED"
    assert not (root / f"{rows[0].version:06d}").exists()
    assert session.scalar(
        select(RetrievalIndexVersion).where(
            RetrievalIndexVersion.version == active_version
        )
    ).status == "ACTIVATED"


def test_dense_search_returns_ranked_postgresql_metadata(
    retrieval_environment, client: TestClient, valid_api_key: str
) -> None:
    session, _, _ = retrieval_environment
    document = _add_document(session, "facts.txt", ["ocean water", "apple fruit nutrition"])
    _rebuild(client, valid_api_key)
    response = client.post(
        "/search/dense",
        headers={"X-API-Key": valid_api_key},
        json={"query": "Which fruit is an apple?", "k": 1},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["rank"] == 1
    assert result["text"] == "apple fruit nutrition"
    assert result["document_id"] == str(document.id)
    assert result["source_name"] == "facts.txt"


def test_dense_search_k_limit_is_enforced(
    retrieval_environment, client: TestClient, valid_api_key: str
) -> None:
    response = client.post(
        "/search/dense",
        headers={"X-API-Key": valid_api_key},
        json={"query": "query", "k": settings.dense_max_k + 1},
    )
    assert response.status_code == 422


def test_dense_search_skips_content_deleted_after_snapshot(
    retrieval_environment, client: TestClient, valid_api_key: str
) -> None:
    session, _, _ = retrieval_environment
    document = _add_document(session, "knowledge.txt", ["apple fruit"])
    _rebuild(client, valid_api_key)
    document.status = "DELETED"
    document.deleted_at = datetime.now(UTC)
    session.commit()
    response = client.post(
        "/search/dense",
        headers={"X-API-Key": valid_api_key},
        json={"query": "apple", "k": 1},
    )
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_dense_search_without_snapshot_fails_cleanly(
    retrieval_environment, client: TestClient, valid_api_key: str
) -> None:
    snapshot_manager.clear()
    response = client.post(
        "/search/dense",
        headers={"X-API-Key": valid_api_key},
        json={"query": "query"},
    )
    assert response.status_code == 503


def test_persisted_active_snapshot_loads_after_restart(
    retrieval_environment, client: TestClient, valid_api_key: str
) -> None:
    session, _, root = retrieval_environment
    _add_document(session, "knowledge.txt", ["apple fruit"])
    version = _rebuild(client, valid_api_key).json()["snapshot_version"]
    snapshot_manager.clear()
    loaded = load_active_snapshot(session, root, snapshot_manager)
    assert loaded is not None
    assert loaded.version == version
    assert snapshot_manager.get().version == version


def test_rebuild_empty_knowledge_base_fails_cleanly(
    retrieval_environment, client: TestClient, valid_api_key: str
) -> None:
    response = _rebuild(client, valid_api_key)
    assert response.status_code == 409
    assert snapshot_manager.get() is None


def test_database_mapping_validation_rejects_missing_chunk(
    retrieval_environment,
) -> None:
    session, provider, root = retrieval_environment
    snapshot = DenseSnapshot(
        version=1,
        store=FaissStore.build([[1.0, 0.0]]),
        chunk_ids=(uuid4(),),
        embedding_model=provider.model,
        embedding_dimension=2,
    )
    service = RetrievalService(session, provider, root, snapshot_manager)
    with pytest.raises(SnapshotError, match="missing or inactive"):
        service._validate_mapping(snapshot)


def test_retrieval_status_reports_loaded_snapshot(
    retrieval_environment, client: TestClient, valid_api_key: str
) -> None:
    session, _, _ = retrieval_environment
    _add_document(session, "knowledge.txt", ["apple fruit"])
    version = _rebuild(client, valid_api_key).json()["snapshot_version"]
    response = client.get(
        "/retrieval/status", headers={"X-API-Key": valid_api_key}
    )
    assert response.status_code == 200
    assert response.json()["available"] is True
    assert response.json()["snapshot_version"] == version
