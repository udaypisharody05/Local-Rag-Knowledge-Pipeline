"""Grounded query API behavior independent of PostgreSQL and live Ollama."""

from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.api.dependencies import get_db, get_embedding_provider, get_generation_provider
from app.core.config import settings
from app.generation.service import GroundedGenerationService, GroundedQueryResult
from app.main import app


@dataclass
class FakeProvider:
    model: str = "fake-model"


def test_query_requires_authentication(client: TestClient) -> None:
    assert client.post("/query", json={"query": "question"}).status_code == 401


def test_query_k_limit_is_enforced(client: TestClient, valid_api_key: str) -> None:
    response = client.post(
        "/query",
        headers={"X-API-Key": valid_api_key},
        json={"query": "question", "k": settings.hybrid_max_k + 1},
    )
    assert response.status_code == 422


def test_query_returns_structured_generation_response(
    client: TestClient, valid_api_key: str, monkeypatch
) -> None:
    def fake_answer(self, query: str, k: int) -> GroundedQueryResult:
        return GroundedQueryResult(
            query=query,
            answer="Not enough verified detail.",
            citations=[],
            snapshot_version=4,
            retrieved_chunk_ids=[],
            context_chunk_ids=[],
            model="fake-model",
        )

    monkeypatch.setattr(GroundedGenerationService, "answer", fake_answer)
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_embedding_provider] = lambda: FakeProvider()
    app.dependency_overrides[get_generation_provider] = lambda: FakeProvider()
    try:
        response = client.post(
            "/query",
            headers={"X-API-Key": valid_api_key},
            json={"query": "question"},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_generation_provider, None)
    assert response.status_code == 200
    assert response.json() == {
        "query": "question",
        "answer": "Not enough verified detail.",
        "citations": [],
        "retrieval": {
            "snapshot_version": 4,
            "retrieved_chunk_ids": [],
            "context_chunk_ids": [],
            "fusion_strategy": "rrf",
        },
        "model": "fake-model",
    }
