"""Dense retrieval API behavior that does not require PostgreSQL."""

from fastapi.testclient import TestClient

from app.core.config import settings
from app.retrieval import snapshot_manager


def test_retrieval_status_is_unavailable_without_snapshot(
    client: TestClient, valid_api_key: str
) -> None:
    previous = snapshot_manager.get()
    snapshot_manager.clear()
    try:
        response = client.get(
            "/retrieval/status", headers={"X-API-Key": valid_api_key}
        )
    finally:
        if previous is not None:
            snapshot_manager.replace(previous)
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["snapshot_version"] is None


def test_dense_search_rejects_k_above_configured_limit(
    client: TestClient, valid_api_key: str
) -> None:
    response = client.post(
        "/search/dense",
        headers={"X-API-Key": valid_api_key},
        json={"query": "query", "k": settings.dense_max_k + 1},
    )
    assert response.status_code == 422


def test_dense_search_is_unavailable_without_snapshot(
    client: TestClient, valid_api_key: str
) -> None:
    previous = snapshot_manager.get()
    snapshot_manager.clear()
    try:
        response = client.post(
            "/search/dense",
            headers={"X-API-Key": valid_api_key},
            json={"query": "query"},
        )
    finally:
        if previous is not None:
            snapshot_manager.replace(previous)
    assert response.status_code == 503
    assert response.json() == {"detail": "No active retrieval index is loaded"}
