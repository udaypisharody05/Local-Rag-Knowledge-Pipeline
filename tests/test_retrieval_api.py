"""Dense retrieval API behavior that does not require PostgreSQL."""

from fastapi.testclient import TestClient

from app.core.config import settings
from app.retrieval import snapshot_manager


def test_phase4_search_endpoints_require_authentication(client: TestClient) -> None:
    for path in ("/search/sparse", "/search/hybrid"):
        assert client.post(path, json={"query": "query"}).status_code == 401


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


def test_phase4_status_reports_both_indexes_unavailable(
    client: TestClient, valid_api_key: str
) -> None:
    previous = snapshot_manager.get()
    snapshot_manager.clear()
    try:
        body = client.get(
            "/retrieval/status", headers={"X-API-Key": valid_api_key}
        ).json()
    finally:
        if previous is not None:
            snapshot_manager.replace(previous)
    assert body["dense_available"] is False
    assert body["sparse_available"] is False
    assert body["fusion_strategy"] is None


def test_phase4_search_limits_are_enforced_without_database(
    client: TestClient, valid_api_key: str
) -> None:
    for path, limit in (
        ("/search/sparse", settings.sparse_max_k),
        ("/search/hybrid", settings.hybrid_max_k),
    ):
        response = client.post(
            path,
            headers={"X-API-Key": valid_api_key},
            json={"query": "query", "k": limit + 1},
        )
        assert response.status_code == 422


def test_phase4_search_is_unavailable_without_snapshot(
    client: TestClient, valid_api_key: str
) -> None:
    previous = snapshot_manager.get()
    snapshot_manager.clear()
    try:
        responses = [
            client.post(
                path,
                headers={"X-API-Key": valid_api_key},
                json={"query": "query"},
            )
            for path in ("/search/sparse", "/search/hybrid")
        ]
    finally:
        if previous is not None:
            snapshot_manager.replace(previous)
    assert all(response.status_code == 503 for response in responses)
