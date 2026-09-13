"""API key dependency behavior."""

from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api.dependencies import get_db
from app.main import app


def _working_db() -> Mock:
    session = Mock()
    session.execute.return_value = Mock()
    return session


def test_missing_api_key_is_rejected(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 401


def test_wrong_api_key_is_rejected(client: TestClient) -> None:
    response = client.get("/ready", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_correct_api_key_is_accepted(client: TestClient, valid_api_key: str) -> None:
    app.dependency_overrides[get_db] = _working_db
    try:
        response = client.get("/ready", headers={"X-API-Key": valid_api_key})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "ok"}
