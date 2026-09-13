"""Liveness and readiness endpoint tests."""

from unittest.mock import Mock

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import get_db
from app.main import app


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_failure_is_sanitized(client: TestClient, valid_api_key: str) -> None:
    broken_session = Mock()
    broken_session.execute.side_effect = SQLAlchemyError("secret database detail")
    app.dependency_overrides[get_db] = lambda: broken_session
    try:
        response = client.get("/ready", headers={"X-API-Key": valid_api_key})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Database is unavailable"}
    assert "secret database detail" not in response.text
