"""Shared deterministic test configuration."""

import os

from dotenv import load_dotenv

external_database_url = os.environ.get("DATABASE_URL")
load_dotenv()
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("API_KEY", "unit-test-placeholder")
os.environ["DATABASE_URL"] = (
    os.environ.get("TEST_DATABASE_URL")
    or external_database_url
    or "postgresql+psycopg://localhost/rag_test"
)
os.environ.setdefault("LOG_LEVEL", "WARNING")

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_api_key() -> str:
    return settings.api_key.get_secret_value()
