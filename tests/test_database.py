"""PostgreSQL migration integration tests."""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings

EXPECTED_TABLES = {
    "alembic_version",
    "documents",
    "document_chunks",
    "ingestion_jobs",
    "retrieval_index_versions",
}


@pytest.fixture(scope="module")
def migrated_engine():
    engine = create_engine(settings.database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar_one() == 1
    except SQLAlchemyError as exc:
        engine.dispose()
        pytest.skip(f"PostgreSQL integration database unavailable: {type(exc).__name__}")

    command.upgrade(Config("alembic.ini"), "head")
    yield engine
    engine.dispose()


def test_database_connection(migrated_engine) -> None:
    with migrated_engine.connect() as connection:
        assert connection.execute(text("SELECT 1")).scalar_one() == 1


def test_migrations_create_expected_tables(migrated_engine) -> None:
    assert EXPECTED_TABLES <= set(inspect(migrated_engine).get_table_names())
