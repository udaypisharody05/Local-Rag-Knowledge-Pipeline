"""Reusable FastAPI dependencies."""

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.db.session import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Provide one SQLAlchemy session per request."""
    with SessionLocal() as session:
        yield session
