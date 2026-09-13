"""API key authentication."""

import logging
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided_key: str | None = Security(api_key_header)) -> None:
    expected_key = settings.api_key.get_secret_value()
    if provided_key is None or not secrets.compare_digest(provided_key, expected_key):
        logger.warning("api_key_authentication_failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
