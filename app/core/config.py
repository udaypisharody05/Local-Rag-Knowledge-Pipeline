"""Typed environment-backed application settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Local RAG Knowledge Pipeline"
    app_env: Literal["development", "test", "staging", "production"] = "development"
    api_key: SecretStr = Field(min_length=1)
    database_url: str = Field(min_length=1)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
