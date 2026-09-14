"""Typed environment-backed application settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
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
    max_upload_size_mb: int = Field(default=10, gt=0)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    storage_root: Path = Path("storage/documents")
    ollama_base_url: AnyHttpUrl = "http://localhost:11434"
    embedding_model: str = Field(default="nomic-embed-text", min_length=1)
    embedding_batch_size: int = Field(default=32, gt=0)
    ollama_timeout_seconds: float = Field(default=60.0, gt=0)
    dense_top_k: int = Field(default=5, gt=0)
    dense_max_k: int = Field(default=20, gt=0)
    index_storage_root: Path = Path("storage/indexes/versions")

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.dense_top_k > self.dense_max_k:
            raise ValueError("DENSE_TOP_K must not exceed DENSE_MAX_K")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
