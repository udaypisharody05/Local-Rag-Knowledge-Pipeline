"""Minimal local text-generation provider contract."""

from collections.abc import AsyncIterator
from typing import Protocol


class GenerationError(RuntimeError):
    """A sanitized generation-provider failure."""


class GenerationProvider(Protocol):
    model: str

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...

    def stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]: ...
