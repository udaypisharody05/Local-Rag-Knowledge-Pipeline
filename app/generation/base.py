"""Minimal local text-generation provider contract."""

from typing import Protocol


class GenerationError(RuntimeError):
    """A sanitized generation-provider failure."""


class GenerationProvider(Protocol):
    model: str

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
