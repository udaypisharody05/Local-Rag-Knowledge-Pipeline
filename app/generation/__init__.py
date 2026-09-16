"""Grounded local answer generation."""

from app.generation.base import GenerationError, GenerationProvider
from app.generation.ollama import OllamaGenerationProvider
from app.generation.service import (
    INSUFFICIENT_CONTEXT_ANSWER,
    GroundedGenerationService,
    GroundedQueryResult,
    PreparedGroundedQuery,
    is_insufficient_context_answer,
)

__all__ = [
    "GenerationError",
    "GenerationProvider",
    "GroundedGenerationService",
    "GroundedQueryResult",
    "PreparedGroundedQuery",
    "INSUFFICIENT_CONTEXT_ANSWER",
    "OllamaGenerationProvider",
    "is_insufficient_context_answer",
]
