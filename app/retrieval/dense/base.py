"""Minimal dense-store result structure."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DenseMatch:
    position: int
    score: float
