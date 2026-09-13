"""Common structures shared by document loaders."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class LoaderError(ValueError):
    """Raised when a supported file cannot be decoded or extracted."""


@dataclass(frozen=True, slots=True)
class ExtractedUnit:
    text: str
    page_number: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    units: list[ExtractedUnit]
    parser_version: str


class DocumentLoader(Protocol):
    parser_version: str

    def load(self, path: Path, source_filename: str) -> LoadedDocument: ...
