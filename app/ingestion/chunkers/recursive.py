"""Small recursive-separator character chunker."""

import hashlib
from dataclasses import dataclass
from typing import Any

from app.ingestion.loaders.base import ExtractedUnit

SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_index: int
    text_content: str
    content_hash: str
    page_number: int | None
    section_title: str | None
    start_offset: int
    end_offset: int
    metadata: dict[str, Any]


class RecursiveCharacterChunker:
    strategy_name = "recursive-character-v1"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Require 0 <= chunk_overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, units: list[ExtractedUnit]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for unit in units:
            for text, start, end in self._split(unit.text):
                metadata = dict(unit.metadata)
                if unit.page_number is not None:
                    metadata["page_number"] = unit.page_number
                if unit.section_title is not None:
                    metadata["section_title"] = unit.section_title
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        text_content=text,
                        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        page_number=unit.page_number,
                        section_title=unit.section_title,
                        start_offset=start,
                        end_offset=end,
                        metadata=metadata,
                    )
                )
        return chunks

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        pieces: list[tuple[str, int, int]] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                search_floor = start + max(1, self.chunk_size // 2)
                for separator in SEPARATORS:
                    boundary = text.rfind(separator, search_floor, end)
                    if boundary >= 0:
                        end = boundary + len(separator)
                        break

            raw = text[start:end]
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw) - len(raw.rstrip())
            clean_start = start + leading
            clean_end = end - trailing
            if clean_start < clean_end:
                clean_text = text[clean_start:clean_end]
                pieces.append((clean_text, clean_start, clean_end))

            if end >= length:
                break
            start = max(end - self.chunk_overlap, start + 1)
        return pieces
