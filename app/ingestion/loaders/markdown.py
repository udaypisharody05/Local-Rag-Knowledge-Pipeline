"""Lightweight Markdown loader that tracks ATX headings."""

import re
from pathlib import Path

from app.ingestion.loaders.base import ExtractedUnit, LoadedDocument, LoaderError

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


class MarkdownLoader:
    parser_version = "markdown-v1"

    def load(self, path: Path, source_filename: str) -> LoadedDocument:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise LoaderError("Markdown file must be valid UTF-8") from exc
        if "\x00" in text:
            raise LoaderError("Markdown file contains binary data")
        if not text.strip():
            raise LoaderError("Markdown file contains no useful text")

        units: list[ExtractedUnit] = []
        current_lines: list[str] = []
        current_heading: str | None = None

        def flush() -> None:
            section = "".join(current_lines)
            if section.strip():
                units.append(
                    ExtractedUnit(
                        text=section,
                        section_title=current_heading,
                        metadata={"source_filename": source_filename},
                    )
                )

        for line in text.splitlines(keepends=True):
            match = HEADING_RE.match(line.rstrip("\r\n"))
            if match:
                flush()
                current_lines = [line]
                current_heading = match.group(1).strip()
            else:
                current_lines.append(line)
        flush()

        return LoadedDocument(units=units, parser_version=self.parser_version)
