"""UTF-8 plain-text loader."""

from pathlib import Path

from app.ingestion.loaders.base import ExtractedUnit, LoadedDocument, LoaderError


class TextLoader:
    parser_version = "txt-v1"

    def load(self, path: Path, source_filename: str) -> LoadedDocument:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise LoaderError("Text file must be valid UTF-8") from exc
        if "\x00" in text:
            raise LoaderError("Text file contains binary data")
        if not text.strip():
            raise LoaderError("Text file contains no useful text")
        return LoadedDocument(
            units=[ExtractedUnit(text=text, metadata={"source_filename": source_filename})],
            parser_version=self.parser_version,
        )
