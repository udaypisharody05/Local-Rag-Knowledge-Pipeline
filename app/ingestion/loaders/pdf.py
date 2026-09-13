"""Page-aware PDF text loader."""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.loaders.base import ExtractedUnit, LoadedDocument, LoaderError


class PdfLoader:
    parser_version = "pdf-v1"

    def load(self, path: Path, source_filename: str) -> LoadedDocument:
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise LoaderError("Encrypted PDFs are not supported")
            units = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    units.append(
                        ExtractedUnit(
                            text=text,
                            page_number=page_number,
                            metadata={"source_filename": source_filename},
                        )
                    )
        except LoaderError:
            raise
        except (PdfReadError, OSError, ValueError) as exc:
            raise LoaderError("PDF could not be read") from exc

        if not units:
            raise LoaderError("PDF contains no extractable text; OCR is not supported")
        return LoadedDocument(units=units, parser_version=self.parser_version)
