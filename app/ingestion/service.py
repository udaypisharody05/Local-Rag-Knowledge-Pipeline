"""Transactional orchestration for synchronous document ingestion."""

import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.ingestion.chunkers import RecursiveCharacterChunker
from app.ingestion.loaders import LOADERS
from app.ingestion.loaders.base import LoaderError
from app.ingestion.storage import DocumentStorage, StagedUpload, UploadTooLargeError
from app.models import Document, DocumentChunk
from app.schemas.documents import UploadResponse

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = {
    "txt": {"text/plain", "application/octet-stream"},
    "md": {"text/markdown", "text/plain", "application/octet-stream"},
    "pdf": {"application/pdf", "application/octet-stream"},
}
CANONICAL_MIME_TYPES = {"txt": "text/plain", "md": "text/markdown", "pdf": "application/pdf"}


class IngestionError(Exception):
    def __init__(
        self, status_code: int, detail: str, *, extra: dict[str, Any] | None = None
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.extra = extra or {}

    def response_body(self) -> dict[str, Any]:
        return {"detail": self.detail, **self.extra}


class IngestionService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.storage = DocumentStorage(
            settings.storage_root, settings.max_upload_size_mb * 1024 * 1024
        )
        self.chunker = RecursiveCharacterChunker(settings.chunk_size, settings.chunk_overlap)

    def ingest(self, upload: UploadFile) -> UploadResponse:
        filename, extension = self._validate_filename(upload.filename)
        self._validate_declared_mime(extension, upload.content_type)

        staged: StagedUpload | None = None
        document_id: uuid.UUID | None = None
        storage_created = False
        completed = False
        try:
            try:
                staged = self.storage.stage(upload.file)
            except UploadTooLargeError:
                raise IngestionError(
                    413,
                    f"File exceeds the {self.settings.max_upload_size_mb} MB upload limit",
                ) from None

            if staged.size_bytes == 0:
                raise IngestionError(400, "Uploaded file is empty")
            self._validate_signature(extension, staged)

            duplicate = self._find_duplicate(staged.content_hash)
            if duplicate is not None:
                raise self._duplicate_error(duplicate)

            document_id = uuid.uuid4()
            stored_file = self.storage.finalize(staged, document_id, extension)
            storage_created = True
            loaded = LOADERS[extension].load(stored_file, filename)
            chunks = self.chunker.chunk(loaded.units)
            if not chunks:
                raise IngestionError(422, "Document contains no chunkable text")

            now = datetime.now(UTC)
            document = Document(
                id=document_id,
                source_name=filename,
                source_type=extension,
                original_filename=filename,
                storage_path=(PurePosixPath("documents") / str(document_id) / f"original.{extension}").as_posix(),
                mime_type=CANONICAL_MIME_TYPES[extension],
                size_bytes=staged.size_bytes,
                content_hash=staged.content_hash,
                status="INDEXED",
                parser_version=loaded.parser_version,
                chunking_config_hash=self._chunking_config_hash(),
                indexed_at=now,
            )
            document.chunks = [
                DocumentChunk(
                    text_content=chunk.text_content,
                    chunk_index=chunk.chunk_index,
                    content_hash=chunk.content_hash,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    chunk_metadata={
                        **chunk.metadata,
                        "source_type": extension,
                        "start_offset": chunk.start_offset,
                        "end_offset": chunk.end_offset,
                    },
                )
                for chunk in chunks
            ]
            response = UploadResponse(
                document_id=document_id,
                source_name=filename,
                source_type=extension,
                status="INDEXED",
                chunk_count=len(chunks),
            )
            self.db.add(document)
            self.db.commit()
            completed = True
            return response
        except IngestionError:
            self.db.rollback()
            raise
        except LoaderError as exc:
            self.db.rollback()
            raise IngestionError(422, str(exc)) from None
        except IntegrityError:
            self.db.rollback()
            existing = self._find_duplicate(staged.content_hash) if staged else None
            if existing is not None:
                raise self._duplicate_error(existing) from None
            logger.exception("document_ingestion_database_conflict")
            raise IngestionError(500, "Document ingestion failed") from None
        except Exception:
            self.db.rollback()
            logger.exception("document_ingestion_failed source_type=%s", extension)
            raise IngestionError(500, "Document ingestion failed") from None
        finally:
            if staged is not None:
                staged.path.unlink(missing_ok=True)
            if document_id is not None and storage_created and not completed:
                self.storage.remove_document(document_id)

    def _find_duplicate(self, content_hash: str) -> Document | None:
        return self.db.scalar(
            select(Document).where(
                Document.content_hash == content_hash,
                Document.deleted_at.is_(None),
            )
        )

    @staticmethod
    def _duplicate_error(document: Document) -> IngestionError:
        return IngestionError(
            409,
            "Document with identical content already exists",
            extra={"document_id": str(document.id)},
        )

    @staticmethod
    def _validate_filename(raw_filename: str | None) -> tuple[str, str]:
        if not raw_filename:
            raise IngestionError(400, "A filename is required")
        filename = PurePosixPath(raw_filename.replace("\\", "/")).name.strip()
        filename = "".join(character for character in filename if character.isprintable())[:512]
        if not filename or filename in {".", ".."}:
            raise IngestionError(400, "Filename is invalid")
        suffix = PurePosixPath(filename).suffix.lower()
        if suffix not in {".txt", ".md", ".pdf"}:
            raise IngestionError(400, "Unsupported file type; use TXT, Markdown, or PDF")
        return filename, suffix.removeprefix(".")

    @staticmethod
    def _validate_declared_mime(extension: str, content_type: str | None) -> None:
        if content_type is None:
            return
        normalized = content_type.partition(";")[0].strip().lower()
        if normalized and normalized not in SUPPORTED_MIME_TYPES[extension]:
            raise IngestionError(400, "File content type does not match its extension")

    @staticmethod
    def _validate_signature(extension: str, staged: StagedUpload) -> None:
        if extension == "pdf":
            with staged.path.open("rb") as source:
                if source.read(5) != b"%PDF-":
                    raise IngestionError(422, "File is not a valid PDF")

    def _chunking_config_hash(self) -> str:
        payload = json.dumps(
            {
                "strategy": self.chunker.strategy_name,
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
