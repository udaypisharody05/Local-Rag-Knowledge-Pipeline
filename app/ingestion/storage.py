"""Controlled streaming storage for uploaded source files."""

import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


class UploadTooLargeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StagedUpload:
    path: Path
    size_bytes: int
    content_hash: str


class DocumentStorage:
    def __init__(self, root: Path, max_size_bytes: int) -> None:
        self.root = root.resolve()
        self.max_size_bytes = max_size_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def stage(self, source: BinaryIO) -> StagedUpload:
        temporary = self.root / f".upload-{uuid.uuid4().hex}.tmp"
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("xb") as destination:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > self.max_size_bytes:
                        raise UploadTooLargeError
                    digest.update(block)
                    destination.write(block)
            return StagedUpload(temporary, size, digest.hexdigest())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def finalize(self, staged: StagedUpload, document_id: uuid.UUID, extension: str) -> Path:
        directory = self.root / str(document_id)
        directory.mkdir(mode=0o750)
        destination = directory / f"original.{extension}"
        os.replace(staged.path, destination)
        return destination

    def remove_document(self, document_id: uuid.UUID) -> None:
        directory = self.root / str(document_id)
        if directory.parent == self.root and directory.exists():
            shutil.rmtree(directory)


class JobStagingStorage:
    """Server-generated staging paths shared by the API and ingestion worker."""

    def __init__(self, root: Path, max_size_bytes: int) -> None:
        self.root = root.resolve()
        self.max_size_bytes = max_size_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def stage(self, source: BinaryIO, job_id: uuid.UUID, extension: str) -> StagedUpload:
        directory = self._directory(job_id)
        directory.mkdir(mode=0o750)
        destination = directory / self._filename(extension)
        digest = hashlib.sha256()
        size = 0
        try:
            with destination.open("xb") as output:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > self.max_size_bytes:
                        raise UploadTooLargeError
                    digest.update(block)
                    output.write(block)
            return StagedUpload(destination, size, digest.hexdigest())
        except Exception:
            self.remove(job_id)
            raise

    def load(self, job_id: uuid.UUID, extension: str) -> StagedUpload:
        path = self._directory(job_id) / self._filename(extension)
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > self.max_size_bytes:
                        raise UploadTooLargeError
                    digest.update(block)
        except FileNotFoundError:
            raise ValueError("Staged upload is unavailable") from None
        return StagedUpload(path, size, digest.hexdigest())

    def remove(self, job_id: uuid.UUID) -> None:
        directory = self._directory(job_id)
        if directory.exists():
            shutil.rmtree(directory)

    def _directory(self, job_id: uuid.UUID) -> Path:
        directory = (self.root / str(job_id)).resolve()
        if directory.parent != self.root:
            raise ValueError("Invalid staging job identifier")
        return directory

    @staticmethod
    def _filename(extension: str) -> str:
        if extension not in {"txt", "md", "pdf"}:
            raise ValueError("Invalid staged upload extension")
        return f"original.{extension}"
