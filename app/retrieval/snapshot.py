"""Immutable in-memory snapshots and atomic filesystem persistence."""

import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from app.retrieval.dense import FaissStore, FaissStoreError


class SnapshotError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DenseSnapshot:
    version: int
    store: FaissStore
    chunk_ids: tuple[UUID, ...]
    embedding_model: str
    embedding_dimension: int

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_ids)


class SnapshotManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: DenseSnapshot | None = None

    def get(self) -> DenseSnapshot | None:
        with self._lock:
            return self._snapshot

    def replace(self, snapshot: DenseSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def clear(self) -> None:
        with self._lock:
            self._snapshot = None


snapshot_manager = SnapshotManager()


class SnapshotStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save_atomic(
        self,
        snapshot: DenseSnapshot,
        *,
        chunking_config_hashes: list[str],
    ) -> Path:
        final_directory = self.version_directory(snapshot.version)
        temporary = self.root / f".{snapshot.version:06d}-{uuid.uuid4().hex}.tmp"
        if final_directory.exists():
            raise SnapshotError("Snapshot version already exists on disk")
        temporary.mkdir(mode=0o750)
        try:
            snapshot.store.save(temporary / "faiss.index")
            self._write_json(
                temporary / "faiss_mapping.json",
                [str(chunk_id) for chunk_id in snapshot.chunk_ids],
            )
            self._write_json(
                temporary / "manifest.json",
                {
                    "snapshot_version": snapshot.version,
                    "created_at": datetime.now(UTC).isoformat(),
                    "embedding_model": snapshot.embedding_model,
                    "embedding_dimension": snapshot.embedding_dimension,
                    "chunk_count": snapshot.chunk_count,
                    "chunking_config_hashes": sorted(set(chunking_config_hashes)),
                    "index_type": snapshot.store.index_type,
                    "similarity_metric": snapshot.store.similarity_metric,
                },
            )
            validated = self._load_directory(temporary, expected_version=snapshot.version)
            if validated.chunk_ids != snapshot.chunk_ids:
                raise SnapshotError("Persisted snapshot mapping validation failed")
            os.replace(temporary, final_directory)
            return final_directory
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def load(self, version: int) -> DenseSnapshot:
        return self._load_directory(self.version_directory(version), expected_version=version)

    def remove(self, version: int) -> None:
        directory = self.version_directory(version)
        if directory.parent == self.root and directory.exists():
            shutil.rmtree(directory)

    def version_directory(self, version: int) -> Path:
        if version < 1:
            raise SnapshotError("Snapshot version must be positive")
        return self.root / f"{version:06d}"

    def _load_directory(self, directory: Path, *, expected_version: int) -> DenseSnapshot:
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            raw_mapping = json.loads(
                (directory / "faiss_mapping.json").read_text(encoding="utf-8")
            )
            if not isinstance(raw_mapping, list):
                raise ValueError("Snapshot mapping must be a list")
            store = FaissStore.load(directory / "faiss.index")
            chunk_ids = tuple(UUID(value) for value in raw_mapping)
            snapshot = DenseSnapshot(
                version=int(manifest["snapshot_version"]),
                store=store,
                chunk_ids=chunk_ids,
                embedding_model=str(manifest["embedding_model"]),
                embedding_dimension=int(manifest["embedding_dimension"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, FaissStoreError) as exc:
            raise SnapshotError("Snapshot files are missing, corrupt, or invalid") from exc

        if snapshot.version != expected_version:
            raise SnapshotError("Snapshot version does not match its directory")
        if not snapshot.embedding_model or snapshot.embedding_dimension < 1:
            raise SnapshotError("Snapshot embedding metadata is invalid")
        if snapshot.embedding_dimension != store.dimension:
            raise SnapshotError("Snapshot embedding dimension is inconsistent")
        if snapshot.chunk_count != store.count:
            raise SnapshotError("FAISS and mapping counts are inconsistent")
        if int(manifest.get("chunk_count", -1)) != snapshot.chunk_count:
            raise SnapshotError("Snapshot manifest count is inconsistent")
        if manifest.get("index_type") != store.index_type:
            raise SnapshotError("Snapshot index type is unsupported")
        if manifest.get("similarity_metric") != store.similarity_metric:
            raise SnapshotError("Snapshot similarity metric is unsupported")
        return snapshot

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
