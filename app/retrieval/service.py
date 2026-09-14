"""Combined snapshot rebuild, activation, and dense/sparse/hybrid search."""

import hashlib
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.embeddings import EmbeddingError, EmbeddingProvider
from app.models import Document, DocumentChunk, RetrievalIndexVersion
from app.retrieval.dense import FaissStore, FaissStoreError
from app.retrieval.fusion import RankedCandidate, reciprocal_rank_fusion
from app.retrieval.sparse import BM25Store, SparseStoreError
from app.retrieval.snapshot import (
    DenseSnapshot,
    SnapshotError,
    SnapshotManager,
    SnapshotStorage,
)

logger = logging.getLogger(__name__)
_rebuild_lock = threading.Lock()


class RetrievalError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class RebuildResult:
    snapshot_version: int
    status: str
    chunk_count: int
    embedding_model: str
    embedding_dimension: int


@dataclass(frozen=True, slots=True)
class SearchHit:
    rank: int
    score: float
    chunk_id: UUID
    document_id: UUID
    text: str
    source_name: str
    source_type: str
    page_number: int | None
    section_title: str | None


@dataclass(frozen=True, slots=True)
class HybridSearchHit:
    rank: int
    rrf_score: float
    dense_rank: int | None
    dense_score: float | None
    sparse_rank: int | None
    sparse_score: float | None
    chunk_id: UUID
    document_id: UUID
    text: str
    source_name: str
    source_type: str
    page_number: int | None
    section_title: str | None


class RetrievalService:
    def __init__(
        self,
        db: Session,
        provider: EmbeddingProvider,
        storage_root: Path,
        manager: SnapshotManager,
        rrf_k: int = 60,
    ) -> None:
        self.db = db
        self.provider = provider
        self.storage_root = storage_root
        self._storage: SnapshotStorage | None = None
        self.manager = manager
        self.rrf_k = rrf_k

    @property
    def storage(self) -> SnapshotStorage:
        if self._storage is None:
            self._storage = SnapshotStorage(self.storage_root)
        return self._storage

    def rebuild(self) -> RebuildResult:
        if not _rebuild_lock.acquire(blocking=False):
            raise RetrievalError(409, "A retrieval snapshot rebuild is already running")
        version_row_id: UUID | None = None
        version: int | None = None
        try:
            rows = self.db.execute(
                select(DocumentChunk, Document.chunking_config_hash)
                .join(Document)
                .where(Document.deleted_at.is_(None), Document.status != "DELETED")
                .order_by(DocumentChunk.id)
            ).all()
            if not rows:
                raise RetrievalError(409, "Cannot build a retrieval index without document chunks")

            version = int(
                self.db.scalar(select(func.coalesce(func.max(RetrievalIndexVersion.version), 0)))
                or 0
            ) + 1
            config_hashes = sorted({value for _, value in rows if value})
            version_row = RetrievalIndexVersion(
                version=version,
                status="BUILDING",
                chunk_count=len(rows),
                embedding_model=self.provider.model,
                chunking_config_hash=self._combined_hash(config_hashes),
                storage_path=(PurePosixPath("indexes") / "versions" / f"{version:06d}").as_posix(),
            )
            self.db.add(version_row)
            self.db.commit()
            version_row_id = version_row.id

            chunks = [chunk for chunk, _ in rows]
            texts = [chunk.text_content for chunk in chunks]
            vectors = self.provider.embed_documents(texts)
            store = FaissStore.build(vectors)
            sparse_store = BM25Store.build(texts)
            chunk_ids = tuple(chunk.id for chunk in chunks)
            snapshot = DenseSnapshot(
                version=version,
                store=store,
                chunk_ids=chunk_ids,
                embedding_model=self.provider.model,
                embedding_dimension=store.dimension,
                sparse_store=sparse_store,
                sparse_chunk_ids=chunk_ids,
                rrf_k=self.rrf_k,
            )
            self._validate_mapping(snapshot)
            self.storage.save_atomic(snapshot, chunking_config_hashes=config_hashes)
            persisted = self.storage.load(version)
            self._validate_mapping(persisted)

            self.db.execute(
                update(RetrievalIndexVersion)
                .where(RetrievalIndexVersion.status == "ACTIVATED")
                .values(status="DEPRECATED")
            )
            self.db.flush()
            active_row = self.db.get(RetrievalIndexVersion, version_row_id)
            if active_row is None:
                raise SnapshotError("Snapshot database row disappeared during activation")
            active_row.status = "ACTIVATED"
            active_row.activated_at = datetime.now(UTC)
            self.db.commit()
            self.manager.replace(persisted)
            return RebuildResult(
                snapshot_version=version,
                status="ACTIVATED",
                chunk_count=persisted.chunk_count,
                embedding_model=persisted.embedding_model,
                embedding_dimension=persisted.embedding_dimension,
            )
        except RetrievalError:
            self.db.rollback()
            raise
        except EmbeddingError as exc:
            self._record_failure(version_row_id, version, "Embedding service failed")
            raise RetrievalError(503, str(exc)) from None
        except (SnapshotError, FaissStoreError, SparseStoreError):
            logger.exception("retrieval_snapshot_build_failed version=%s", version)
            self._record_failure(version_row_id, version, "Snapshot validation failed")
            raise RetrievalError(500, "Retrieval snapshot build failed") from None
        except Exception:
            logger.exception("retrieval_snapshot_build_failed version=%s", version)
            self._record_failure(version_row_id, version, "Snapshot build failed")
            raise RetrievalError(500, "Retrieval snapshot build failed") from None
        finally:
            _rebuild_lock.release()

    def search(self, query: str, k: int) -> tuple[int, list[SearchHit]]:
        snapshot = self.manager.get()
        if snapshot is None:
            raise RetrievalError(503, "No active retrieval index is loaded")
        if self.provider.model != snapshot.embedding_model:
            raise RetrievalError(
                503, "Configured embedding model does not match the active retrieval index"
            )
        try:
            query_vector = self.provider.embed_query(query)
            matches = snapshot.store.search(query_vector, k)
        except EmbeddingError as exc:
            raise RetrievalError(503, str(exc)) from None
        except FaissStoreError as exc:
            raise RetrievalError(503, str(exc)) from None

        mapped_ids = [snapshot.chunk_ids[match.position] for match in matches]
        canonical = self._hydrate(mapped_ids, snapshot.version, "dense")
        hits: list[SearchHit] = []
        for match, chunk_id in zip(matches, mapped_ids, strict=True):
            row = canonical.get(chunk_id)
            if row is None:
                self._log_stale(snapshot.version, chunk_id)
                continue
            hits.append(self._search_hit(len(hits) + 1, match.score, row))
        return snapshot.version, hits

    def search_sparse(self, query: str, k: int) -> tuple[int, list[SearchHit]]:
        snapshot = self._active_snapshot()
        matches = snapshot.sparse_store.search(query, k)
        mapped_ids = [snapshot.sparse_chunk_ids[match.position] for match in matches]
        canonical = self._hydrate(mapped_ids, snapshot.version, "sparse")
        hits: list[SearchHit] = []
        for match, chunk_id in zip(matches, mapped_ids, strict=True):
            row = canonical.get(chunk_id)
            if row is None:
                self._log_stale(snapshot.version, chunk_id)
                continue
            hits.append(self._search_hit(len(hits) + 1, match.score, row))
        return snapshot.version, hits

    def search_hybrid(
        self,
        query: str,
        k: int,
        dense_candidate_k: int,
        sparse_candidate_k: int,
    ) -> tuple[int, list[HybridSearchHit]]:
        snapshot = self._active_snapshot()
        if self.provider.model != snapshot.embedding_model:
            raise RetrievalError(
                503, "Configured embedding model does not match the active retrieval index"
            )
        try:
            dense_matches = snapshot.store.search(
                self.provider.embed_query(query), dense_candidate_k
            )
        except EmbeddingError as exc:
            raise RetrievalError(503, str(exc)) from None
        except FaissStoreError as exc:
            raise RetrievalError(503, str(exc)) from None
        sparse_matches = snapshot.sparse_store.search(query, sparse_candidate_k)
        dense = [
            RankedCandidate(snapshot.chunk_ids[item.position], rank, item.score)
            for rank, item in enumerate(dense_matches, start=1)
        ]
        sparse = [
            RankedCandidate(snapshot.sparse_chunk_ids[item.position], rank, item.score)
            for rank, item in enumerate(sparse_matches, start=1)
        ]
        fused = reciprocal_rank_fusion(dense, sparse, snapshot.rrf_k)[:k]
        canonical = self._hydrate(
            [item.chunk_id for item in fused], snapshot.version, "hybrid"
        )
        hits: list[HybridSearchHit] = []
        for item in fused:
            row = canonical.get(item.chunk_id)
            if row is None:
                self._log_stale(snapshot.version, item.chunk_id)
                continue
            chunk, document = row
            hits.append(
                HybridSearchHit(
                    rank=len(hits) + 1,
                    rrf_score=item.rrf_score,
                    dense_rank=item.dense_rank,
                    dense_score=item.dense_score,
                    sparse_rank=item.sparse_rank,
                    sparse_score=item.sparse_score,
                    chunk_id=chunk.id,
                    document_id=document.id,
                    text=chunk.text_content,
                    source_name=document.source_name,
                    source_type=document.source_type,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                )
            )
        return snapshot.version, hits

    def _active_snapshot(self) -> DenseSnapshot:
        snapshot = self.manager.get()
        if snapshot is None:
            raise RetrievalError(503, "No active retrieval index is loaded")
        return snapshot

    def _hydrate(
        self, mapped_ids: list[UUID], version: int, search_type: str
    ) -> dict[UUID, tuple[DocumentChunk, Document]]:
        if not mapped_ids:
            return {}
        try:
            rows = self.db.execute(
                select(DocumentChunk, Document)
                .join(Document)
                .where(
                    DocumentChunk.id.in_(mapped_ids),
                    Document.deleted_at.is_(None),
                    Document.status != "DELETED",
                )
            ).all()
        except SQLAlchemyError:
            logger.exception("%s_search_database_lookup_failed version=%d", search_type, version)
            raise RetrievalError(500, f"{search_type.capitalize()} search failed") from None
        return {chunk.id: (chunk, document) for chunk, document in rows}

    @staticmethod
    def _search_hit(
        rank: int, score: float, row: tuple[DocumentChunk, Document]
    ) -> SearchHit:
        chunk, document = row
        return SearchHit(
            rank=rank,
            score=score,
            chunk_id=chunk.id,
            document_id=document.id,
            text=chunk.text_content,
            source_name=document.source_name,
            source_type=document.source_type,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
        )

    @staticmethod
    def _log_stale(version: int, chunk_id: UUID) -> None:
        logger.error(
            "retrieval_snapshot_stale_mapping version=%d chunk_id=%s", version, chunk_id
        )

    def _validate_mapping(self, snapshot: DenseSnapshot) -> None:
        if snapshot.store.count != snapshot.chunk_count:
            raise SnapshotError("FAISS and mapping counts are inconsistent")
        if snapshot.sparse_store.count != len(snapshot.sparse_chunk_ids):
            raise SnapshotError("BM25 and sparse mapping counts are inconsistent")
        if snapshot.sparse_store.count != snapshot.chunk_count:
            raise SnapshotError("Dense and sparse snapshot counts are inconsistent")
        if len(set(snapshot.chunk_ids)) != snapshot.chunk_count:
            raise SnapshotError("Dense mapping contains duplicate chunks")
        if len(set(snapshot.sparse_chunk_ids)) != snapshot.chunk_count:
            raise SnapshotError("Sparse mapping contains duplicate chunks")
        if set(snapshot.chunk_ids) != set(snapshot.sparse_chunk_ids):
            raise SnapshotError("Dense and sparse mappings contain different chunks")
        valid_ids = set(
            self.db.scalars(
                select(DocumentChunk.id)
                .join(Document)
                .where(
                    DocumentChunk.id.in_(snapshot.chunk_ids),
                    Document.deleted_at.is_(None),
                    Document.status != "DELETED",
                )
            ).all()
        )
        if valid_ids != set(snapshot.chunk_ids) or len(valid_ids) != snapshot.chunk_count:
            raise SnapshotError("Snapshot mapping contains missing or inactive chunks")

    def _record_failure(
        self, version_row_id: UUID | None, version: int | None, message: str
    ) -> None:
        self.db.rollback()
        if version is not None:
            try:
                self.storage.remove(version)
            except Exception:
                logger.exception("retrieval_snapshot_cleanup_failed version=%s", version)
        if version_row_id is None:
            return
        try:
            row = self.db.get(RetrievalIndexVersion, version_row_id)
            if row is not None:
                row.status = "FAILED"
                row.error_message = message
                self.db.commit()
        except Exception:
            self.db.rollback()
            logger.exception("retrieval_snapshot_failure_recording_failed version=%s", version)

    @staticmethod
    def _combined_hash(values: list[str]) -> str | None:
        if not values:
            return None
        return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def load_active_snapshot(
    db: Session, storage_root: Path, manager: SnapshotManager
) -> DenseSnapshot | None:
    row = db.scalar(
        select(RetrievalIndexVersion).where(RetrievalIndexVersion.status == "ACTIVATED")
    )
    if row is None:
        manager.clear()
        return None
    snapshot = SnapshotStorage(storage_root).load(row.version)
    if row.chunk_count != snapshot.chunk_count or row.embedding_model != snapshot.embedding_model:
        raise SnapshotError("Active snapshot does not match its database record")
    valid_count = db.scalar(
        select(func.count(DocumentChunk.id))
        .join(Document)
        .where(
            DocumentChunk.id.in_(snapshot.chunk_ids),
            Document.deleted_at.is_(None),
            Document.status != "DELETED",
        )
    )
    if valid_count != snapshot.chunk_count:
        raise SnapshotError("Active snapshot maps missing or inactive chunks")
    manager.replace(snapshot)
    return snapshot
