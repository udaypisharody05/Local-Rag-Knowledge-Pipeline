"""Protected combined-snapshot management and retrieval endpoints."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_embedding_provider
from app.core.config import settings
from app.core.security import require_api_key
from app.embeddings import EmbeddingProvider
from app.retrieval import snapshot_manager
from app.retrieval.service import RetrievalError, RetrievalService, SearchHit
from app.schemas.retrieval import (
    DenseSearchRequest,
    DenseSearchResponse,
    DenseSearchResult,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResult,
    RebuildResponse,
    RetrievalStatusResponse,
    SparseSearchRequest,
    SparseSearchResponse,
    SparseSearchResult,
)

router = APIRouter(tags=["retrieval"], dependencies=[Depends(require_api_key)])


@router.post("/retrieval/index/rebuild", response_model=RebuildResponse)
def rebuild_index(
    db: Session = Depends(get_db),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> RebuildResponse | JSONResponse:
    service = _service(db, provider)
    try:
        result = service.rebuild()
    except RetrievalError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return RebuildResponse(
        snapshot_version=result.snapshot_version,
        status=result.status,
        chunk_count=result.chunk_count,
        embedding_model=result.embedding_model,
        embedding_dimension=result.embedding_dimension,
    )


@router.post("/search/dense", response_model=DenseSearchResponse)
def dense_search(
    request: DenseSearchRequest,
    db: Session = Depends(get_db),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> DenseSearchResponse | JSONResponse:
    k = request.k if request.k is not None else settings.dense_top_k
    if k > settings.dense_max_k:
        return JSONResponse(
            status_code=422,
            content={"detail": f"k must be less than or equal to {settings.dense_max_k}"},
        )
    service = _service(db, provider)
    try:
        version, hits = service.search(request.query, k)
    except RetrievalError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return DenseSearchResponse(
        query=request.query,
        snapshot_version=version,
        results=[
            DenseSearchResult(
                rank=hit.rank,
                score=hit.score,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                text=hit.text,
                source_name=hit.source_name,
                source_type=hit.source_type,
                page_number=hit.page_number,
                section_title=hit.section_title,
            )
            for hit in hits
        ],
    )


@router.post("/search/sparse", response_model=SparseSearchResponse)
def sparse_search(
    request: SparseSearchRequest,
    db: Session = Depends(get_db),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> SparseSearchResponse | JSONResponse:
    k = request.k if request.k is not None else settings.sparse_top_k
    if k > settings.sparse_max_k:
        return JSONResponse(
            status_code=422,
            content={"detail": f"k must be less than or equal to {settings.sparse_max_k}"},
        )
    try:
        version, hits = _service(db, provider).search_sparse(request.query, k)
    except RetrievalError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return SparseSearchResponse(
        query=request.query,
        snapshot_version=version,
        results=[SparseSearchResult(**_standard_hit(hit)) for hit in hits],
    )


@router.post("/search/hybrid", response_model=HybridSearchResponse)
def hybrid_search(
    request: HybridSearchRequest,
    db: Session = Depends(get_db),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> HybridSearchResponse | JSONResponse:
    k = request.k if request.k is not None else settings.hybrid_top_k
    if k > settings.hybrid_max_k:
        return JSONResponse(
            status_code=422,
            content={"detail": f"k must be less than or equal to {settings.hybrid_max_k}"},
        )
    try:
        version, hits = _service(db, provider).search_hybrid(
            request.query,
            k,
            settings.dense_candidate_k,
            settings.sparse_candidate_k,
        )
    except RetrievalError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return HybridSearchResponse(
        query=request.query,
        snapshot_version=version,
        fusion="rrf",
        results=[
            HybridSearchResult(
                rank=hit.rank,
                rrf_score=hit.rrf_score,
                dense_rank=hit.dense_rank,
                dense_score=hit.dense_score,
                sparse_rank=hit.sparse_rank,
                sparse_score=hit.sparse_score,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                text=hit.text,
                source_name=hit.source_name,
                source_type=hit.source_type,
                page_number=hit.page_number,
                section_title=hit.section_title,
            )
            for hit in hits
        ],
    )


@router.get("/retrieval/status", response_model=RetrievalStatusResponse)
def retrieval_status() -> RetrievalStatusResponse:
    snapshot = snapshot_manager.get()
    if snapshot is None:
        return RetrievalStatusResponse(
            available=False,
            dense_available=False,
            sparse_available=False,
            fusion_strategy=None,
            snapshot_version=None,
        )
    return RetrievalStatusResponse(
        available=True,
        dense_available=True,
        sparse_available=True,
        fusion_strategy="rrf",
        snapshot_version=snapshot.version,
        chunk_count=snapshot.chunk_count,
        embedding_model=snapshot.embedding_model,
    )


def _service(db: Session, provider: EmbeddingProvider) -> RetrievalService:
    return RetrievalService(
        db, provider, settings.index_storage_root, snapshot_manager, settings.rrf_k
    )


def _standard_hit(hit: SearchHit) -> dict[str, object]:
    return {
        "rank": hit.rank,
        "score": hit.score,
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "text": hit.text,
        "source_name": hit.source_name,
        "source_type": hit.source_type,
        "page_number": hit.page_number,
        "section_title": hit.section_title,
    }
