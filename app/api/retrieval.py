"""Protected snapshot management and dense-search endpoints."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_embedding_provider
from app.core.config import settings
from app.core.security import require_api_key
from app.embeddings import EmbeddingProvider
from app.retrieval import snapshot_manager
from app.retrieval.service import RetrievalError, RetrievalService
from app.schemas.retrieval import (
    DenseSearchRequest,
    DenseSearchResponse,
    DenseSearchResult,
    RebuildResponse,
    RetrievalStatusResponse,
)

router = APIRouter(tags=["retrieval"], dependencies=[Depends(require_api_key)])


@router.post("/retrieval/index/rebuild", response_model=RebuildResponse)
def rebuild_index(
    db: Session = Depends(get_db),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> RebuildResponse | JSONResponse:
    service = RetrievalService(db, provider, settings.index_storage_root, snapshot_manager)
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
    service = RetrievalService(db, provider, settings.index_storage_root, snapshot_manager)
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


@router.get("/retrieval/status", response_model=RetrievalStatusResponse)
def retrieval_status() -> RetrievalStatusResponse:
    snapshot = snapshot_manager.get()
    if snapshot is None:
        return RetrievalStatusResponse(available=False, snapshot_version=None)
    return RetrievalStatusResponse(
        available=True,
        snapshot_version=snapshot.version,
        chunk_count=snapshot.chunk_count,
        embedding_model=snapshot.embedding_model,
    )
