"""Protected grounded question-answering endpoint."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_db,
    get_embedding_provider,
    get_generation_provider,
)
from app.core.config import settings
from app.core.security import require_api_key
from app.embeddings import EmbeddingProvider
from app.generation import GenerationError, GenerationProvider, GroundedGenerationService
from app.retrieval import snapshot_manager
from app.retrieval.service import RetrievalError, RetrievalService
from app.schemas.query import (
    CitationResponse,
    QueryRequest,
    QueryResponse,
    QueryRetrievalMetadata,
)

router = APIRouter(tags=["generation"], dependencies=[Depends(require_api_key)])


@router.post("/query", response_model=QueryResponse)
def query_knowledge_base(
    request: QueryRequest,
    db: Session = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    generation_provider: GenerationProvider = Depends(get_generation_provider),
) -> QueryResponse | JSONResponse:
    k = request.k if request.k is not None else settings.hybrid_top_k
    if k > settings.hybrid_max_k:
        return JSONResponse(
            status_code=422,
            content={"detail": f"k must be less than or equal to {settings.hybrid_max_k}"},
        )
    retrieval = RetrievalService(
        db,
        embedding_provider,
        settings.index_storage_root,
        snapshot_manager,
        settings.rrf_k,
    )
    service = GroundedGenerationService(
        retrieval,
        generation_provider,
        dense_candidate_k=settings.dense_candidate_k,
        sparse_candidate_k=settings.sparse_candidate_k,
        max_context_chunks=settings.generation_max_context_chunks,
        max_context_chars=settings.generation_max_context_chars,
    )
    try:
        result = service.answer(request.query, k)
    except RetrievalError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    except GenerationError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        citations=[
            CitationResponse(
                source_id=source.source_id,
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                source_name=source.source_name,
                source_type=source.source_type,
                page_number=source.page_number,
                section_title=source.section_title,
                repo_relative_path=source.repo_relative_path,
            )
            for source in result.citations
        ],
        retrieval=QueryRetrievalMetadata(
            snapshot_version=result.snapshot_version,
            retrieved_chunk_ids=result.retrieved_chunk_ids,
            context_chunk_ids=result.context_chunk_ids,
            fusion_strategy="rrf",
        ),
        model=result.model,
    )
