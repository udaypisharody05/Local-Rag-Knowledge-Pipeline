"""Protected non-streaming and SSE grounded question-answering endpoints."""

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_db,
    get_embedding_provider,
    get_generation_provider,
)
from app.core.config import settings
from app.core.security import require_api_key
from app.embeddings import EmbeddingProvider
from app.generation import (
    GenerationError,
    GenerationProvider,
    GroundedGenerationService,
    INSUFFICIENT_CONTEXT_ANSWER,
    PreparedGroundedQuery,
    is_insufficient_context_answer,
)
from app.generation.citations import LabeledSource, extract_verified_citations
from app.generation.sse import encode_sse_event
from app.retrieval import snapshot_manager
from app.retrieval.service import RetrievalError, RetrievalService
from app.schemas.query import (
    CitationResponse,
    QueryRequest,
    QueryResponse,
    QueryRetrievalMetadata,
)

router = APIRouter(tags=["generation"], dependencies=[Depends(require_api_key)])
logger = logging.getLogger(__name__)


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
    service = _service(db, embedding_provider, generation_provider)
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


@router.post("/query/stream", response_model=None)
def stream_query_knowledge_base(
    body: QueryRequest,
    request: Request,
    db: Session = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    generation_provider: GenerationProvider = Depends(get_generation_provider),
) -> StreamingResponse | JSONResponse:
    k = body.k if body.k is not None else settings.hybrid_top_k
    if k > settings.hybrid_max_k:
        return JSONResponse(
            status_code=422,
            content={"detail": f"k must be less than or equal to {settings.hybrid_max_k}"},
        )
    service = _service(db, embedding_provider, generation_provider)
    try:
        prepared = service.prepare(body.query, k)
    except RetrievalError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return StreamingResponse(
        stream_query_events(request, service, prepared),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


async def stream_query_events(
    request: Request,
    service: GroundedGenerationService,
    prepared: PreparedGroundedQuery,
) -> AsyncIterator[str]:
    yield encode_sse_event(
        "start",
        {
            "query": prepared.query,
            "snapshot_version": prepared.snapshot_version,
            "model": prepared.model,
        },
    )
    if await request.is_disconnected():
        logger.info("query_stream_client_disconnected")
        return

    if not prepared.sources:
        yield encode_sse_event("token", {"text": INSUFFICIENT_CONTEXT_ANSWER})
        yield encode_sse_event("citations", {"citations": []})
        yield encode_sse_event("metadata", _metadata_payload(prepared))
        yield encode_sse_event("done", {})
        return

    tokens: list[str] = []
    upstream = service.stream(prepared)
    try:
        async for token in upstream:
            if await request.is_disconnected():
                logger.info("query_stream_client_disconnected")
                return
            tokens.append(token)
            yield encode_sse_event("token", {"text": token})
    except asyncio.CancelledError:
        logger.info("query_stream_client_disconnected")
        return
    except GenerationError:
        if not await request.is_disconnected():
            yield encode_sse_event(
                "error", {"detail": "Generation service unavailable."}
            )
        return
    except Exception:
        logger.exception("query_stream_generation_failed")
        if not await request.is_disconnected():
            yield encode_sse_event(
                "error", {"detail": "Generation service unavailable."}
            )
        return
    finally:
        close = getattr(upstream, "aclose", None)
        if close is not None:
            await close()

    answer = "".join(tokens)
    citations = extract_verified_citations(answer, prepared.sources)
    if not citations and not is_insufficient_context_answer(answer):
        yield encode_sse_event(
            "error",
            {
                "detail": (
                    "The generated response could not be validated with source citations."
                )
            },
        )
        return
    yield encode_sse_event(
        "citations",
        {"citations": [_citation_payload(source) for source in citations]},
    )
    yield encode_sse_event("metadata", _metadata_payload(prepared))
    yield encode_sse_event("done", {})


def _service(
    db: Session,
    embedding_provider: EmbeddingProvider,
    generation_provider: GenerationProvider,
) -> GroundedGenerationService:
    retrieval = RetrievalService(
        db,
        embedding_provider,
        settings.index_storage_root,
        snapshot_manager,
        settings.rrf_k,
    )
    return GroundedGenerationService(
        retrieval,
        generation_provider,
        dense_candidate_k=settings.dense_candidate_k,
        sparse_candidate_k=settings.sparse_candidate_k,
        max_context_chunks=settings.generation_max_context_chunks,
        max_context_chars=settings.generation_max_context_chars,
    )


def _citation_payload(source: LabeledSource) -> dict[str, object]:
    return {
        "source_id": source.source_id,
        "document_id": str(source.document_id),
        "chunk_id": str(source.chunk_id),
        "source_name": source.source_name,
        "source_type": source.source_type,
        "page_number": source.page_number,
        "section_title": source.section_title,
        "repo_relative_path": source.repo_relative_path,
    }


def _metadata_payload(prepared: PreparedGroundedQuery) -> dict[str, object]:
    return {
        "snapshot_version": prepared.snapshot_version,
        "retrieved_chunk_ids": [str(value) for value in prepared.retrieved_chunk_ids],
        "context_chunk_ids": [str(value) for value in prepared.context_chunk_ids],
        "fusion_strategy": "rrf",
        "model": prepared.model,
    }
