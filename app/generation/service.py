"""Hybrid retrieval to bounded context to grounded generation orchestration."""

from dataclasses import dataclass, replace
from uuid import UUID

from app.generation.base import GenerationError, GenerationProvider
from app.generation.citations import (
    LabeledSource,
    assign_source_labels,
    extract_verified_citations,
)
from app.generation.prompts import (
    SYSTEM_PROMPT,
    build_citation_repair_prompt,
    build_user_prompt,
    render_source,
)
from app.generation.sanitizer import sanitize_context_text
from app.retrieval.service import RetrievalService

INSUFFICIENT_CONTEXT_ANSWER = (
    "The indexed knowledge base does not contain enough information to answer this question."
)
UNCITED_ANSWER_ERROR = (
    "Unable to produce a properly cited answer from the supplied context"
)


@dataclass(frozen=True, slots=True)
class GroundedQueryResult:
    query: str
    answer: str
    citations: list[LabeledSource]
    snapshot_version: int
    retrieved_chunk_ids: list[UUID]
    context_chunk_ids: list[UUID]
    model: str


class GroundedGenerationService:
    def __init__(
        self,
        retrieval: RetrievalService,
        provider: GenerationProvider,
        *,
        dense_candidate_k: int,
        sparse_candidate_k: int,
        max_context_chunks: int,
        max_context_chars: int,
    ) -> None:
        self.retrieval = retrieval
        self.provider = provider
        self.dense_candidate_k = dense_candidate_k
        self.sparse_candidate_k = sparse_candidate_k
        self.max_context_chunks = max_context_chunks
        self.max_context_chars = max_context_chars

    def answer(self, query: str, k: int) -> GroundedQueryResult:
        version, hits = self.retrieval.search_hybrid(
            query, k, self.dense_candidate_k, self.sparse_candidate_k
        )
        retrieved_chunk_ids = [hit.chunk_id for hit in hits]
        sanitized_hits = [
            replace(hit, text=sanitized)
            for hit in hits
            if (sanitized := sanitize_context_text(hit.text)).strip()
        ]
        sources = self._select_context(assign_source_labels(sanitized_hits))
        if not sources:
            return GroundedQueryResult(
                query=query,
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                citations=[],
                snapshot_version=version,
                retrieved_chunk_ids=retrieved_chunk_ids,
                context_chunk_ids=[],
                model=self.provider.model,
            )

        answer = self.provider.generate(SYSTEM_PROMPT, build_user_prompt(query, sources))
        citations = extract_verified_citations(answer, sources)
        if not citations and not _is_insufficient_context_answer(answer):
            answer = self.provider.generate(
                SYSTEM_PROMPT,
                build_citation_repair_prompt(query, sources, answer),
            )
            citations = extract_verified_citations(answer, sources)
            if not citations and not _is_insufficient_context_answer(answer):
                raise GenerationError(UNCITED_ANSWER_ERROR)
        return GroundedQueryResult(
            query=query,
            answer=answer,
            citations=citations,
            snapshot_version=version,
            retrieved_chunk_ids=retrieved_chunk_ids,
            context_chunk_ids=[source.chunk_id for source in sources],
            model=self.provider.model,
        )

    def _select_context(self, sources: list[LabeledSource]) -> list[LabeledSource]:
        selected: list[LabeledSource] = []
        used_chars = 0
        for source in sources:
            if len(selected) >= self.max_context_chunks:
                break
            block_length = len(render_source(source))
            separator_length = 2 if selected else 0
            if used_chars + separator_length + block_length > self.max_context_chars:
                break
            selected.append(source)
            used_chars += separator_length + block_length
        return selected


def _is_insufficient_context_answer(answer: str) -> bool:
    normalized = " ".join(answer.lower().split())
    if "insufficient context" in normalized:
        return True
    if "does not contain enough information" in normalized:
        return True
    lacks_answer = "cannot answer" in normalized or "unable to answer" in normalized
    return lacks_answer and ("context" in normalized or "knowledge base" in normalized)
