"""Application-owned source labels and verified citation extraction."""

import logging
import re
from dataclasses import dataclass
from uuid import UUID

from app.retrieval.service import HybridSearchHit

logger = logging.getLogger(__name__)
CITATION_PATTERN = re.compile(r"\[(SOURCE_[1-9][0-9]*)\]")


@dataclass(frozen=True, slots=True)
class LabeledSource:
    source_id: str
    document_id: UUID
    chunk_id: UUID
    text: str
    source_name: str
    source_type: str
    page_number: int | None
    section_title: str | None
    repo_relative_path: str | None = None


def assign_source_labels(hits: list[HybridSearchHit]) -> list[LabeledSource]:
    return [
        LabeledSource(
            source_id=f"SOURCE_{index}",
            document_id=hit.document_id,
            chunk_id=hit.chunk_id,
            text=hit.text,
            source_name=hit.source_name,
            source_type=hit.source_type,
            page_number=hit.page_number,
            section_title=hit.section_title,
        )
        for index, hit in enumerate(hits, start=1)
    ]


def extract_verified_citations(
    answer: str, sources: list[LabeledSource]
) -> list[LabeledSource]:
    available = {source.source_id: source for source in sources}
    citations: list[LabeledSource] = []
    seen: set[str] = set()
    for source_id in CITATION_PATTERN.findall(answer):
        if source_id in seen:
            continue
        seen.add(source_id)
        source = available.get(source_id)
        if source is None:
            logger.warning("generation_invalid_source_label label=%s", source_id)
            continue
        citations.append(source)
    return citations
