"""Deterministic Reciprocal Rank Fusion over 1-based rankings."""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    chunk_id: UUID
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class FusedResult:
    chunk_id: UUID
    rrf_score: float
    dense_rank: int | None
    dense_score: float | None
    sparse_rank: int | None
    sparse_score: float | None


def reciprocal_rank_fusion(
    dense: list[RankedCandidate], sparse: list[RankedCandidate], rrf_k: int
) -> list[FusedResult]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    _validate_ranks(dense)
    _validate_ranks(sparse)
    dense_by_id = {item.chunk_id: item for item in dense}
    sparse_by_id = {item.chunk_id: item for item in sparse}
    if len(dense_by_id) != len(dense) or len(sparse_by_id) != len(sparse):
        raise ValueError("Rankings must not contain duplicate chunk IDs")

    fused = []
    for chunk_id in dense_by_id.keys() | sparse_by_id.keys():
        dense_item = dense_by_id.get(chunk_id)
        sparse_item = sparse_by_id.get(chunk_id)
        score = 0.0
        if dense_item is not None:
            score += 1.0 / (rrf_k + dense_item.rank)
        if sparse_item is not None:
            score += 1.0 / (rrf_k + sparse_item.rank)
        fused.append(
            FusedResult(
                chunk_id=chunk_id,
                rrf_score=score,
                dense_rank=dense_item.rank if dense_item else None,
                dense_score=dense_item.score if dense_item else None,
                sparse_rank=sparse_item.rank if sparse_item else None,
                sparse_score=sparse_item.score if sparse_item else None,
            )
        )

    infinity = len(dense) + len(sparse) + 1
    fused.sort(
        key=lambda item: (
            -item.rrf_score,
            min(item.dense_rank or infinity, item.sparse_rank or infinity),
            item.dense_rank or infinity,
            str(item.chunk_id),
        )
    )
    return fused


def _validate_ranks(items: list[RankedCandidate]) -> None:
    if any(item.rank < 1 for item in items):
        raise ValueError("RRF rankings must use 1-based positive ranks")
