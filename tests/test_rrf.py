"""Reciprocal Rank Fusion formula and deterministic ordering tests."""

from uuid import UUID

import pytest

from app.retrieval.fusion import RankedCandidate, reciprocal_rank_fusion


def _id(value: int) -> UUID:
    return UUID(int=value)


def test_rrf_combines_one_based_rank_contributions() -> None:
    dense = [RankedCandidate(_id(1), 1, 0.9), RankedCandidate(_id(2), 2, 0.8)]
    sparse = [RankedCandidate(_id(2), 1, 4.0), RankedCandidate(_id(3), 2, 3.0)]
    results = reciprocal_rank_fusion(dense, sparse, 60)
    by_id = {item.chunk_id: item for item in results}
    assert results[0].chunk_id == _id(2)
    assert by_id[_id(2)].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[_id(1)].rrf_score == pytest.approx(1 / 61)
    assert by_id[_id(3)].dense_rank is None
    assert by_id[_id(1)].sparse_rank is None


def test_rrf_uses_deterministic_best_rank_dense_rank_and_uuid_ties() -> None:
    dense = [RankedCandidate(_id(3), 1, 0.9), RankedCandidate(_id(2), 2, 0.8)]
    sparse = [RankedCandidate(_id(1), 1, 3.0), RankedCandidate(_id(2), 2, 2.0)]
    results = reciprocal_rank_fusion(dense, sparse, 60)
    assert results[0].chunk_id == _id(2)
    # Equal single-list rank-1 scores prefer the dense result.
    assert [item.chunk_id for item in results[1:]] == [_id(3), _id(1)]


def test_rrf_rejects_zero_based_ranks() -> None:
    with pytest.raises(ValueError, match="1-based"):
        reciprocal_rank_fusion([RankedCandidate(_id(1), 0, 1.0)], [], 60)
