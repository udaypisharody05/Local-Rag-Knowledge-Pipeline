"""Unit tests for deterministic evaluation metrics; no live API is required."""

from pathlib import Path

import pytest

from evaluation.evaluate import (
    EvaluationCase,
    EvaluationError,
    generation_metrics,
    load_dataset,
    parse_retrieval_sources,
    retrieval_metrics,
)


def _case(**overrides) -> EvaluationCase:
    values = {
        "id": "case_001",
        "category": "citation_required",
        "query": "How does retrieval work?",
        "expected_source_names": ("dense-retrieval.md",),
        "forbidden_source_names": (),
        "should_answer": True,
        "expected_answer_terms": ("cosine",),
    }
    values.update(overrides)
    return EvaluationCase(**values)


def test_hit_at_k_mrr_and_multi_source_recall() -> None:
    metrics = retrieval_metrics(
        ["other.md", "dense-retrieval.md", "sparse-retrieval.md"],
        ["dense-retrieval.md", "sparse-retrieval.md", "hybrid-retrieval.md"],
    )
    assert metrics == {"hit_at_k": True, "mrr": 0.5, "recall_at_k": 2 / 3}


def test_missing_relevant_source_scores_zero() -> None:
    assert retrieval_metrics(["other.md"], ["expected.md"]) == {
        "hit_at_k": False,
        "mrr": 0.0,
        "recall_at_k": 0.0,
    }


def test_generation_metrics_validate_citations_sources_and_terms() -> None:
    metrics = generation_metrics(
        _case(),
        {
            "answer": "L2 normalization enables cosine comparison [SOURCE_1].",
            "citations": [
                {"source_id": "SOURCE_1", "source_name": "dense-retrieval.md"}
            ],
            "retrieval": {"snapshot_version": 3},
        },
    )
    assert metrics["citation_compliance"] is True
    assert metrics["citation_ids_match_metadata"] is True
    assert metrics["expected_source_cited"] is True
    assert metrics["expected_terms_present"] is True


def test_generation_metrics_reject_unbacked_source_label() -> None:
    metrics = generation_metrics(
        _case(),
        {
            "answer": "A claim [SOURCE_1] plus an invented source [SOURCE_9].",
            "citations": [
                {"source_id": "SOURCE_1", "source_name": "dense-retrieval.md"}
            ],
            "retrieval": {},
        },
    )
    assert metrics["citation_compliance"] is False
    assert metrics["citation_ids_match_metadata"] is False


def test_refusal_scoring_requires_no_citations() -> None:
    metrics = generation_metrics(
        _case(
            should_answer=False,
            expected_source_names=(),
            expected_answer_terms=(),
        ),
        {
            "answer": "I do not have enough verified context to answer.",
            "citations": [],
            "retrieval": {"context_chunk_ids": []},
        },
    )
    assert metrics["citation_compliance"] is None
    assert metrics["refusal_no_citations"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"answer": "missing fields"},
        {"answer": "x", "citations": "bad", "retrieval": {}},
        {"answer": "x", "citations": [{}], "retrieval": {}},
    ],
)
def test_generation_metrics_reject_malformed_responses(payload: dict) -> None:
    with pytest.raises(EvaluationError):
        generation_metrics(_case(), payload)


def test_retrieval_parser_rejects_malformed_results() -> None:
    with pytest.raises(EvaluationError):
        parse_retrieval_sources({"results": [{"text": "missing source"}]})
    with pytest.raises(EvaluationError):
        parse_retrieval_sources([])  # type: ignore[arg-type]


def test_committed_dataset_has_expected_size_and_categories() -> None:
    cases = load_dataset(Path("evaluation/dataset.json"))
    assert len(cases) == 18
    assert {
        "semantic_retrieval",
        "exact_identifier",
        "multi_source",
        "insufficient_context",
        "citation_required",
        "deleted_content_exclusion",
        "prompt_injection_resistance",
    }.issubset({case.category for case in cases})
