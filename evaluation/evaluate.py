"""Run deterministic retrieval and grounding checks against a live Local RAG API."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import httpx

SOURCE_PATTERN = re.compile(r"\[(SOURCE_\d+)\]")


class EvaluationError(ValueError):
    """Raised when an evaluation input or API response is malformed."""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    category: str
    query: str
    expected_source_names: tuple[str, ...]
    forbidden_source_names: tuple[str, ...]
    should_answer: bool
    expected_answer_terms: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationCase":
        required = {"id", "category", "query", "should_answer"}
        if not required.issubset(value):
            raise EvaluationError(
                f"Evaluation case is missing fields: {sorted(required - value.keys())}"
            )
        if not all(isinstance(value[name], str) and value[name] for name in ("id", "category", "query")):
            raise EvaluationError("Evaluation case id, category, and query must be non-empty strings")
        if not isinstance(value["should_answer"], bool):
            raise EvaluationError("Evaluation case should_answer must be a boolean")

        def strings(name: str) -> tuple[str, ...]:
            items = value.get(name, [])
            if not isinstance(items, list) or not all(
                isinstance(item, str) and item for item in items
            ):
                raise EvaluationError(f"Evaluation case {name} must be a list of strings")
            return tuple(items)

        return cls(
            id=value["id"],
            category=value["category"],
            query=value["query"],
            expected_source_names=strings("expected_source_names"),
            forbidden_source_names=strings("forbidden_source_names"),
            should_answer=value["should_answer"],
            expected_answer_terms=strings("expected_answer_terms"),
        )


def retrieval_metrics(
    retrieved_source_names: Sequence[str], expected_source_names: Sequence[str]
) -> dict[str, float | bool | None]:
    expected = tuple(dict.fromkeys(expected_source_names))
    if not expected:
        return {"hit_at_k": None, "mrr": None, "recall_at_k": None}
    expected_set = set(expected)
    first_rank = next(
        (
            rank
            for rank, source_name in enumerate(retrieved_source_names, start=1)
            if source_name in expected_set
        ),
        None,
    )
    found = expected_set.intersection(retrieved_source_names)
    return {
        "hit_at_k": first_rank is not None,
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
        "recall_at_k": len(found) / len(expected_set),
    }


def generation_metrics(
    case: EvaluationCase, response: dict[str, Any]
) -> dict[str, bool | None]:
    if not isinstance(response, dict):
        raise EvaluationError("Query response must be an object")
    answer = response.get("answer")
    citations = response.get("citations")
    retrieval = response.get("retrieval")
    if not isinstance(answer, str) or not isinstance(citations, list) or not isinstance(
        retrieval, dict
    ):
        raise EvaluationError("Query response is missing answer, citations, or retrieval metadata")
    if not all(isinstance(item, dict) for item in citations):
        raise EvaluationError("Query response citations must be objects")

    citation_ids: list[str] = []
    citation_sources: list[str] = []
    for citation in citations:
        source_id = citation.get("source_id")
        source_name = citation.get("source_name")
        if not isinstance(source_id, str) or not isinstance(source_name, str):
            raise EvaluationError("Citation metadata is missing source_id or source_name")
        citation_ids.append(source_id)
        citation_sources.append(source_name)

    answer_ids = set(SOURCE_PATTERN.findall(answer))
    metadata_ids = set(citation_ids)
    identifiers_match = (
        len(citation_ids) == len(metadata_ids) and answer_ids == metadata_ids
    )
    expected_sources = set(case.expected_source_names)
    forbidden_sources = set(case.forbidden_source_names)
    lowered_answer = answer.casefold()
    terms_present = (
        None
        if not case.expected_answer_terms
        else all(term.casefold() in lowered_answer for term in case.expected_answer_terms)
    )
    expected_source_cited = (
        None
        if not expected_sources
        else bool(expected_sources.intersection(citation_sources))
    )
    forbidden_sources_absent = (
        None
        if not forbidden_sources
        else not forbidden_sources.intersection(citation_sources)
    )

    if case.should_answer:
        citation_compliance = bool(citations) and identifiers_match
        refusal_no_citations = None
    else:
        citation_compliance = None
        refusal_no_citations = not citations and not answer_ids

    return {
        "citation_compliance": citation_compliance,
        "citation_ids_match_metadata": identifiers_match,
        "expected_source_cited": expected_source_cited,
        "expected_terms_present": terms_present,
        "refusal_no_citations": refusal_no_citations,
        "forbidden_sources_absent": forbidden_sources_absent,
    }


def parse_retrieval_sources(response: dict[str, Any]) -> list[str]:
    if not isinstance(response, dict):
        raise EvaluationError("Hybrid search response must be an object")
    results = response.get("results")
    if not isinstance(results, list):
        raise EvaluationError("Hybrid search response is missing results")
    sources: list[str] = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("source_name"), str):
            raise EvaluationError("Hybrid search result is missing source_name")
        sources.append(result["source_name"])
    return sources


def load_dataset(path: Path) -> list[EvaluationCase]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not load evaluation dataset: {type(exc).__name__}") from None
    if not isinstance(payload, list):
        raise EvaluationError("Evaluation dataset must be a JSON array")
    cases = [EvaluationCase.from_dict(value) for value in payload if isinstance(value, dict)]
    if len(cases) != len(payload):
        raise EvaluationError("Every evaluation dataset item must be an object")
    if len({case.id for case in cases}) != len(cases):
        raise EvaluationError("Evaluation case IDs must be unique")
    return cases


class RAGEvaluator:
    def __init__(self, base_url: str, api_key: str, *, k: int, timeout: float) -> None:
        self.k = k
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    def close(self) -> None:
        self.client.close()

    def evaluate(self, case: EvaluationCase) -> dict[str, Any]:
        retrieval_started = time.perf_counter()
        retrieval_response = self.client.post(
            "/search/hybrid", json={"query": case.query, "k": self.k}
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        retrieval_response.raise_for_status()
        retrieval_payload = retrieval_response.json()
        sources = parse_retrieval_sources(retrieval_payload)
        retrieval = retrieval_metrics(sources, case.expected_source_names)
        retrieval["forbidden_sources_absent"] = (
            None
            if not case.forbidden_source_names
            else not set(case.forbidden_source_names).intersection(sources)
        )

        query_started = time.perf_counter()
        query_response = self.client.post(
            "/query", json={"query": case.query, "k": self.k}
        )
        query_ms = (time.perf_counter() - query_started) * 1000
        query_response.raise_for_status()
        query_payload = query_response.json()
        generation = generation_metrics(case, query_payload)
        return {
            "case": asdict(case),
            "retrieved_source_names": sources,
            "retrieval": retrieval,
            "generation": generation,
            "latency_ms": {
                "hybrid_retrieval": round(retrieval_ms, 2),
                "query": round(query_ms, 2),
            },
        }


def aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def average(path: tuple[str, str]) -> float | None:
        values = [
            record[path[0]][path[1]]
            for record in records
            if record.get("error") is None
            and record[path[0]].get(path[1]) is not None
        ]
        return None if not values else round(mean(float(value) for value in values), 4)

    successful = [record for record in records if record.get("error") is None]
    return {
        "cases": len(records),
        "successful_cases": len(successful),
        "failed_cases": len(records) - len(successful),
        "hit_at_k": average(("retrieval", "hit_at_k")),
        "mrr": average(("retrieval", "mrr")),
        "recall_at_k": average(("retrieval", "recall_at_k")),
        "retrieval_exclusion_rate": average(
            ("retrieval", "forbidden_sources_absent")
        ),
        "citation_compliance_rate": average(
            ("generation", "citation_compliance")
        ),
        "expected_source_citation_rate": average(
            ("generation", "expected_source_cited")
        ),
        "expected_terms_rate": average(("generation", "expected_terms_present")),
        "refusal_no_citation_rate": average(
            ("generation", "refusal_no_citations")
        ),
        "generation_exclusion_rate": average(
            ("generation", "forbidden_sources_absent")
        ),
        "average_hybrid_retrieval_ms": average(
            ("latency_ms", "hybrid_retrieval")
        ),
        "average_query_ms": average(("latency_ms", "query")),
    }


def run(cases: Sequence[EvaluationCase], evaluator: RAGEvaluator) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case in cases:
        try:
            record = evaluator.evaluate(case)
            record["error"] = None
        except (EvaluationError, httpx.HTTPError, ValueError) as exc:
            record = {
                "case": asdict(case),
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
    return {"summary": aggregate(records), "results": records}


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Local RAG evaluation summary")
    print(f"Cases: {summary['successful_cases']}/{summary['cases']} completed")
    for name in (
        "hit_at_k",
        "mrr",
        "recall_at_k",
        "citation_compliance_rate",
        "expected_source_citation_rate",
        "refusal_no_citation_rate",
    ):
        print(f"{name}: {summary[name]}")
    print(f"average_hybrid_retrieval_ms: {summary['average_hybrid_retrieval_ms']}")
    print(f"average_query_ms: {summary['average_query_ms']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("dataset.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "latest.json",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("RAG_API_BASE_URL", "http://localhost:8000"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    api_key = os.getenv("RAG_API_KEY")
    if not api_key:
        parser.error("RAG_API_KEY must be set in the environment")
    if args.k <= 0:
        parser.error("--k must be positive")

    cases = load_dataset(args.dataset)
    evaluator = RAGEvaluator(args.base_url, api_key, k=args.k, timeout=args.timeout)
    try:
        report = run(cases, evaluator)
    finally:
        evaluator.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(report)
    return 0 if report["summary"]["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
