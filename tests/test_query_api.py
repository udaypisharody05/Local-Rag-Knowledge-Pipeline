"""Grounded query API behavior independent of PostgreSQL and live Ollama."""

import asyncio
import json
from dataclasses import dataclass, field
from uuid import UUID

from fastapi.testclient import TestClient

from app.api.dependencies import get_db, get_embedding_provider, get_generation_provider
from app.core.config import settings
from app.generation.citations import LabeledSource
from app.generation.service import (
    GroundedGenerationService,
    GroundedQueryResult,
    PreparedGroundedQuery,
)
from app.main import app
from app.api.query import stream_query_events


@dataclass
class FakeProvider:
    model: str = "fake-model"


@dataclass
class FakeStreamingProvider:
    tokens: list[str]
    model: str = "fake-stream-model"
    fail_after: int | None = None
    calls: int = 0
    closed: bool = False

    async def stream(self, system_prompt: str, user_prompt: str):
        self.calls += 1
        try:
            for index, token in enumerate(self.tokens):
                if self.fail_after == index:
                    from app.generation import GenerationError

                    raise GenerationError("private failure")
                yield token
            if self.fail_after == len(self.tokens):
                from app.generation import GenerationError

                raise GenerationError("private failure")
        finally:
            self.closed = True


def _source(value: int = 1) -> LabeledSource:
    return LabeledSource(
        source_id=f"SOURCE_{value}",
        document_id=UUID(int=100 + value),
        chunk_id=UUID(int=value),
        text="verified context",
        source_name="facts.txt",
        source_type="txt",
        page_number=None,
        section_title=None,
    )


def _prepared(*, sources=None) -> PreparedGroundedQuery:
    selected = [_source()] if sources is None else sources
    return PreparedGroundedQuery(
        query="question",
        snapshot_version=6,
        retrieved_chunk_ids=[UUID(int=1), UUID(int=2)],
        sources=selected,
        model="fake-stream-model",
    )


def _events(response) -> list[tuple[str, dict]]:
    parsed = []
    for block in response.text.strip().split("\n\n"):
        lines = block.splitlines()
        parsed.append((lines[0].removeprefix("event: "), json.loads(lines[1][6:])))
    return parsed


def _set_stream_overrides(provider: FakeStreamingProvider, monkeypatch, prepared) -> None:
    monkeypatch.setattr(GroundedGenerationService, "prepare", lambda self, query, k: prepared)
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_embedding_provider] = lambda: FakeProvider()
    app.dependency_overrides[get_generation_provider] = lambda: provider


def _clear_stream_overrides() -> None:
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_embedding_provider, None)
    app.dependency_overrides.pop(get_generation_provider, None)


def test_query_requires_authentication(client: TestClient) -> None:
    assert client.post("/query", json={"query": "question"}).status_code == 401
    assert client.post("/query/stream", json={"query": "question"}).status_code == 401


def test_query_k_limit_is_enforced(client: TestClient, valid_api_key: str) -> None:
    response = client.post(
        "/query",
        headers={"X-API-Key": valid_api_key},
        json={"query": "question", "k": settings.hybrid_max_k + 1},
    )
    assert response.status_code == 422


def test_query_returns_structured_generation_response(
    client: TestClient, valid_api_key: str, monkeypatch
) -> None:
    def fake_answer(self, query: str, k: int) -> GroundedQueryResult:
        return GroundedQueryResult(
            query=query,
            answer="Not enough verified detail.",
            citations=[],
            snapshot_version=4,
            retrieved_chunk_ids=[],
            context_chunk_ids=[],
            model="fake-model",
        )

    monkeypatch.setattr(GroundedGenerationService, "answer", fake_answer)
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_embedding_provider] = lambda: FakeProvider()
    app.dependency_overrides[get_generation_provider] = lambda: FakeProvider()
    try:
        response = client.post(
            "/query",
            headers={"X-API-Key": valid_api_key},
            json={"query": "question"},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_generation_provider, None)
    assert response.status_code == 200
    assert response.json() == {
        "query": "question",
        "answer": "Not enough verified detail.",
        "citations": [],
        "retrieval": {
            "snapshot_version": 4,
            "retrieved_chunk_ids": [],
            "context_chunk_ids": [],
            "fusion_strategy": "rrf",
        },
        "model": "fake-model",
    }


def test_successful_sse_stream_has_tokens_verified_citations_metadata_and_done(
    client: TestClient, valid_api_key: str, monkeypatch
) -> None:
    provider = FakeStreamingProvider(
        ["The", " answer [SOURCE_1] and invalid [SOURCE_99]."]
    )
    _set_stream_overrides(provider, monkeypatch, _prepared())
    try:
        response = client.post(
            "/query/stream",
            headers={"X-API-Key": valid_api_key},
            json={"query": "question", "k": 2},
        )
    finally:
        _clear_stream_overrides()
    events = _events(response)
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [name for name, _ in events] == [
        "start", "token", "token", "citations", "metadata", "done"
    ]
    assert "".join(payload["text"] for name, payload in events if name == "token") == (
        "The answer [SOURCE_1] and invalid [SOURCE_99]."
    )
    citations = next(payload["citations"] for name, payload in events if name == "citations")
    assert [item["source_id"] for item in citations] == ["SOURCE_1"]
    metadata = next(payload for name, payload in events if name == "metadata")
    assert metadata["snapshot_version"] == 6
    assert metadata["retrieved_chunk_ids"] == [str(UUID(int=1)), str(UUID(int=2))]
    assert metadata["context_chunk_ids"] == [str(UUID(int=1))]
    assert metadata["fusion_strategy"] == "rrf"
    assert metadata["model"] == "fake-stream-model"
    assert provider.closed is True


def test_uncited_stream_ends_with_error_without_done(
    client: TestClient, valid_api_key: str, monkeypatch
) -> None:
    provider = FakeStreamingProvider(["Uncited factual answer."])
    _set_stream_overrides(provider, monkeypatch, _prepared())
    try:
        response = client.post(
            "/query/stream", headers={"X-API-Key": valid_api_key}, json={"query": "question"}
        )
    finally:
        _clear_stream_overrides()
    events = _events(response)
    assert [name for name, _ in events][-1] == "error"
    assert not any(name == "done" for name, _ in events)


def test_uncited_refusal_stream_completes_with_empty_citations(
    client: TestClient, valid_api_key: str, monkeypatch
) -> None:
    provider = FakeStreamingProvider(
        ["The indexed knowledge base does not contain enough information to answer."]
    )
    _set_stream_overrides(provider, monkeypatch, _prepared())
    try:
        response = client.post(
            "/query/stream", headers={"X-API-Key": valid_api_key}, json={"query": "question"}
        )
    finally:
        _clear_stream_overrides()
    events = _events(response)
    assert events[-1][0] == "done"
    assert next(payload for name, payload in events if name == "citations") == {"citations": []}


def test_no_context_stream_skips_provider_and_returns_deterministic_sequence(
    client: TestClient, valid_api_key: str, monkeypatch
) -> None:
    provider = FakeStreamingProvider(["must not stream"])
    _set_stream_overrides(provider, monkeypatch, _prepared(sources=[]))
    try:
        response = client.post(
            "/query/stream", headers={"X-API-Key": valid_api_key}, json={"query": "question"}
        )
    finally:
        _clear_stream_overrides()
    events = _events(response)
    assert [name for name, _ in events] == ["start", "token", "citations", "metadata", "done"]
    assert provider.calls == 0


def test_stream_failure_before_and_after_partial_output_is_sanitized(
    client: TestClient, valid_api_key: str, monkeypatch
) -> None:
    for tokens, fail_after, expected_tokens in (([], 0, 0), (["partial"], 1, 1)):
        provider = FakeStreamingProvider(tokens, fail_after=fail_after)
        _set_stream_overrides(provider, monkeypatch, _prepared())
        try:
            response = client.post(
                "/query/stream",
                headers={"X-API-Key": valid_api_key},
                json={"query": "question"},
            )
        finally:
            _clear_stream_overrides()
        events = _events(response)
        assert sum(name == "token" for name, _ in events) == expected_tokens
        assert events[-1] == ("error", {"detail": "Generation service unavailable."})
        assert not any(name in {"citations", "done"} for name, _ in events)


def test_disconnect_stops_and_closes_upstream_generator() -> None:
    provider = FakeStreamingProvider(["first", "second", "third"])
    service = GroundedGenerationService(
        retrieval=object(),
        provider=provider,
        dense_candidate_k=20,
        sparse_candidate_k=20,
        max_context_chunks=5,
        max_context_chars=12_000,
    )

    class DisconnectAfterFirstToken:
        calls = 0

        async def is_disconnected(self) -> bool:
            self.calls += 1
            return self.calls >= 3

    async def consume() -> list[str]:
        return [
            event
            async for event in stream_query_events(
                DisconnectAfterFirstToken(), service, _prepared()
            )
        ]

    encoded = asyncio.run(consume())
    assert sum(event.startswith("event: token") for event in encoded) == 1
    assert not any(event.startswith("event: done") for event in encoded)
    assert provider.closed is True
