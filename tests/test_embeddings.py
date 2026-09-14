"""Ollama embedding client contract tests with no live server."""

import json

import httpx
import pytest

from app.embeddings import EmbeddingError, OllamaEmbeddingProvider


def _provider(handler: httpx.MockTransport) -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider(
        "http://ollama.test",
        "test-model",
        batch_size=2,
        timeout_seconds=1,
        client=httpx.Client(transport=handler, base_url="http://ollama.test"),
    )


def test_ollama_request_and_successful_batch_parsing() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={"embeddings": [[float(len(value)), 1.0] for value in body["input"]]},
        )

    provider = _provider(httpx.MockTransport(handler))
    vectors = provider.embed_documents(["one", "three", "seven"])

    assert vectors == [[3.0, 1.0], [5.0, 1.0], [5.0, 1.0]]
    assert requests == [
        {"model": "test-model", "input": ["one", "three"]},
        {"model": "test-model", "input": ["seven"]},
    ]


def test_ollama_timeout_is_sanitized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private transport detail", request=request)

    with pytest.raises(EmbeddingError, match="timed out") as error:
        _provider(httpx.MockTransport(handler)).embed_query("query")
    assert "private transport detail" not in str(error.value)


def test_ollama_http_error_is_sanitized() -> None:
    provider = _provider(httpx.MockTransport(lambda _: httpx.Response(500, text="secret")))
    with pytest.raises(EmbeddingError, match="unavailable") as error:
        provider.embed_query("query")
    assert "secret" not in str(error.value)


def test_ollama_embedding_count_mismatch_is_rejected() -> None:
    provider = _provider(
        httpx.MockTransport(lambda _: httpx.Response(200, json={"embeddings": [[1.0, 0.0]]}))
    )
    with pytest.raises(EmbeddingError, match="count"):
        provider.embed_documents(["one", "two"])


def test_ollama_dimension_mismatch_is_rejected() -> None:
    provider = _provider(
        httpx.MockTransport(
            lambda _: httpx.Response(200, json={"embeddings": [[1.0, 0.0], [1.0]]})
        )
    )
    with pytest.raises(EmbeddingError, match="dimensions"):
        provider.embed_documents(["one", "two"])
