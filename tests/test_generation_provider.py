"""Ollama generation client tests without a live model."""

import json

import httpx
import pytest

from app.generation import GenerationError, OllamaGenerationProvider


def _provider(handler: httpx.MockTransport) -> OllamaGenerationProvider:
    return OllamaGenerationProvider(
        "http://ollama.test",
        "local-test-model",
        temperature=0.15,
        timeout_seconds=1,
        client=httpx.Client(transport=handler, base_url="http://ollama.test"),
    )


def test_ollama_generation_request_and_response() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": " Grounded answer. "}})

    answer = _provider(httpx.MockTransport(handler)).generate("system", "user")
    assert answer == "Grounded answer."
    assert captured == {
        "model": "local-test-model",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "stream": False,
        "options": {"temperature": 0.15},
    }


def test_ollama_generation_timeout_is_sanitized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private endpoint detail", request=request)

    with pytest.raises(GenerationError, match="timed out") as error:
        _provider(httpx.MockTransport(handler)).generate("system", "user")
    assert "private endpoint detail" not in str(error.value)


def test_ollama_generation_http_error_is_sanitized() -> None:
    provider = _provider(httpx.MockTransport(lambda _: httpx.Response(404, text="secret")))
    with pytest.raises(GenerationError, match="unavailable") as error:
        provider.generate("system", "user")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("payload", [{}, {"message": {}}, {"message": {"content": "  "}}])
def test_ollama_generation_malformed_response_is_rejected(payload: dict) -> None:
    provider = _provider(httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    with pytest.raises(GenerationError, match="invalid response"):
        provider.generate("system", "user")
