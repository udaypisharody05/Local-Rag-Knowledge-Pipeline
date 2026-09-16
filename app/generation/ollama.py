"""Direct non-streaming Ollama chat client."""

import logging
import json
from collections.abc import AsyncIterator

import httpx

from app.generation.base import GenerationError

logger = logging.getLogger(__name__)


class OllamaGenerationProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        timeout_seconds: float,
        *,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._owns_client = client is None
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._async_client = async_client
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException:
            logger.warning("ollama_generation_timeout model=%s", self.model)
            raise GenerationError("Local generation service timed out") from None
        except (httpx.HTTPError, ValueError):
            logger.exception("ollama_generation_request_failed model=%s", self.model)
            raise GenerationError("Local generation service is unavailable") from None

        message = payload.get("message") if isinstance(payload, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("Local generation service returned an invalid response")
        return content.strip()

    async def stream(
        self, system_prompt: str, user_prompt: str
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        if self._async_client is not None:
            async for text in self._stream_with_client(self._async_client, payload):
                yield text
            return
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout_seconds
        ) as client:
            async for text in self._stream_with_client(client, payload):
                yield text

    async def _stream_with_client(
        self, client: httpx.AsyncClient, payload: dict[str, object]
    ) -> AsyncIterator[str]:
        saw_event = False
        completed = False
        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        raise GenerationError(
                            "Local generation service returned an invalid stream"
                        ) from None
                    if not isinstance(event, dict) or event.get("error"):
                        raise GenerationError("Local generation service is unavailable")
                    message = event.get("message")
                    content = message.get("content") if isinstance(message, dict) else None
                    if not isinstance(content, str):
                        raise GenerationError(
                            "Local generation service returned an invalid stream"
                        )
                    saw_event = True
                    if content:
                        yield content
                    if event.get("done") is True:
                        completed = True
                        break
        except httpx.TimeoutException:
            logger.warning("ollama_generation_stream_timeout model=%s", self.model)
            raise GenerationError("Local generation service timed out") from None
        except httpx.HTTPError:
            logger.exception("ollama_generation_stream_failed model=%s", self.model)
            raise GenerationError("Local generation service is unavailable") from None
        if not saw_event or not completed:
            raise GenerationError("Local generation service returned an invalid stream")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaGenerationProvider":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
