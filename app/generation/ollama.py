"""Direct non-streaming Ollama chat client."""

import logging

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
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._owns_client = client is None
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

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaGenerationProvider":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
