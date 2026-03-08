from __future__ import annotations

from dataclasses import dataclass
import json
from json import JSONDecodeError
from typing import Any, Protocol
import urllib.error
import urllib.request

from treeqa.config import TreeQASettings


class LLMClient(Protocol):
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        ...

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | list[Any]:
        ...


@dataclass(slots=True)
class OpenAICompatibleLLMClient:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 30
    temperature: float = 0.0
    extra_headers: dict[str, str] | None = None

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = self._post_json(payload)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("LLM response did not include any choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str):
            raise RuntimeError("LLM response content was not text.")
        return content.strip()

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | list[Any]:
        text = self.generate_text(system_prompt, user_prompt)
        parsed = self._parse_json_payload(text)
        if parsed is None:
            raise RuntimeError("LLM response did not contain valid JSON.")
        return parsed

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.extra_headers:
            headers.update(self.extra_headers)
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error}") from error

        try:
            decoded = json.loads(body)
        except JSONDecodeError as error:
            raise RuntimeError("LLM returned a non-JSON response body.") from error
        if not isinstance(decoded, dict):
            raise RuntimeError("Unexpected LLM response format.")
        return decoded

    def _parse_json_payload(self, text: str) -> dict[str, Any] | list[Any] | None:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except JSONDecodeError:
                continue
            if isinstance(payload, (dict, list)):
                return payload
        return None


def build_llm_client(settings: TreeQASettings) -> LLMClient | None:
    provider = settings.llm_provider.strip().lower()
    if provider in {"", "stub", "none"}:
        return None
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set when TREEQA_LLM_PROVIDER=openai.")
        if not settings.llm_model:
            raise ValueError("TREEQA_LLM_MODEL must be set when TREEQA_LLM_PROVIDER=openai.")
        return OpenAICompatibleLLMClient(
            api_key=settings.openai_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
        )
    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be set when TREEQA_LLM_PROVIDER=openrouter."
            )
        if not settings.llm_model:
            raise ValueError(
                "TREEQA_LLM_MODEL must be set when TREEQA_LLM_PROVIDER=openrouter."
            )
        extra_headers: dict[str, str] = {}
        if settings.openrouter_site_url:
            extra_headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            extra_headers["X-Title"] = settings.openrouter_app_name
        return OpenAICompatibleLLMClient(
            api_key=settings.openrouter_api_key,
            base_url=settings.llm_base_url or "https://openrouter.ai/api/v1",
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
            extra_headers=extra_headers or None,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
