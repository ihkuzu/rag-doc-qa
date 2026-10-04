from __future__ import annotations

import os
from typing import Protocol

import httpx


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


def _normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    # OLLAMA_HOST is often set without a scheme, e.g. "0.0.0.0:11434"
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.replace("//0.0.0.0", "//localhost")


class OllamaLLM:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str | None = None,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.host = _normalize_host(host or os.getenv("OLLAMA_HOST") or "http://localhost:11434")
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            response = self._client.post(f"{self.host}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise LLMError(f"request to Ollama at {self.host} failed: {error}") from error


class GeminiLLM:
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        api_key: str | None = None,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise LLMError("set GEMINI_API_KEY to use the gemini model")
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "system_instruction": system,
            "input": prompt,
            "generation_config": {"thinking_level": "low"},
        }
        try:
            response = self._client.post(
                self.url, json=payload, headers={"x-goog-api-key": self._api_key}
            )
            response.raise_for_status()
            steps = response.json()["steps"]
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:200]
            raise LLMError(f"Gemini returned {error.response.status_code}: {detail}") from error
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise LLMError(f"request to Gemini failed: {error}") from error

        # thought steps are skipped, only the model output carries the answer
        texts = [
            block["text"]
            for step in steps
            if step.get("type") == "model_output"
            for block in step.get("content", [])
            if block.get("type") == "text"
        ]
        if not texts:
            raise LLMError("Gemini response contained no text")
        return "".join(texts).strip()


def get_llm(name: str | None = None, model: str | None = None) -> LLM:
    name = (name or os.getenv("RAGQA_LLM", "ollama")).lower()
    if name == "ollama":
        return OllamaLLM(model or os.getenv("RAGQA_MODEL", "llama3.2:3b"))
    if name == "gemini":
        return GeminiLLM(model or os.getenv("RAGQA_MODEL", "gemini-3.8-flash"))
    raise ValueError(f"unknown llm: {name}")
