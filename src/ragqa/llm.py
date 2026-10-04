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


def get_llm(name: str | None = None, model: str | None = None) -> LLM:
    name = (name or os.getenv("RAGQA_LLM", "ollama")).lower()
    if name == "ollama":
        return OllamaLLM(model or os.getenv("RAGQA_MODEL", "llama3.2:3b"))
    raise ValueError(f"unknown llm: {name}")
