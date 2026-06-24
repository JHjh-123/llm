from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    def __init__(self, base_url: str, model: str, api_key: str = "dummy", timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def chat(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        return payload["choices"][0]["message"]["content"]


class OllamaClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 120,
        think: bool = False,
        num_predict: int = 256,
    ) -> None:
        self.base_url = _normalize_ollama_base_url(base_url)
        self.model = model
        self.timeout = timeout
        self.think = think
        self.num_predict = num_predict
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def chat(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": self.think,
                "options": {
                    "temperature": 0,
                    "num_predict": self.num_predict,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        return payload["message"]["content"]


def build_llm_from_env() -> LLMClient:
    backend = os.getenv("LLM_BACKEND")
    if not backend:
        raise ValueError("LLM_BACKEND is required. Supported values: ollama, openai_compatible")
    backend = backend.lower()
    timeout = int(os.getenv("LLM_TIMEOUT", "600"))
    if backend == "ollama":
        return OllamaClient(
            base_url=os.getenv("OLLAMA_BASE_URL", os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434")),
            model=os.getenv("LLM_MODEL", "qwen3:8b"),
            think=_env_bool("OLLAMA_THINK", default=False),
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "256")),
            timeout=timeout,
        )
    if backend == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=os.environ["LLM_BASE_URL"],
            model=os.environ["LLM_MODEL"],
            api_key=os.getenv("LLM_API_KEY", "dummy"),
            timeout=timeout,
        )
    raise ValueError(f"Unsupported LLM_BACKEND: {backend}. Supported values: ollama, openai_compatible")


def _normalize_ollama_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
