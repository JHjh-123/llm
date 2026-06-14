from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    name: str

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbeddingProvider(EmbeddingProvider):
    name = "hash"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = digest[0] % self.dim
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize(vector)


class OllamaEmbeddingProvider(EmbeddingProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: int = 10) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model = model
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def embed(self, text: str) -> list[float]:
        body = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

        embeddings = payload.get("embeddings") or []
        if not embeddings:
            raise RuntimeError("Ollama embedding response did not include embeddings")
        return [float(value) for value in embeddings[0]]


def build_embedding_provider_from_env() -> EmbeddingProvider:
    backend = os.getenv("EMBEDDING_BACKEND", "hash").lower()
    if backend == "hash":
        return HashEmbeddingProvider(dim=int(os.getenv("HASH_EMBEDDING_DIM", "64")))
    if backend == "ollama":
        return OllamaEmbeddingProvider(
            base_url=os.getenv("OLLAMA_BASE_URL", os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434")),
            model=os.getenv("OLLAMA_EMBED_MODEL", os.getenv("EMBEDDING_MODEL", "nomic-embed-text")),
            timeout=int(os.getenv("EMBEDDING_TIMEOUT", "10")),
        )
    raise ValueError(f"Unsupported EMBEDDING_BACKEND: {backend}")


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized
