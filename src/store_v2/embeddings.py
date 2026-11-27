"""Embedding providers for the v2 store.

One adapter speaks the OpenAI-compatible ``POST {base_url}/embeddings`` API, which
Fireworks, OpenAI, Together, Mistral, vLLM, Ollama, LM Studio and llama.cpp all serve.
Changing provider is configuration, not code: base URL, model, dimension and the name
of the environment variable that holds the key.

The key is read from the environment at request time and never logged; when it is
absent no Authorization header is sent, so a proxy that injects credentials (or a
local server without auth) works as is. Texts are never logged either.

Uses only the standard library: no HTTP client dependency for one POST.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "fireworks/qwen3-embedding-8b"
DEFAULT_DIM = 1024
DEFAULT_KEY_ENV = "FIREWORKS_API_KEY"
# Qwen3-Embedding is instruction-aware on the query side only; documents go in as they are.
DEFAULT_QUERY_PREFIX = "Instruct: Given a question, retrieve the project notes that answer it\nQuery: "


class EmbeddingError(RuntimeError):
    """The provider could not return vectors. Items stay queued; search falls back to text."""


class EmbeddingClient(Protocol):
    provider: str
    model: str
    dim: int

    def version_config(self) -> dict:
        """Settings that change the vectors: a change means a new embedding version."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class EmbeddingSettings:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    dim: int = DEFAULT_DIM
    api_key_env: str = DEFAULT_KEY_ENV
    query_prefix: str = DEFAULT_QUERY_PREFIX
    send_dimensions: bool = True  # Matryoshka models accept "dimensions"; turn off for providers that reject it
    batch_size: int = 64
    timeout_s: float = 30.0
    max_retries: int = 4
    max_chars: int = 8000  # per text; longer texts are cut, the full text stays in the store
    provider: str = ""

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.provider:
            host = self.base_url.split("://", 1)[-1].split("/", 1)[0]
            self.provider = host or "openai-compatible"
        if self.dim <= 0 or self.batch_size <= 0:
            raise ValueError("embedding dim and batch_size must be positive")

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "EmbeddingSettings":
        env = os.environ if env is None else env
        defaults = cls()

        def get(name: str, default):
            value = env.get(f"MEMORY_EMBEDDING_{name}")
            return default if value in (None, "") else value

        return cls(
            base_url=get("BASE_URL", defaults.base_url),
            model=get("MODEL", defaults.model),
            dim=int(get("DIM", defaults.dim)),
            api_key_env=get("API_KEY_ENV", defaults.api_key_env),
            query_prefix=get("QUERY_PREFIX", defaults.query_prefix),
            send_dimensions=str(get("SEND_DIMENSIONS", "true")).lower() in ("1", "true", "yes"),
            batch_size=int(get("BATCH_SIZE", defaults.batch_size)),
            timeout_s=float(get("TIMEOUT_S", defaults.timeout_s)),
            provider=get("PROVIDER", ""),
        )


@dataclass
class OpenAICompatibleEmbeddings:
    settings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)

    @property
    def provider(self) -> str:
        return self.settings.provider

    @property
    def model(self) -> str:
        return self.settings.model

    @property
    def dim(self) -> int:
        return self.settings.dim

    def version_config(self) -> dict:
        return {"query_prefix": self.settings.query_prefix, "max_chars": self.settings.max_chars}

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.settings.batch_size):
            out.extend(self._request(list(texts[start:start + self.settings.batch_size])))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._request([self.settings.query_prefix + text])[0]

    # -- transport --------------------------------------------------------------

    def _request(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        texts = [(t or " ")[: self.settings.max_chars] for t in texts]
        body = {"model": self.settings.model, "input": texts, "encoding_format": "float"}
        if self.settings.send_dimensions:
            body["dimensions"] = self.settings.dim
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "llm-memory"}
        key = os.environ.get(self.settings.api_key_env, "") if self.settings.api_key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = json.dumps(body).encode()
        url = f"{self.settings.base_url}/embeddings"

        attempt = 0
        while True:
            try:
                request = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.settings.timeout_s) as response:
                    payload = json.loads(response.read().decode())
                return self._parse(payload, len(texts))
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                detail = f"HTTP {exc.code}"
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                retryable, retry_after = True, None
                detail = type(exc).__name__
            except (ValueError, KeyError, TypeError) as exc:
                raise EmbeddingError(f"invalid response from {self.provider}: {exc}") from None
            attempt += 1
            if not retryable or attempt > self.settings.max_retries:
                raise EmbeddingError(f"{self.provider} embeddings failed: {detail}")
            delay = min(30.0, 0.5 * 2 ** (attempt - 1)) + random.uniform(0, 0.25)
            if retry_after and retry_after.isdigit():
                delay = max(delay, min(60.0, float(retry_after)))
            self._sleep(delay)

    def _parse(self, payload: dict, expected: int) -> list[list[float]]:
        rows = sorted(payload["data"], key=lambda row: row.get("index", 0))
        if len(rows) != expected:
            raise ValueError(f"expected {expected} vectors, got {len(rows)}")
        vectors = []
        for row in rows:
            vector = [float(v) for v in row["embedding"]]
            if len(vector) != self.settings.dim:
                raise ValueError(f"expected dimension {self.settings.dim}, got {len(vector)}")
            if not all(math.isfinite(v) for v in vector):
                raise ValueError("vector contains non-finite values")
            vectors.append(vector)
        return vectors
