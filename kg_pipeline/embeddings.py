"""Reusable client for batched OpenAI-compatible text embeddings."""

from __future__ import annotations

import json
import logging
import time
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.env import PROJECT_ENV_FILE

logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseSettings):
    """Embedding settings loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=PROJECT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    base_url: str
    model: str
    api_key: str = "not-needed"
    batch_size: int = Field(default=128, gt=0)
    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=5, ge=0)

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Compatibility wrapper; BaseSettings performs the actual loading."""
        return cls()


class OpenAICompatibleEmbedder:
    """Embed arrays of text and return a finite float32 NumPy matrix."""

    def __init__(self, config: EmbeddingConfig):
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.config = config

    def embed(self, texts: Sequence[str], *, dimension: int) -> np.ndarray:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        values = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("Every embedding input must be a non-empty string")
        if not values:
            return np.empty((0, dimension), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for start in range(0, len(values), self.config.batch_size):
            chunks.append(self._embed_batch(values[start : start + self.config.batch_size], dimension))
        return np.concatenate(chunks, axis=0)

    def _embed_batch(self, texts: list[str], dimension: int) -> np.ndarray:
        payload = {
            "model": self.config.model,
            "input": texts,
            "dimensions": dimension,
            "encoding_format": "float",
        }
        response = self._post(payload)
        items = sorted(response.get("data", []), key=lambda item: item.get("index", -1))
        if len(items) != len(texts):
            raise ValueError(f"Embedding API returned {len(items)} vectors for {len(texts)} inputs")
        matrix = np.asarray([item["embedding"] for item in items], dtype=np.float32)
        if matrix.shape != (len(texts), dimension):
            raise ValueError(
                f"Expected embedding shape {(len(texts), dimension)}, got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Embedding API returned NaN or infinite values")
        return matrix

    def _post(self, payload: dict) -> dict:
        endpoint = f"{self.config.base_url.rstrip('/')}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
        for attempt in range(self.config.max_retries + 1):
            try:
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return json.load(response)
            except HTTPError as error:
                retryable = error.code == 429 or error.code >= 500
                detail = error.read(1000).decode(errors="replace")
                if not retryable or attempt == self.config.max_retries:
                    raise RuntimeError(f"Embedding API HTTP {error.code}: {detail}") from error
                delay = min(2**attempt, 30)
                logger.warning(
                    "Embedding request failed (attempt %s/%s, endpoint=%s, "
                    "HTTP status=%s, retry_in=%ss): %s",
                    attempt + 1,
                    self.config.max_retries + 1,
                    endpoint,
                    error.code,
                    delay,
                    detail or error.reason,
                )
            except (URLError, TimeoutError) as error:
                if attempt == self.config.max_retries:
                    raise RuntimeError(
                        f"Embedding API request failed at {endpoint}: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                delay = min(2**attempt, 30)
                logger.warning(
                    "Embedding request failed (attempt %s/%s, endpoint=%s, "
                    "error_type=%s, retry_in=%ss): %s",
                    attempt + 1,
                    self.config.max_retries + 1,
                    endpoint,
                    type(error).__name__,
                    delay,
                    error,
                )
            time.sleep(delay)
        raise AssertionError("unreachable")
