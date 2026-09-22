"""Reusable client for OpenAI-compatible reranking endpoints."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.env import PROJECT_ENV_FILE
from .embeddings import EmbeddingConfig


logger = logging.getLogger(__name__)


class RerankerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RERANK_",
        env_file=PROJECT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=5, ge=0)

    @classmethod
    def from_env(cls) -> "RerankerConfig":
        values = cls()
        embeddings = EmbeddingConfig.from_env()
        return cls(
            base_url=values.base_url or embeddings.base_url,
            api_key=values.api_key if values.api_key is not None else embeddings.api_key,
            model=values.model or embeddings.model,
            timeout_seconds=values.timeout_seconds,
            max_retries=values.max_retries,
        )


@dataclass(frozen=True)
class RerankHit:
    index: int
    score: float


class OpenAICompatibleReranker:
    def __init__(self, config: RerankerConfig):
        if not config.base_url or not config.model:
            raise ValueError("Reranker base_url and model are required")
        self.config = config

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[RerankHit]:
        if not query.strip() or any(not text.strip() for text in documents):
            raise ValueError("Reranking requires a query and non-empty documents")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if not documents:
            return []
        payload = {
            "model": self.config.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        response = self._post(payload)
        hits = [
            RerankHit(index=int(item["index"]), score=float(item["relevance_score"]))
            for item in response.get("results", [])
        ]
        if any(hit.index < 0 or hit.index >= len(documents) for hit in hits):
            raise ValueError("Reranker returned an out-of-range document index")
        return hits

    def _post(self, payload: dict) -> dict:
        endpoint = f"{self.config.base_url.rstrip('/')}/rerank"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
        for attempt in range(self.config.max_retries + 1):
            try:
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return json.load(response)
            except HTTPError as error:
                detail = error.read(1000).decode(errors="replace")
                retryable = error.code == 429 or error.code >= 500
                if not retryable or attempt == self.config.max_retries:
                    raise RuntimeError(f"Reranker API HTTP {error.code}: {detail}") from error
            except (URLError, TimeoutError) as error:
                if attempt == self.config.max_retries:
                    raise RuntimeError(f"Reranker request failed at {endpoint}: {error}") from error
            delay = min(2**attempt, 30)
            logger.warning("Reranker request failed; retrying in %ss", delay)
            time.sleep(delay)
        raise AssertionError("unreachable")
