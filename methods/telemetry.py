from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Iterator


logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class StageRecorder:
    """Collect machine-readable latency, token, call, and status data per stage."""

    def __init__(self, **context: Any):
        self.events: list[dict[str, Any]] = []
        self.context = context

    @contextmanager
    def stage(self, name: str, **details: Any) -> Iterator[dict[str, Any]]:
        event: dict[str, Any] = {
            "stage": name,
            "started_at": utc_now(),
            "status": "ok",
            "latency_ms": 0,
            "llm_calls": 0,
            "embedding_calls": 0,
            "retrieval_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "details": details,
        }
        started = time.perf_counter()
        try:
            yield event
        except Exception as error:
            event["status"] = "error"
            event["error_type"] = type(error).__name__
            event["error"] = str(error)
            raise
        finally:
            event["ended_at"] = utc_now()
            event["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self.events.append(event)
            logger.info(
                "stage=%s status=%s latency_ms=%.3f llm_calls=%s tokens_in=%s tokens_out=%s context=%s",
                name,
                event["status"],
                event["latency_ms"],
                event["llm_calls"],
                event["input_tokens"],
                event["output_tokens"],
                self.context,
            )
