from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    dataset: str
    split: str
    text: str
    gold_label: str | None
    dataset_row_index: int
    source_row_index: str | None
    valid_labels: tuple[str, ...] = ()
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class MethodResult:
    seed_entities: list[dict[str, Any]] = field(default_factory=list)
    triples: list[dict[str, Any]] = field(default_factory=list)
    passages: list[dict[str, Any]] = field(default_factory=list)
    paths: list[dict[str, Any]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    label_scores: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    raw_response: str = ""
    parse_status: str = "not_run"
    stopping_reason: str = "retrieval_complete"
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    traversal_steps: int = 0
    errors: list[str] = field(default_factory=list)
    method_metadata: dict[str, Any] = field(default_factory=dict)
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    stages: list[dict[str, Any]] = field(default_factory=list)


class RetrievalMethod(Protocol):
    name: str

    def run(self, sample: BenchmarkSample) -> MethodResult: ...

    def close(self) -> None: ...
