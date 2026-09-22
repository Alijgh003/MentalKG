"""Embedding-based semantic score for generated explanations.

This is intentionally a small, reproducible BERTScore-style baseline: each
generated/reference explanation is represented by the configured embedding
model and compared with cosine similarity. It is not the original token-level
BERTScore algorithm; the name is kept in the output schema for experiment
tracking and the implementation can later be replaced without changing the
runner interface.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from kg_pipeline.embeddings import EmbeddingConfig, OpenAICompatibleEmbedder
from methods.datasets import load_samples


REFERENCE_FIELDS = ("explanation", "response", "gpt-3.5-turbo", "reference_explanation")


def reasoning_only(text: str) -> str:
    """Remove the reference answer prefix when a benchmark stores it inline."""
    value = str(text or "").strip()
    match = re.split(r"\breasoning\s*:", value, maxsplit=1, flags=re.IGNORECASE)
    return match[1].strip() if len(match) == 2 and match[1].strip() else value


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def reference_explanation(raw: dict[str, Any], field: str | None = None) -> tuple[str, str]:
    """Return normalized reference explanation and the source field name."""
    candidates = (field,) if field else REFERENCE_FIELDS
    for name in candidates:
        if name and str(raw.get(name, "")).strip():
            return reasoning_only(str(raw[name])), name
    return "", ""


def evaluate_run(
    run_path: Path,
    *,
    dataset: str,
    split: str,
    reference_field: str | None = None,
    dimension: int = 512,
) -> dict[str, Any]:
    records = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = load_samples(dataset, split, limit=None)
    by_id = {sample.sample_id: sample for sample in samples}
    pairs: list[tuple[dict[str, Any], str, str, str]] = []
    skipped: list[dict[str, str]] = []
    for record in records:
        sample = by_id.get(record.get("sample_id"))
        generated = str(record.get("output", {}).get("explanation", "")).strip()
        reference, field = reference_explanation(sample.raw if sample else {}, reference_field)
        if sample is None or not generated or not reference:
            skipped.append({"sample_id": str(record.get("sample_id", "")), "reason": "missing_generated_or_reference"})
            continue
        pairs.append((record, generated, reference, field))

    embedder = OpenAICompatibleEmbedder(EmbeddingConfig.from_env())
    texts = [text for _, generated, reference, _ in pairs for text in (generated, reference)]
    vectors = embedder.embed(texts, dimension=dimension)
    rows = []
    for index, (record, generated, reference, field) in enumerate(pairs):
        score = cosine_similarity(vectors[index * 2], vectors[index * 2 + 1])
        rows.append({
            "sample_id": record["sample_id"],
            "dataset_row_index": record.get("dataset_row_index"),
            "gold_label": record.get("gold", {}).get("label"),
            "reference_field": field,
            "generated_explanation": generated,
            "reference_explanation": reference,
            "bert_score_cosine": score,
        })
    scores = [row["bert_score_cosine"] for row in rows]
    return {
        "metric": "bert_score_cosine",
        "description": "Cosine similarity between whole-explanation embeddings; BERTScore-style proxy, not token-level BERTScore.",
        "run": str(run_path),
        "dataset": dataset,
        "split": split,
        "samples": len(records),
        "evaluated": len(rows),
        "skipped": skipped,
        "mean": float(np.mean(scores)) if scores else None,
        "median": float(np.median(scores)) if scores else None,
        "per_sample": rows,
    }

