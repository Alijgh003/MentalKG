from __future__ import annotations

import csv
import json
from pathlib import Path

from .base import BenchmarkSample


BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "benchmarks" / "dsm_grounded_v1"


def dataset_labels(dataset: str, split: str, root: Path = BENCHMARK_ROOT) -> tuple[str, ...]:
    """Return the frozen label vocabulary declared by the benchmark manifest."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        # Small ad-hoc datasets used by tests or local experiments may not have a
        # manifest. Production frozen benchmarks do, and therefore use its order.
        path = resolve_dataset_path(dataset, split, root)
        with path.open(newline="", encoding="utf-8") as handle:
            labels = {
                (row.get("gold_label") or "").strip()
                for row in csv.DictReader(handle)
                if (row.get("gold_label") or "").strip()
            }
        if labels:
            return tuple(sorted(labels))
        raise ValueError(f"No label vocabulary is available for {dataset}/{split}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for entry in manifest.get("splits", []):
        if (
            str(entry.get("dataset", "")).casefold() == dataset.casefold()
            and str(entry.get("split", "")).casefold() == split.casefold()
        ):
            counts = entry.get("selected_label_counts") or entry.get("source_label_counts")
            if counts:
                return tuple(str(label) for label in counts)
            break
    raise ValueError(f"No label vocabulary is declared for {dataset}/{split} in {manifest_path}")


def available_datasets(root: Path = BENCHMARK_ROOT) -> list[tuple[str, str]]:
    return sorted(
        (path.parent.name, path.stem)
        for path in root.glob("*/*.csv")
    )


def resolve_dataset_path(dataset: str, split: str, root: Path = BENCHMARK_ROOT) -> Path:
    matches = [
        root / name / f"{split}.csv"
        for name in root.iterdir()
        if name.is_dir() and name.name.casefold() == dataset.casefold()
    ]
    if len(matches) != 1 or not matches[0].is_file():
        choices = ", ".join(f"{name}/{part}" for name, part in available_datasets(root))
        raise ValueError(f"Dataset split {dataset}/{split} is unavailable. Available: {choices}")
    return matches[0]


def _sample_text(row: dict[str, str]) -> str:
    for field_name in ("post", "query", "text", "input"):
        value = (row.get(field_name) or "").strip()
        if value:
            return value
    raise ValueError("Dataset row has no supported input field: post, query, text, or input")


def load_samples(
    dataset: str,
    split: str,
    *,
    limit: int | None = None,
    offset: int = 0,
    root: Path = BENCHMARK_ROOT,
) -> list[BenchmarkSample]:
    path = resolve_dataset_path(dataset, split, root)
    valid_labels = dataset_labels(path.parent.name, split, root)
    samples: list[BenchmarkSample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index < offset:
                continue
            if limit is not None and len(samples) >= limit:
                break
            samples.append(BenchmarkSample(
                sample_id=(row.get("benchmark_id") or f"{path.parent.name}-{split}-{index}"),
                dataset=path.parent.name,
                split=split,
                text=_sample_text(row),
                gold_label=(row.get("gold_label") or None),
                dataset_row_index=index,
                source_row_index=(row.get("source_row_index") or None),
                valid_labels=valid_labels,
                raw=dict(row),
            ))
    return samples
