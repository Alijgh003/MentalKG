#!/usr/bin/env python3
"""Freeze deterministic validation/test subsets for DSM-grounded evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "datasets" / "benchmarks" / "dsm_grounded_v1"
SEED = "dsm-grounded-v1-2026-09-19"


@dataclass(frozen=True)
class SplitSpec:
    dataset: str
    split: str
    source: str
    cap: int


SPECS = (
    SplitSpec("DR", "validation", "datasets/external/MentalLLaMA/train_data/complete_data/DR/reddit_valid.csv", 430),
    SplitSpec("DR", "test", "datasets/external/MentalLLaMA/test_data/test_complete/DR.csv", 430),
    SplitSpec("T-SID", "validation", "datasets/external/MentalLLaMA/train_data/complete_data/t-sid/val.csv", 430),
    SplitSpec("T-SID", "test", "datasets/external/MentalLLaMA/test_data/test_instruction/t-sid.csv", 430),
    SplitSpec("SWMH", "validation", "datasets/external/MentalLLaMA/train_data/complete_data/swmh/val.csv", 600),
    SplitSpec("SWMH", "test", "datasets/external/MentalLLaMA/test_data/test_instruction/swmh.csv", 600),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_label(dataset: str, row: dict[str, str]) -> str:
    response = row.get("response") or row.get("gpt-3.5-turbo") or row.get("label") or "unknown"
    normalized = " ".join(response.strip().casefold().split())
    if dataset == "SWMH":
        patterns = {
            "this post shows no mental disorder symptoms": "no mental disorders",
            "this post shows mental disorder symptoms related to anxiety": "anxiety",
            "this post shows mental disorder symptoms related to bipolar disorder": "bipolar disorder",
            "this post shows mental disorder symptoms related to depression": "depression",
            "this post shows mental disorder symptoms related to suicide": "suicide",
        }
        for prefix, label in patterns.items():
            if normalized.startswith(prefix):
                return label
        return "__unrecognized__"
    if dataset == "T-SID":
        patterns = {
            "this post shows no mental disorders": "no mental disorders",
            "this post shows depression": "depression",
            "this post shows ptsd": "ptsd",
            "this post shows suicide or self-harm tendency": "suicide or self-harm tendency",
        }
        for prefix, label in patterns.items():
            if normalized.startswith(prefix):
                return label
        return "__unrecognized__"
    return response.strip().split(".", 1)[0].strip().casefold() or "unknown"


def row_fingerprint(dataset: str, split: str, index: int, row: dict[str, str]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{SEED}|{dataset}|{split}|{index}|{payload}".encode()).hexdigest()


def proportional_quotas(counts: Counter[str], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if target >= total:
        return dict(counts)
    exact = {label: target * count / total for label, count in counts.items()}
    quotas = {label: int(value) for label, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(counts, key=lambda label: (-(exact[label] - quotas[label]), label))
    for label in order[:remaining]:
        quotas[label] += 1
    return quotas


def select_rows(dataset: str, split: str, rows: list[dict[str, str]], cap: int) -> list[tuple[int, dict[str, str], str]]:
    candidates: dict[str, list[tuple[str, int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        label = row_label(dataset, row)
        if label == "__unrecognized__":
            continue
        candidates[label].append((row_fingerprint(dataset, split, index, row), index, row))
    quotas = proportional_quotas(Counter({label: len(items) for label, items in candidates.items()}), min(cap, len(rows)))
    selected: list[tuple[int, dict[str, str], str]] = []
    for label, items in candidates.items():
        for fingerprint, index, row in sorted(items)[: quotas[label]]:
            selected.append((index, row, fingerprint))
    return sorted(selected, key=lambda item: item[2])


def freeze(spec: SplitSpec) -> dict[str, object]:
    source = ROOT / spec.source
    relative_output = Path(spec.dataset) / f"{spec.split}.csv"
    output = OUTPUT_ROOT / relative_output
    if not source.exists():
        return {
            "dataset": spec.dataset,
            "split": spec.split,
            "status": "pending_access",
            "source": spec.source,
            "requested_cap": spec.cap,
            "reason": "The official split is not present in the current MentalLLaMA checkout.",
        }

    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        source_fields = reader.fieldnames or []
    selected = select_rows(spec.dataset, spec.split, rows, spec.cap)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["benchmark_id", "source_row_index", "gold_label", *source_fields]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source_index, row, fingerprint in selected:
            writer.writerow({
                "benchmark_id": f"{spec.dataset.lower()}-{spec.split}-{fingerprint[:16]}",
                "source_row_index": source_index,
                "gold_label": row_label(spec.dataset, row),
                **row,
            })

    selected_labels = Counter(row_label(spec.dataset, row) for _, row, _ in selected)
    source_labels = Counter(row_label(spec.dataset, row) for row in rows)
    rejected_rows = source_labels.pop("__unrecognized__", 0)
    return {
        "dataset": spec.dataset,
        "split": spec.split,
        "status": "frozen",
        "source": spec.source,
        "source_sha256": file_sha256(source),
        "source_rows": len(rows),
        "eligible_source_rows": len(rows) - rejected_rows,
        "rejected_unrecognized_label_rows": rejected_rows,
        "requested_cap": spec.cap,
        "selected_rows": len(selected),
        "source_label_counts": dict(sorted(source_labels.items())),
        "selected_label_counts": dict(sorted(selected_labels.items())),
        "output": str(relative_output),
        "output_sha256": file_sha256(output),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark_version": "dsm_grounded_v1",
        "sampling_seed": SEED,
        "sampling_method": "deterministic SHA-256 ranking with proportional label stratification",
        "primary_datasets": ["ReDSM5", "PsySym"],
        "secondary_datasets": ["DR", "T-SID", "SWMH"],
        "splits": [freeze(spec) for spec in SPECS],
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
