#!/usr/bin/env python3
"""Extract LocalGraphEval claims for the fixed DR/SWMH gold subsets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import Settings
from evaluation.bert_score import reference_explanation
from evaluation.local_graph_eval import LocalGraphEvaluator


BENCHMARK_ROOT = ROOT / "datasets" / "benchmarks" / "dsm_grounded_v1"
REPRESENTATIVES = {
    "DR": ROOT / "outputs/method_runs/DR-70-cot-gemma4-31b-20260924T144232Z.jsonl",
    "SWMH": ROOT / "outputs/method_runs/SWMH-rows65-hipporag2-labelON-gemma4-31b-aggregated.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def display_path(path: Path) -> str:
    """Render relative CLI paths safely against the project root."""
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def benchmark_rows(dataset: str) -> dict[str, dict[str, str]]:
    path = BENCHMARK_ROOT / dataset / "test.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["benchmark_id"]: row for row in csv.DictReader(stream)}


def prepare_records() -> list[dict]:
    prepared: list[dict] = []
    for dataset, representative in REPRESENTATIVES.items():
        rows = benchmark_rows(dataset)
        for record in read_jsonl(representative):
            sample_id = record["sample_id"]
            row = rows.get(sample_id)
            if row is None:
                raise KeyError(f"{sample_id} is missing from {dataset}/test.csv")
            gold, reference_field = reference_explanation(row, field="gpt-3.5-turbo")
            prepared.append({
                "dataset": dataset,
                "split": "test",
                "sample_id": sample_id,
                "dataset_row_index": record.get("dataset_row_index"),
                "source_row_index": record.get("source_row_index"),
                "benchmark_id": row["benchmark_id"],
                "gold_label": row.get("gold_label"),
                "reference_field": reference_field,
                "gold_explanation": gold,
                "representative_run": str(representative.relative_to(ROOT)),
            })
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/evaluations/gold_claim_extractions_dr70_swmh65.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/evaluations/gold_claim_extractions_dr70_swmh65_manifest.json",
    )
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    prepared = prepare_records()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.output.exists() and not args.retry_failed:
        existing = {
            row["sample_id"]: row
            for row in read_jsonl(args.output)
            if row.get("status") == "ok"
        }
    pending = [row for row in prepared if row["sample_id"] not in existing]
    evaluator = LocalGraphEvaluator()
    with args.output.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] {item['dataset']} {item['sample_id']}", flush=True)
            try:
                entities, claims, usage = evaluator.extract(item["gold_explanation"], prefix="G")
                output = {
                    **item,
                    "status": "ok",
                    "entities": entities,
                    "claims": [claim.as_dict() for claim in claims],
                    "usage": usage,
                    "extractor": {
                        "model": settings.llm_model,
                        "api_base": settings.llm_api_base,
                    },
                }
            except Exception as error:
                output = {
                    **item,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")
            stream.flush()

    all_rows = read_jsonl(args.output)
    args.manifest.write_text(json.dumps({
        "artifact": display_path(args.output),
        "datasets": {"DR": 70, "SWMH": 65},
        "total_expected": 135,
        "representatives": {key: str(value.relative_to(ROOT)) for key, value in REPRESENTATIVES.items()},
        "settings": {
            "model": settings.llm_model,
            "api_base": settings.llm_api_base,
            "api_key_present": bool(settings.llm_api_key),
        },
        "status_counts": {
            "ok": sum(row.get("status") == "ok" for row in all_rows),
            "error": sum(row.get("status") == "error" for row in all_rows),
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
