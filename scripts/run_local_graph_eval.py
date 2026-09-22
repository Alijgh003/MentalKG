#!/usr/bin/env python3
"""Run LocalGraphEval on cached/generated method explanations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.bert_score import reference_explanation
from evaluation.local_graph_eval import LocalGraphEvaluator, evaluate_records
from methods.datasets import load_samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.run.read_text(encoding="utf-8").splitlines() if line.strip()][:args.limit]
    samples = {sample.sample_id: sample for sample in load_samples(args.dataset, args.split, limit=None)}
    prepared = []
    for record in records:
        sample = samples[record["sample_id"]]
        reference, _ = reference_explanation(sample.raw)
        prepared.append({
            "sample_id": record["sample_id"],
            "gold_explanation": reference,
            "generated_explanation": record.get("output", {}).get("explanation", ""),
        })
    result = evaluate_records(LocalGraphEvaluator(), prepared)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metric": result["metric"], "samples": result["samples"], "output": str(args.output), "cost": result["cost"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

