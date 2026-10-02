#!/usr/bin/env python3
"""Report per-sample support-score means and their 0--10 distribution.

The global factual evaluator stores an integer support score for every
clinical claim.  This report defensively discretizes any numeric value into
unit interval bin in [0, 10), computes a mean per sample, and reports the
distribution of those per-sample means using the same bins.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discrete_score(value: Any) -> int:
    """Put a score in [0, 10) into [k, k+1); reserve 10 for exactly 10."""
    match = re.search(r"-?(?:\d+(?:\.\d*)?|\.\d+)", str(value))
    if not match:
        raise ValueError(f"Invalid support score: {value!r}")
    number = float(match.group(0))
    if not math.isfinite(number):
        raise ValueError(f"Invalid support score: {value!r}")
    if number >= 10:
        return 10
    return max(0, min(9, math.floor(number)))


def histogram(values: list[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(score): counts.get(score, 0) for score in range(11)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/evaluations/gold_global_factual_dr70_swmh65_score_top3.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/evaluations/gold_global_factual_dr70_swmh65_per_sample_support.jsonl",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    reports: list[dict[str, Any]] = []
    all_claim_scores: list[int] = []

    for row in rows:
        items = row.get("global_factuality", {}).get("clinical_triples", [])
        scores = [discrete_score(item["support_score"]) for item in items if "support_score" in item]
        all_claim_scores.extend(scores)
        mean = sum(scores) / len(scores) if scores else None
        mean_bin = discrete_score(mean) if mean is not None else None
        reports.append(
            {
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "sample_id": row.get("sample_id"),
                "status": row.get("status"),
                "clinical_claims": len(scores),
                "discrete_claim_scores": scores,
                "mean_support_score": round(mean, 6) if mean is not None else None,
                "mean_support_score_bin": mean_bin,
                "claim_score_distribution": histogram(scores),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(report, ensure_ascii=False) + "\n" for report in reports),
        encoding="utf-8",
    )

    means = [report["mean_support_score_bin"] for report in reports if report["mean_support_score_bin"] is not None]
    summary = {
        "input": str(args.input.relative_to(ROOT)) if args.input.is_relative_to(ROOT) else str(args.input),
        "output": str(args.output.relative_to(ROOT)) if args.output.is_relative_to(ROOT) else str(args.output),
        "samples": len(reports),
        "samples_with_clinical_claims": len(means),
        "clinical_claims": len(all_claim_scores),
        "claim_score_distribution": histogram(all_claim_scores),
        "binning": "bin k means [k, k+1), except bin 10 means exactly 10",
        "per_sample_mean": {
            "mean_of_sample_means": round(sum(report["mean_support_score"] for report in reports if report["mean_support_score"] is not None) / len(means), 6) if means else None,
            "distribution_binned_to_0_10": histogram(means),
        },
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nPER_SAMPLE_MEAN_DISTRIBUTION (unit interval bins: [k, k+1), 10 exact)")
    for score, count in summary["per_sample_mean"]["distribution_binned_to_0_10"].items():
        print(f"{score}\t{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
