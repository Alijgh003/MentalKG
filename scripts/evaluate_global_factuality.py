#!/usr/bin/env python3
"""Evaluate generated clinical triples against KG facts in Milvus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.factual_global import GlobalFactualEvaluator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSONL records containing clinical_claims")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dimension", type=int)
    parser.add_argument(
        "--stop-after",
        choices=("clinical_triples", "fact_retrieval", "judge"),
        default="judge",
        help="Stop after reading triples, Milvus fact retrieval, or complete judge evaluation",
    )
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = GlobalFactualEvaluator(top_k=args.top_k, dimension=args.dimension).evaluate_records(
        records, stop_after=args.stop_after
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
