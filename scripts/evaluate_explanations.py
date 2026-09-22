#!/usr/bin/env python3
"""Score generated explanations against benchmark references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.bert_score import evaluate_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding-based BERTScore-style explanation evaluation")
    parser.add_argument("--run", type=Path, required=True, help="Method run JSONL")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--reference-field", help="Override auto-detected reference field")
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_run(
        args.run,
        dataset=args.dataset,
        split=args.split,
        reference_field=args.reference_field,
        dimension=args.dimension,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
