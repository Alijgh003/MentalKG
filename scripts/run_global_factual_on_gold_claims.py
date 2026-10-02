#!/usr/bin/env python3
"""Run GlobalFactualEvaluator on clinical claims extracted from gold explanations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import Settings
from evaluation.factual_global import GlobalFactualEvaluator


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/evaluations/gold_claim_extractions_dr70_swmh65.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/evaluations/gold_global_factual_dr70_swmh65.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dimension", type=int)
    args = parser.parse_args()

    source = read_jsonl(args.input)
    settings = Settings()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.output.exists():
        existing = {
            row["sample_id"]: row
            for row in read_jsonl(args.output)
            if row.get("status") == "ok"
        }

    pending = [row for row in source if row["sample_id"] not in existing]
    evaluator = None
    if pending:
        evaluator = GlobalFactualEvaluator(top_k=args.top_k, dimension=args.dimension)
    with args.output.open("a", encoding="utf-8") as stream:
        for index, source_row in enumerate(pending, start=1):
            clinical_claims = [
                {
                    "subject": claim["subject"],
                    "predicate": claim["predicate"],
                    "object": claim["object"],
                }
                for claim in source_row.get("claims", [])
                if claim.get("scope") == "clinical"
            ]
            started = time.perf_counter()
            try:
                result = evaluator.evaluate_records(
                    [{"sample_id": source_row["sample_id"], "clinical_claims": clinical_claims}],
                    stop_after="judge",
                )
                elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                sample_result = result["results"][0]
                row = {
                    "dataset": source_row.get("dataset", "unknown"),
                    "split": source_row.get("split", "unknown"),
                    "sample_id": source_row["sample_id"],
                    "benchmark_id": source_row.get("benchmark_id"),
                    "dataset_row_index": source_row.get("dataset_row_index"),
                    "source_row_index": source_row.get("source_row_index"),
                    "gold_label": source_row.get("gold_label"),
                    "global_factuality": sample_result,
                    "cost": {
                        "latency_ms": elapsed_ms,
                        "input_tokens": result.get("cost", {}).get("input_tokens", 0),
                        "output_tokens": result.get("cost", {}).get("output_tokens", 0),
                        "clinical_claims": len(clinical_claims),
                        "retrieval_top_k": args.top_k,
                    },
                    "settings": {
                        "model": settings.llm_model,
                        "api_base": settings.llm_api_base,
                        "milvus_collection": evaluator.collection,
                        "embedding_dimension": evaluator.dimension,
                    },
                    "status": "ok",
                }
                supported = sample_result["summary"]["supported"]
                generated = sample_result["summary"]["generated"]
                print(
                    f"[{index}/{len(pending)}] {source_row.get('dataset', 'unknown')} {source_row['sample_id']} "
                    f"claims={generated} supported={supported} latency_ms={elapsed_ms}",
                    flush=True,
                )
            except Exception as error:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                row = {
                    "dataset": source_row.get("dataset", "unknown"),
                    "split": source_row.get("split", "unknown"),
                    "sample_id": source_row["sample_id"],
                    "benchmark_id": source_row.get("benchmark_id"),
                    "gold_label": source_row.get("gold_label"),
                    "cost": {"latency_ms": elapsed_ms, "clinical_claims": len(clinical_claims)},
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"[{index}/{len(pending)}] ERROR {source_row['sample_id']}: {error}", flush=True)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()

    # Keep only the latest result per sample. This removes stale error rows from
    # an earlier interrupted run after a later retry succeeds.
    latest = {row["sample_id"]: row for row in read_jsonl(args.output)}
    all_rows = list(latest.values())
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    summary = {
        "input": str(args.input.relative_to(ROOT)) if args.input.is_relative_to(ROOT) else str(args.input),
        "output": str(args.output.relative_to(ROOT)) if args.output.is_relative_to(ROOT) else str(args.output),
        "metric": "GlobalFactualExplanationEval",
        "samples": len(all_rows),
        "ok": sum(row.get("status") == "ok" for row in all_rows),
        "errors": sum(row.get("status") == "error" for row in all_rows),
        "clinical_claims": sum(row.get("cost", {}).get("clinical_claims", 0) for row in all_rows),
        "supported_claims": sum(
            row.get("global_factuality", {}).get("summary", {}).get("supported", 0)
            for row in all_rows
        ),
        "generated_claims": sum(
            row.get("global_factuality", {}).get("summary", {}).get("generated", 0)
            for row in all_rows
        ),
        "mean_support_score": (
            sum(
                sum(item.get("support_score", 0) for item in row.get("global_factuality", {}).get("clinical_triples", []))
                for row in all_rows
            )
            / sum(row.get("cost", {}).get("clinical_claims", 0) for row in all_rows)
            if sum(row.get("cost", {}).get("clinical_claims", 0) for row in all_rows)
            else 0.0
        ),
        "support_score_max": 10,
        "total_latency_ms": round(sum(row.get("cost", {}).get("latency_ms", 0) for row in all_rows), 3),
        "total_input_tokens": sum(row.get("cost", {}).get("input_tokens", 0) for row in all_rows),
        "total_output_tokens": sum(row.get("cost", {}).get("output_tokens", 0) for row in all_rows),
        "settings": {
            "model": settings.llm_model,
            "api_base": settings.llm_api_base,
            "api_key_present": bool(settings.llm_api_key),
            "milvus_collection": evaluator.collection if evaluator else "raw_triples_v1",
            "embedding_dimension": evaluator.dimension if evaluator else (args.dimension or 256),
            "top_k": args.top_k,
        },
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
