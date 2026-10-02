#!/usr/bin/env python3
"""Extract gloss-enriched claims from a fixed Gemma 4 31B method run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import Settings
from evaluation.local_graph_eval import LocalGraphEvaluator


METHOD_RUNS = {
    "tog2_gemma31b": [],
    "hipporag_labeloff_gemma31b": [
        ROOT / "outputs/method_runs/DR-70-hipporag-labelOFF-gemma4-31b-20260924T143533Z.jsonl",
        ROOT / "outputs/method_runs/SWMH-rows65-hipporag2-labelOFF-gemma4-31b.jsonl",
    ],
    "vanilla_rag_gemma31b": [
        ROOT / "outputs/method_runs/DR-70-vanilla-gemma4-31b-20260924T144002Z.jsonl",
        ROOT / "outputs/method_runs/SWMH-rows65-vanilla_rag-none-gemma4-31b.jsonl",
    ],
    "cot_gemma31b": [
        ROOT / "outputs/method_runs/DR-70-cot-gemma4-31b-20260924T144232Z.jsonl",
        ROOT / "outputs/method_runs/SWMH-rows65-cot-none-gemma4-31b.jsonl",
    ],
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHOD_RUNS), required=True)
    parser.add_argument("--input", type=Path, action="append", help="Method JSONL input file(s); overrides built-in paths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    source_paths = args.input or METHOD_RUNS[args.method]
    if not source_paths:
        raise RuntimeError("--input is required for tog2_gemma31b (provide the aggregated ToG2 JSONL run)")
    source_rows = [row for path in source_paths for row in read_jsonl(path)]
    unique_count = len({row.get("sample_id") for row in source_rows})
    if not args.input and unique_count != 135:
        raise RuntimeError(f"Expected 135 unique samples, got {unique_count}")
    if args.input and unique_count != len(source_rows):
        raise RuntimeError(f"Input contains duplicate sample IDs: {unique_count} unique for {len(source_rows)} rows")

    previous = {row["sample_id"]: row for row in read_jsonl(args.output)} if args.output.exists() else {}
    if args.retry_failed:
        pending = [row for row in source_rows if previous.get(row["sample_id"], {}).get("status") != "ok"]
    else:
        pending = [row for row in source_rows if previous.get(row["sample_id"], {}).get("status") != "ok"]
    evaluator = LocalGraphEvaluator() if pending else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for index, source in enumerate(pending, 1):
            explanation = (source.get("output") or {}).get("explanation", "")
            cot_reasoning = (source.get("stage_outputs", {}).get("cot_generation") or {}).get("reasoning", "")
            try:
                if not explanation.strip():
                    raise ValueError("method output.explanation is empty")
                entities, claims, usage = evaluator.extract(explanation, "M")
                out = {
                    "method": args.method,
                    "dataset": source.get("dataset"),
                    "split": source.get("split"),
                    "sample_id": source["sample_id"],
                    "dataset_row_index": source.get("dataset_row_index"),
                    "source_row_index": source.get("source_row_index"),
                    "gold_label": (source.get("gold") or {}).get("label"),
                    "source_run_files": [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path) for path in source_paths],
                    "generated_labels": (source.get("output") or {}).get("labels", []),
                    "generated_explanation": explanation,
                    "cot_reasoning": cot_reasoning,
                    "entities": entities,
                    "claims": [claim.as_dict() for claim in claims],
                    "usage": usage,
                    "extractor": {"model": settings.llm_model, "api_base": settings.llm_api_base},
                    "status": "ok",
                }
            except Exception as error:
                # Preserve provenance even when a method produced no explanation;
                # downstream global evaluation must be able to resume safely.
                out = {
                    "method": args.method,
                    "dataset": source.get("dataset"),
                    "split": source.get("split"),
                    "sample_id": source["sample_id"],
                    "dataset_row_index": source.get("dataset_row_index"),
                    "source_row_index": source.get("source_row_index"),
                    "gold_label": (source.get("gold") or {}).get("label"),
                    "generated_labels": (source.get("output") or {}).get("labels", []),
                    "generated_explanation": explanation,
                    "cot_reasoning": cot_reasoning,
                    "claims": [],
                    "entities": {"case": [], "clinical": []},
                    "method": args.method,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            stream.write(json.dumps(out, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(pending)}] {source['sample_id']} {out['status']}", flush=True)

    latest = {row["sample_id"]: row for row in read_jsonl(args.output)}
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in latest.values()), encoding="utf-8")
    print(json.dumps({"method": args.method, "samples": len(latest), "ok": sum(r.get('status') == 'ok' for r in latest.values()), "errors": sum(r.get('status') == 'error' for r in latest.values())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
