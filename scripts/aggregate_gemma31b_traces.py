#!/usr/bin/env python3
"""Aggregate Gemma31b ToG traces and retain rows matching the selected 135 IDs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_ids(path: Path) -> set[str]:
    return {json.loads(line)["sample_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def explanation_from_raw(raw: str, label: str) -> str:
    text = str(raw or "").strip()
    # ToG-1 uses {{label}}, ToG-2 uses {label}; keep only the explanation.
    text = re.sub(r"\s*\{\{?\s*" + re.escape(str(label)) + r"\s*\}\}?\s*$", "", text, flags=re.I)
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/home/alihoosh/Downloads/aliz/aliz/traces"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/alihoosh/Downloads/aliz/aliz/gemma31b_aggregated"))
    parser.add_argument("--selected-claims", type=Path, default=ROOT / "outputs/evaluations/gold_claim_extractions_dr70_swmh65_glossed.jsonl")
    args = parser.parse_args()
    selected = read_ids(args.selected_claims)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for setup in ("ToG-1", "ToG-2"):
        rows = []
        for dataset in ("DR", "SWMH"):
            for path in sorted((args.source_root / setup / dataset / "gemma31b").glob("trace_*.json")):
                trace = json.loads(path.read_text(encoding="utf-8"))
                final = trace.get("final") or {}
                label = final.get("pred_label", "")
                rows.append({
                    "setup": setup,
                    "dataset": dataset,
                    "split": "test",
                    "sample_id": trace.get("benchmark_id"),
                    "benchmark_id": trace.get("benchmark_id"),
                    "gold_label": trace.get("gold_label"),
                    "generated_labels": [label] if label else [],
                    "generated_explanation": explanation_from_raw(final.get("raw_answer", ""), label),
                    "trace_path": str(path),
                    "llm_model": trace.get("llm_model"),
                    "final": final,
                    "token_totals": trace.get("token_totals"),
                    "latency_totals": trace.get("latency_totals"),
                })
        rows.sort(key=lambda row: (row.get("dataset", ""), row.get("sample_id", "")))
        matched = [row for row in rows if row.get("sample_id") in selected]
        for suffix, values in (("all", rows), ("matched_selected_135", matched)):
            out = args.output_dir / f"{setup.lower().replace('-', '')}_gemma31b_{suffix}.jsonl"
            out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
            print(json.dumps({"setup": setup, "file": str(out), "rows": len(values), "selected_overlap": len(set(row.get('sample_id') for row in values) & selected)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
