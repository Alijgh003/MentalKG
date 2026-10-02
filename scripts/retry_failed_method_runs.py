#!/usr/bin/env python3
"""Retry failed answer-generation records and merge them into a method JSONL."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--options", action="append", default=[])
    ap.add_argument("--attempt-output", type=Path, required=True)
    ap.add_argument(
        "--min-interval-seconds", type=float, default=0,
        help="When set, run each failed sample separately and wait this many seconds between starts.",
    )
    args = ap.parse_args()

    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    failed = [r for r in records if (r.get("output") or {}).get("parse_status") != "ok"]
    if not failed:
        args.output.write_text(args.input.read_text())
        print("no failed records")
        return 0
    row_indices = ",".join(str(r["dataset_row_index"]) for r in failed)
    print("retrying", len(failed), "rows")
    retries = {}
    if args.min_interval_seconds > 0:
        # A separate process per sample is intentional: HippoRAG may issue
        # several internal LLM calls for one sample.
        args.attempt_output.parent.mkdir(parents=True, exist_ok=True)
        with args.attempt_output.open("w", encoding="utf-8") as combined:
            for pos, failed_row in enumerate(failed):
                single = args.attempt_output.with_name(
                    f"{args.attempt_output.stem}-{pos+1:02d}.jsonl"
                )
                cmd = [sys.executable, "scripts/run_method.py", "--method", args.method,
                       "--dataset", "SWMH", "--split", "test", "--row-indices",
                       str(failed_row["dataset_row_index"]), "--generate-answer",
                       "--output", str(single)]
                for option in args.options:
                    cmd += ["--method-option", option]
                subprocess.run(cmd, check=False)
                if single.exists():
                    for line in single.read_text().splitlines():
                        if line.strip():
                            combined.write(line + "\n")
                            record = json.loads(line)
                            retries[record["sample_id"]] = record
                if pos + 1 < len(failed):
                    print(f"sleeping {args.min_interval_seconds:g}s before next sample")
                    time.sleep(args.min_interval_seconds)
    else:
        cmd = [sys.executable, "scripts/run_method.py", "--method", args.method,
               "--dataset", "SWMH", "--split", "test", "--row-indices", row_indices,
               "--generate-answer", "--output", str(args.attempt_output)]
        for option in args.options:
            cmd += ["--method-option", option]
        subprocess.run(cmd, check=True)
        retries = {r["sample_id"]: r for r in
                   (json.loads(line) for line in args.attempt_output.read_text().splitlines() if line.strip())}
    merged = [retries.get(r["sample_id"], r) for r in records]
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
    ok = sum((r.get("output") or {}).get("parse_status") == "ok" for r in merged)
    print(f"merged {len(merged)} records; parse_status=ok: {ok}/{len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
