#!/usr/bin/env python3
from __future__ import annotations
import subprocess
import sys

jobs = [
    ("hipporag2", "labelON", "38,230", "HIPPORAG2_USE_LABEL_FACT_QUERIES=true"),
    ("hipporag2", "labelOFF", "230,549", "HIPPORAG2_USE_LABEL_FACT_QUERIES=false"),
    ("vanilla_rag", "none", "345,356,513", "VANILLA_RAG_QA_TOP_K=8 VANILLA_RAG_RETRIEVAL_TOP_K=8"),
]
procs = []
for method, tag, rows, env_prefix in jobs:
    out = f"outputs/method_runs/SWMH-failed-retry-{method}-{tag}-gemma4-31b.jsonl"
    env = dict(__import__('os').environ)
    for assignment in env_prefix.split():
        key, value = assignment.split('=', 1)
        env[key] = value
    if method == "hipporag2":
        env["HIPPORAG2_QA_TOP_K"] = "8"
        env["HIPPORAG2_RETRIEVAL_TOP_K"] = "8"
    procs.append(subprocess.Popen([
        ".venv/bin/python", "scripts/run_method.py", "--method", method,
        "--dataset", "SWMH", "--split", "test", "--row-indices", rows,
        "--generate-answer", "--output", out,
    ], env=env))
rc = 0
for p in procs:
    rc = max(rc, p.wait())
raise SystemExit(rc)
