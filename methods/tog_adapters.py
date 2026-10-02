"""Thin adapters for the vendored DSM ToG and ToG-2 implementations.

The vendored runners remain the source of truth for traversal, pruning and
reasoning.  This module only translates the common benchmark interface into
their existing JSON trace interfaces; it deliberately does not reimplement a
second ToG algorithm in ``methods``.
"""

from __future__ import annotations

import json
import csv
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from methods.base import BenchmarkSample, MethodResult
from config.settings import DatabaseSettings
from kg_pipeline.vector_store import MilvusConfig
from kg_pipeline.minimal_graph_export import database_name_from_env, target_dsn


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "_tog_vendor"
LABEL_RE = re.compile(r"\{\s*([^{}\n]+?)\s*\}")


def _project_backend_env(env: dict[str, str]) -> None:
    """Populate vendored ToG subprocesses from the project's Settings/.env."""
    db = DatabaseSettings()
    milvus = MilvusConfig()
    # HippoRAG uses the normalized DSN as its source, then projects it to the
    # minimal four-table graph database.  The vendored ToG SQL layer expects
    # that same minimal schema (facts/entities/mentions/chunks), so it must
    # receive the projected DSN rather than the normalized source database.
    env.setdefault(
        "DSM_KG_DATABASE_URL",
        target_dsn(db.database_url, database_name_from_env()),
    )
    env.setdefault("MILVUS_URI", milvus.uri)
    env.setdefault("MILVUS_TOKEN", milvus.token)
    env["MILVUS_ENTITY_COLLECTION"] = "canonical_seed_entities_v1"


def _labels(sample: BenchmarkSample) -> list[str]:
    return list(sample.valid_labels) or ([sample.gold_label] if sample.gold_label else [])


def _trace_result(trace: dict, *, method: str, elapsed_ms: float) -> MethodResult:
    final = trace.get("final") or {}
    label = str(final.get("pred_label") or "").strip()
    allowed = trace.get("labels") or []
    labels = [label] if label and label != "NULL" else []
    if allowed and labels and label.casefold() not in {str(x).casefold() for x in allowed}:
        labels = []
    raw = str(final.get("raw_answer") or "")
    steps = trace.get("steps") or {}
    depths = trace.get("depths") or []
    llm_calls = trace.get("llm_calls") or []
    usage = trace.get("token_totals") or {}
    triples = []
    paths = []
    for depth in depths:
        for item in depth.get("kept", []):
            path = item.get("path") or []
            paths.append({"depth": depth.get("depth"), "entity_id": item.get("id"), "path": path})
            triples.extend({"text": sentence, "depth": depth.get("depth")} for sentence in path)
    passages = []
    initial = steps.get("initial_context") or {}
    passages.extend(initial.get("selected") or [])
    for depth in depths:
        passages.extend(depth.get("top_context") or [])
    return MethodResult(
        seed_entities=[{"id": k, "text": v} for k, v in (steps.get("topic_entity") or {}).items()],
        triples=triples,
        passages=passages,
        paths=paths,
        labels=labels,
        explanation=str(final.get("explanation") or raw),
        raw_response=raw,
        parse_status="ok" if labels else "empty",
        stopping_reason=str(final.get("stop_reason") or final.get("end_mode") or "completed"),
        llm_calls=int(usage.get("calls") or len(llm_calls)),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        traversal_steps=len(depths),
        errors=[str(final["error"])] if final.get("error") else [],
        method_metadata={"adapter": method, "trace_version": trace.get("trace_version"), "end_mode": final.get("end_mode")},
        stage_outputs={"trace": trace},
        stages=[{"stage": "tog_trace", "status": "ok" if not final.get("error") else "error",
                  "latency_ms": round(elapsed_ms), "llm_calls": int(usage.get("calls") or len(llm_calls)),
                  "input_tokens": int(usage.get("input_tokens") or 0),
                  "output_tokens": int(usage.get("output_tokens") or 0),
                  "details": {"depths": len(depths)}}],
    )


class _SubprocessToG:
    name = "tog"

    def __init__(self, *, config_overrides=None, stop_after=None, generate_answer=False):
        self.config = dict(config_overrides or {})
        self.stop_after = stop_after
        self.generate_answer = generate_answer
        self.startup_stages = []
        self.last_stages = []

    def close(self):
        return None

    def _run(self, sample: BenchmarkSample, command: list[str], cwd: Path, env: dict[str, str]) -> MethodResult:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"{self.name}-") as temp:
            output = Path(temp) / "trace.json"
            command += ["--out", str(output)] if self.name == "tog" else ["--output", str(output)]
            process = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
            if not output.exists():
                detail = (process.stderr or process.stdout or "runner produced no trace").strip()[-2000:]
                return MethodResult(errors=[detail], stopping_reason="error")
            try:
                trace = json.loads(output.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return MethodResult(errors=[f"invalid ToG trace: {exc}"], stopping_reason="error")
            result = _trace_result(trace, method=self.name, elapsed_ms=(time.perf_counter() - started) * 1000)
            if process.returncode and not result.errors:
                result.errors.append((process.stderr or process.stdout or f"runner exited {process.returncode}").strip()[-2000:])
                result.stopping_reason = "error"
            self.last_stages = result.stages
            return result


class ToGMethod(_SubprocessToG):
    name = "tog"

    def run(self, sample: BenchmarkSample) -> MethodResult:
        env = os.environ.copy()
        _project_backend_env(env)
        # ToG entity linking must use the consolidated canonical index.  Keep
        # the project's unconsolidated setting available to other methods.
        env["MILVUS_ENTITY_COLLECTION"] = "canonical_seed_entities_v1"
        env["PYTHONPATH"] = str(ROOT.parent) + os.pathsep + str(VENDOR / "tog1") + os.pathsep + env.get("PYTHONPATH", "")
        command = [sys.executable, "run_trace.py", "--question", sample.text,
                   "--gold", sample.gold_label or "", "--dataset", sample.dataset,
                   "--bench", sample.sample_id, "--width", str(self.config.get("width", 3)),
                   "--depth", str(self.config.get("depth", 2))]
        return self._run(sample, command, VENDOR / "tog1", env)


class ToG2Method(_SubprocessToG):
    name = "tog2"

    def run(self, sample: BenchmarkSample) -> MethodResult:
        labels = _labels(sample)
        env = os.environ.copy()
        _project_backend_env(env)
        # ToG-2 reuses the ToG-1 Milvus linker; force canonical entities here
        # instead of inheriting MILVUS_ENTITY_COLLECTION from .env.
        env["MILVUS_ENTITY_COLLECTION"] = "canonical_seed_entities_v1"
        env["PYTHONPATH"] = str(ROOT.parent) + os.pathsep + str(VENDOR / "tog2") + os.pathsep + str(VENDOR / "tog1") + os.pathsep + env.get("PYTHONPATH", "")
        with tempfile.TemporaryDirectory(prefix="tog2-input-") as temp:
            source = Path(temp) / "sample.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("benchmark_id", "source_row_index", "query", "gold_label"))
                writer.writeheader()
                writer.writerow({"benchmark_id": sample.sample_id, "source_row_index": sample.source_row_index or "",
                                 "query": sample.text, "gold_label": sample.gold_label or ""})
            command = [sys.executable, "dsm_pipeline.py", "--dataset", sample.dataset,
                       "--csv", str(source), "--row", "0", "--width", str(self.config.get("width", 5)),
                       "--depth", str(self.config.get("depth", 5))]
            return self._run(sample, command, VENDOR / "tog2", env)
