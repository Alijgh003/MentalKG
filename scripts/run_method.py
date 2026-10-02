#!/usr/bin/env python3
"""Run any registered retrieval method on a benchmark split."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from methods import available_methods, create_method
from methods.datasets import available_datasets, load_samples


STAGE_DISPLAY_LIMITS = {"passage_retrieval": 10}


def stage_output_for_display(stage: str, output):
    """Keep artifacts complete while making verbose stage previews readable."""
    limit = STAGE_DISPLAY_LIMITS.get(stage)
    if limit is None or not isinstance(output, list):
        return output, None
    return output[:limit], {
        "showing": min(len(output), limit),
        "total": len(output),
        "truncated": len(output) > limit,
    }


def parse_scalar(value: str):
    lowered = value.casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_options(values: list[str]) -> dict:
    options = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Method option must use key=value syntax: {value!r}")
        key, raw = value.split("=", 1)
        if not key or key in options:
            raise ValueError(f"Invalid or duplicate method option: {key!r}")
        options[key] = parse_scalar(raw)
    return options


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run a retrieval method on a frozen benchmark")
    result.add_argument("--method", choices=available_methods())
    result.add_argument("--dataset")
    result.add_argument("--split", default="test")
    size = result.add_mutually_exclusive_group()
    size.add_argument("--limit", type=int, help="Number of samples; default is 10")
    size.add_argument("--all", action="store_true", help="Run the complete split")
    size.add_argument(
        "--row-indices",
        help="Comma-separated zero-based dataset row indices (useful for stratified samples)",
    )
    result.add_argument("--offset", type=int, default=0)
    result.add_argument("--method-option", action="append", default=[], metavar="KEY=VALUE")
    result.add_argument(
        "--stop-after",
        choices=("fact_generation", "fact_retrieval", "passage_retrieval", "reranking", "seed_weighting", "ppr", "passage_ranking", "answer_generation"),
        help="Stop at this pipeline boundary and print that stage's output",
    )
    result.add_argument("--output", type=Path)
    result.add_argument("--generate-answer", action="store_true", help="Generate a final answer from the top ranked passages")
    result.add_argument("--list", action="store_true", help="List methods and dataset splits")
    result.add_argument("--verbose", action="store_true")
    return result


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def record_for(sample, result, *, run_id: str, method: str, latency_ms: int, commit: str, options: dict):
    component_stages = [stage for stage in result.stages if stage.get("stage") != "sample_total"]
    retrieval_calls = sum(int(stage.get("retrieval_calls", 0)) for stage in component_stages)
    embedding_calls = sum(int(stage.get("embedding_calls", 0)) for stage in component_stages)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "sample_id": sample.sample_id,
        "dataset_row_index": sample.dataset_row_index,
        "source_row_index": sample.source_row_index,
        "dataset": sample.dataset.lower(),
        "split": sample.split,
        "framework": method,
        "bridge_enabled": False,
        "input": {
            "text_hash": hashlib.sha256(sample.text.encode()).hexdigest(),
            "text": sample.text,
        },
        "gold": {"label": sample.gold_label},
        "observations": [{
            "observation_id": "o1",
            "source_type": "raw_input",
            "evidence_span": sample.text,
            "char_start": 0,
            "char_end": len(sample.text),
            "observation": sample.text,
        }],
        "retrieval": {
            "queries": [{"query_id": "q1", "source_observation_ids": ["o1"], "text": sample.text}],
            "seed_entities": result.seed_entities,
            "triples": result.triples,
            "passages": result.passages,
            "method_metadata": result.method_metadata,
        },
        "stage_outputs": result.stage_outputs,
        "reasoning": {
            "paths": result.paths,
            "cited_evidence_ids": [],
            "stopping_reason": result.stopping_reason,
        },
        "output": {
            "labels": result.labels,
            "label_scores": result.label_scores,
            "explanation": result.explanation,
            "raw_response": result.raw_response,
            "parse_status": result.parse_status,
        },
        "cost": {
            "latency_ms": latency_ms,
            "llm_calls": result.llm_calls,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "retrieval_calls": retrieval_calls,
            "embedding_calls": embedding_calls,
            "traversal_steps": result.traversal_steps,
            "stages": result.stages,
        },
        "versions": {
            "code_commit": commit,
            "graph_snapshot": "dsm5_minimal_graph",
            "index_snapshot": "milvus-current",
            "model": None,
            "prompt": "hipporag2-recognition-v1",
            "config": options,
        },
        "errors": result.errors,
    }


def event_record(event: dict, *, run_id: str, method: str, dataset: str, split: str, sample=None) -> dict:
    return {
        "run_id": run_id,
        "framework": method,
        "dataset": dataset,
        "split": split,
        "sample_id": sample.sample_id if sample is not None else None,
        "dataset_row_index": sample.dataset_row_index if sample is not None else None,
        "source_row_index": sample.source_row_index if sample is not None else None,
        **event,
    }


def main() -> int:
    arg_parser = parser()
    args = arg_parser.parse_args()
    if args.list:
        print("Methods:")
        for method in available_methods():
            print(f"  {method}")
        print("Datasets:")
        for dataset, split in available_datasets():
            print(f"  {dataset}/{split}")
        return 0
    if not args.method or not args.dataset:
        arg_parser.error("--method and --dataset are required unless --list is used")
    if args.limit is not None and args.limit <= 0:
        arg_parser.error("--limit must be positive")
    if args.offset < 0:
        arg_parser.error("--offset cannot be negative")

    options = parse_options(args.method_option)
    if args.row_indices:
        try:
            row_indices = [int(value.strip()) for value in args.row_indices.split(",") if value.strip()]
        except ValueError as error:
            arg_parser.error(f"--row-indices must contain integers: {error}")
        if not row_indices or any(index < 0 for index in row_indices):
            arg_parser.error("--row-indices must contain at least one non-negative integer")
        if len(set(row_indices)) != len(row_indices):
            arg_parser.error("--row-indices must not contain duplicates")
        all_samples = load_samples(args.dataset, args.split, limit=None, offset=0)
        by_index = {sample.dataset_row_index: sample for sample in all_samples}
        missing = [index for index in row_indices if index not in by_index]
        if missing:
            arg_parser.error(f"Dataset rows not found: {missing}")
        samples = [by_index[index] for index in row_indices]
    else:
        limit = None if args.all else (args.limit or 10)
        samples = load_samples(args.dataset, args.split, limit=limit, offset=args.offset)
    if not samples:
        raise SystemExit("No samples selected")

    run_id = str(uuid.uuid4())
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("outputs") / "method_runs" / f"{args.method}-{args.dataset}-{args.split}-{timestamp}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    event_output = output.with_name(f"{output.stem}.events.jsonl")
    log_output = output.with_name(f"{output.stem}.log")
    handlers: list[logging.Handler] = [logging.StreamHandler(), logging.FileHandler(log_output, encoding="utf-8")]
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    commit = git_commit()
    logging.info(
        "run_start run_id=%s method=%s dataset=%s split=%s samples=%s output=%s",
        run_id, args.method, args.dataset, args.split, len(samples), output,
    )
    method = create_method(
        args.method,
        config_overrides=options,
        stop_after=args.stop_after,
        generate_answer=args.generate_answer,
    )
    completed = 0
    try:
        with output.open("w", encoding="utf-8") as handle, event_output.open("w", encoding="utf-8") as event_handle:
            for startup_event in getattr(method, "startup_stages", []):
                event_handle.write(json.dumps(event_record(
                    startup_event,
                    run_id=run_id,
                    method=args.method,
                    dataset=args.dataset,
                    split=args.split,
                ), ensure_ascii=False) + "\n")
            event_handle.flush()
            for sample in samples:
                logging.info(
                    "sample_start run_id=%s dataset=%s split=%s dataset_row=%s source_row=%s sample_id=%s",
                    run_id, sample.dataset, sample.split, sample.dataset_row_index,
                    sample.source_row_index, sample.sample_id,
                )
                sample_started_at = datetime.now(UTC).isoformat()
                started = time.perf_counter()
                try:
                    result = method.run(sample)
                except Exception as error:
                    logging.exception("Method failed for sample %s", sample.sample_id)
                    from methods.base import MethodResult

                    result = MethodResult(
                        errors=[f"{type(error).__name__}: {error}"],
                        stopping_reason="error",
                        stages=list(getattr(method, "last_stages", [])),
                    )
                latency_ms = round((time.perf_counter() - started) * 1000)
                result.stages.append({
                    "stage": "sample_total",
                    "started_at": sample_started_at,
                    "ended_at": datetime.now(UTC).isoformat(),
                    "status": "error" if result.errors and result.stopping_reason == "error" else "ok",
                    "latency_ms": latency_ms,
                    "llm_calls": result.llm_calls,
                    "embedding_calls": sum(int(stage.get("embedding_calls", 0)) for stage in result.stages),
                    "retrieval_calls": sum(int(stage.get("retrieval_calls", 0)) for stage in result.stages),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "details": {"stopping_reason": result.stopping_reason},
                })
                for stage_event in result.stages:
                    event_handle.write(json.dumps(event_record(
                        stage_event,
                        run_id=run_id,
                        method=args.method,
                        dataset=sample.dataset,
                        split=sample.split,
                        sample=sample,
                    ), ensure_ascii=False) + "\n")
                event_handle.flush()
                handle.write(json.dumps(record_for(
                    sample,
                    result,
                    run_id=run_id,
                    method=args.method,
                    latency_ms=latency_ms,
                    commit=commit,
                    options=options,
                ), ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                logging.info(
                    "sample_end run_id=%s sample_id=%s status=%s latency_ms=%s tokens_in=%s tokens_out=%s",
                    run_id, sample.sample_id, result.stopping_reason, latency_ms,
                    result.input_tokens, result.output_tokens,
                )
                if args.stop_after:
                    displayed_output, display = stage_output_for_display(
                        args.stop_after,
                        result.stage_outputs.get(args.stop_after),
                    )
                    print(json.dumps({
                        "sample_id": sample.sample_id,
                        "stage": args.stop_after,
                        **({"display": display} if display is not None else {}),
                        "output": displayed_output,
                    }, ensure_ascii=False, indent=2))
                else:
                    print(f"[{completed}/{len(samples)}] {sample.sample_id}: {result.stopping_reason}")
    finally:
        method.close()

    if not args.stop_after:
        print(json.dumps({
            "run_id": run_id,
            "method": args.method,
            "dataset": args.dataset,
            "split": args.split,
            "samples": completed,
            "output": str(output.resolve()),
            "events": str(event_output.resolve()),
            "log": str(log_output.resolve()),
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
