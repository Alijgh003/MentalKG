"""ToG-1-compatible JSON trace for DSPy-powered ToG-2 calls.

Every actual DSPy LM history entry is saved, including full request messages,
raw provider response when available, parsed Signature output and provider
reported input/output token usage. Missing usage is null, never invented.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import dspy


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    return str(value)


def _get(value: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(value, Mapping) and value.get(key) is not None:
            return value[key]
        try:
            result = getattr(value, key, None)
            if result is not None:
                return result() if callable(result) else result
        except Exception:
            pass
    return None


def _usage(entry: Any) -> tuple[int | None, int | None]:
    usage = _get(entry, "usage")
    if usage is None:
        usage = _get(_get(entry, "response"), "usage")
    if usage is None and isinstance(entry, Mapping):
        usage = next(
            (value for value in entry.values()
             if _get(value, "prompt_tokens", "input_tokens") is not None), None
        )
    prompt = _get(usage, "prompt_tokens", "input_tokens")
    completion = _get(usage, "completion_tokens", "output_tokens")
    try:
        return (int(prompt) if prompt is not None else None,
                int(completion) if completion is not None else None)
    except (TypeError, ValueError):
        return None, None


def new_trace(record: dict, model: str) -> dict:
    """Use the same top-level contract as ToG-1's run_dspy.py JSON."""
    return {
        "question": record.get("raw_question", record["question"]),
        "classification_question": record["question"],
        "gold_label": record.get("gold_label", ""),
        "benchmark_id": record.get("benchmark_id", ""),
        "dataset": record["dataset"],
        "labels": list(record["allowed_labels"]),
        "llm_model": model,
        "engine": "dspy",
        "llm_calls": [],
        "token_totals": {"input_tokens": 0, "output_tokens": 0, "calls": 0,
                         "usage_complete": True, "missing_usage_calls": 0,
                         "scope": "this_run"},
        "latency_totals": {"llm_ms": 0.0, "end_to_end_ms": None},
        "steps": {},
        "depths": [],
        "final": {},
    }


def _prediction_dict(prediction: Any) -> dict:
    try:
        return dict(prediction)
    except Exception:
        return {"value": str(prediction)}


def traced_call(trace: dict, tag: str, module, **inputs):
    """Call a typed DSPy Module and record every underlying provider call."""
    lm = dspy.settings.lm
    before = len(lm.history or [])
    started = time.perf_counter()
    prediction = None
    error = None
    try:
        prediction = module(**inputs)
    except Exception as exc:
        error = exc
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    entries = list((lm.history or [])[before:])
    parsed = _jsonable(_prediction_dict(prediction)) if prediction is not None else None
    signature = getattr(module, "signature", None)
    if signature is None:
        signature = getattr(getattr(module, "predict", None), "signature", None)
    signature_name = getattr(signature, "__name__", type(module).__name__)
    full_prompt = "\n".join(
        f"{key}: {json.dumps(_jsonable(value), ensure_ascii=False)}"
        for key, value in inputs.items()
    )
    if not entries:
        trace["steps"].setdefault("llm_call_errors", []).append({
            "tag": tag, "signature": signature_name, "prompt": full_prompt,
            "error": str(error) if error is not None else "No LM history entry recorded",
            "latency_ms": latency_ms,
        })
        if error is not None:
            raise error
        return prediction
    for index, history in enumerate(entries, start=1):
        messages = _jsonable(_get(history, "messages") or []) if history else []
        instruction = "\n".join(
            str(message.get("content", "")) for message in messages
            if isinstance(message, dict) and message.get("role") in ("system", "developer")
        )
        input_tokens, output_tokens = _usage(history) if history else (None, None)
        entry = {
            "tag": tag if len(entries) == 1 else f"{tag}#{index}",
            "signature": signature_name,
            "prompt": full_prompt,
            "response": json.dumps(parsed, ensure_ascii=False) if index == len(entries) and parsed is not None else "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms if index == len(entries) else None,
            "latency_scope": "module_invocation",
            "request_messages": messages,
            "raw_response": _jsonable(_get(history, "response")) if history else None,
        }
        if instruction:
            entry["instruction"] = instruction
        if error is not None and index == len(entries):
            entry["error"] = str(error)
        trace["llm_calls"].append(entry)
        totals = trace["token_totals"]
        totals["calls"] += 1
        if input_tokens is None or output_tokens is None:
            totals["usage_complete"] = False
            totals["missing_usage_calls"] += 1
        if input_tokens is not None:
            totals["input_tokens"] += input_tokens
        if output_tokens is not None:
            totals["output_tokens"] += output_tokens
    trace["latency_totals"]["llm_ms"] = round(
        trace["latency_totals"]["llm_ms"] + latency_ms, 2
    )
    if error is not None:
        raise error
    return prediction
