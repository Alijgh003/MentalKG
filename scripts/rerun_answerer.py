#!/usr/bin/env python3
"""Rerun EvidenceAnswerer on an existing method-run JSONL with a (bigger) LLM.

Does NOT re-do retrieval. Reads final chunks already stored in the run file
(HippoRAG: passage_ranking, Vanilla: reranking) and calls EvidenceAnswerer again.

The bigger model is configured via env (same as normal runs):
  LLM_MODEL, LLM_API_BASE, LLM_API_KEY, LLM_TIMEOUT, LLM_RETRY_ATTEMPTS
Or override via CLI --answerer-model / --answerer-base-url / --answerer-api-key

Usage:
  .venv/bin/python scripts/rerun_answerer.py \
    --input outputs/method_runs/hipporag2-SWMH-test-20260923T140058Z.jsonl \
    --output outputs/method_runs/hipporag2-SWMH-test-20260923T140058Z-rerun-bigger.jsonl \
    --answerer-model openai/gpt-oss-120b --answerer-base-url http://192.168.1.204:43375/v1

  # or via env:
  LLM_MODEL=openai/gpt-4o LLM_API_BASE=https://api.openai.com/v1 LLM_API_KEY=sk-... \
    .venv/bin/python scripts/rerun_answerer.py --input ... --output ...

Outputs a new JSONL with updated `output`, `stage_outputs.answer_generation`,
and `versions.model` (bigger model name). Keeps original `retrieval`/`passages` intact.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Ensure env file loaded before Settings
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def build_answerer_with_override(model=None, base_url=None, api_key=None, timeout=None, provider_only=None, provider_allow_fallbacks=None):
    """Create EvidenceAnswerer using overridden Settings env.

    Supports dedicated rerun env vars (RERUN_LLM_*) so main .env LLM_MODEL stays untouched:
      RERUN_LLM_MODEL, RERUN_LLM_API_BASE, RERUN_LLM_API_KEY, RERUN_LLM_TIMEOUT,
      RERUN_LLM_PROVIDER_ONLY, RERUN_LLM_PROVIDER_ALLOW_FALLBACKS
    CLI args take precedence over those.
    """
    # Dedicated rerun env vars (use another env variable as requested)
    rerun_model = os.getenv("RERUN_LLM_MODEL")
    rerun_base = os.getenv("RERUN_LLM_API_BASE")
    rerun_key = os.getenv("RERUN_LLM_API_KEY")
    rerun_timeout = os.getenv("RERUN_LLM_TIMEOUT")
    rerun_provider_only = os.getenv("RERUN_LLM_PROVIDER_ONLY")
    rerun_provider_allow = os.getenv("RERUN_LLM_PROVIDER_ALLOW_FALLBACKS")
    if rerun_model and not model:
        model = rerun_model
    if rerun_base and not base_url:
        base_url = rerun_base
    if rerun_key and not api_key:
        api_key = rerun_key
    if rerun_timeout and not timeout:
        timeout = int(rerun_timeout)
    if rerun_provider_only and provider_only is None:
        provider_only = rerun_provider_only
    if rerun_provider_allow is not None and provider_allow_fallbacks is None:
        # will be handled below
        pass
    # Patch env for Settings
    if model:
        os.environ["LLM_MODEL"] = model
    if base_url:
        os.environ["LLM_API_BASE"] = base_url
    if api_key:
        os.environ["LLM_API_KEY"] = api_key
    if timeout:
        os.environ["LLM_TIMEOUT"] = str(timeout)
    if provider_only is not None:
        os.environ["LLM_PROVIDER_ONLY"] = provider_only
    if provider_allow_fallbacks is not None:
        os.environ["LLM_PROVIDER_ALLOW_FALLBACKS"] = str(provider_allow_fallbacks).lower()
    elif rerun_provider_allow is not None:
        os.environ["LLM_PROVIDER_ALLOW_FALLBACKS"] = rerun_provider_allow
    from methods.hipporag.answering import EvidenceAnswerer

    return EvidenceAnswerer()


def main():
    ap = argparse.ArgumentParser(description="Rerun answerer on stored chunks")
    ap.add_argument("--input", required=True, help="Input run JSONL")
    ap.add_argument("--output", required=True, help="Output rerun JSONL")
    ap.add_argument("--answerer-model", default=None, help="Override LLM_MODEL")
    ap.add_argument("--answerer-base-url", default=None, help="Override LLM_API_BASE")
    ap.add_argument("--answerer-api-key", default=None, help="Override LLM_API_KEY")
    ap.add_argument("--answerer-timeout", type=int, default=None)
    ap.add_argument("--qa-top-k", type=int, default=None, help="Override how many top chunks to send (default: whatever run used)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build answerer
    answerer = build_answerer_with_override(
        model=args.answerer_model,
        base_url=args.answerer_base_url,
        api_key=args.answerer_api_key,
        timeout=args.answerer_timeout,
    )
    from config.settings import Settings
    settings = Settings()
    print(f"Answerer model: {settings.llm_model} @ {settings.llm_api_base}")

    # Read input
    lines = [l for l in input_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        lines = lines[:args.limit]

    with open(output_path, "w", encoding="utf-8") as out:
        for idx, line in enumerate(lines):
            rec = json.loads(line)
            sample_text = rec.get("input", {}).get("text", "")
            valid_labels = tuple(rec.get("gold", {}).get("valid_labels") or rec.get("input", {}).get("valid_labels") or [])
            # Fallback: try to get valid_labels from stage_outputs answer_generation or retrieval
            if not valid_labels:
                # Try to infer from framework? Use dataset labels
                valid_labels = tuple(rec.get("output", {}).get("valid_labels") or [])
            # More robust: read from record's `versions`? Let's try to get from retrieval stage
            # Actually BenchmarkSample valid_labels stored in original run's stage_outputs.answer_generation.details.valid_labels
            if not valid_labels:
                details = rec.get("stage_outputs", {}).get("answer_generation", {})
                if isinstance(details, dict) and "valid_labels" in details:
                    valid_labels = tuple(details["valid_labels"])
            if not valid_labels:
                # Last fallback: use gold label vocabulary from input? Just use gold + generic
                valid_labels = tuple(rec.get("input", {}).get("valid_labels", []))

            # Determine evidence: HippoRAG uses passage_ranking, Vanilla uses reranking
            stage = rec.get("stage_outputs", {})
            if "reranking" in stage and stage["reranking"]:
                ranked = stage["reranking"]
                # vanilla: reranked entries have passage_id/text
                passages = [{"passage_id": r["passage_id"], "text": r.get("text",""), "score": r.get("score",0)} for r in ranked]
            elif "passage_ranking" in stage and stage["passage_ranking"]:
                passages = [{"passage_id": r["passage_id"], "text": r.get("text",""), "score": r.get("score",0)} for r in stage["passage_ranking"]]
            else:
                # Fallback to retrieval.passages
                passages = rec.get("retrieval", {}).get("passages", [])
                passages = [{"passage_id": p.get("passage_id", p.get("id","")), "text": p.get("text",""), "score": p.get("score",0)} for p in passages]

            qa_k = args.qa_top_k
            if qa_k is None:
                # infer from previous answer_generation evidence length
                prev_ev = stage.get("answer_generation", {}).get("evidence_passage_ids", [])
                qa_k = len(prev_ev) if prev_ev else 5
            evidence = passages[:qa_k]

            # Call answerer
            result = answerer.answer(sample_text, evidence, valid_labels)

            # Update record
            normalized = result.answer.casefold().strip().rstrip(".") if result.answer else ""
            rec["output"] = {
                "labels": [normalized] if normalized else [],
                "label_scores": {},
                "explanation": result.explanation,
                "raw_response": result.raw_response,
                "parse_status": "ok" if normalized and not result.errors else ("invalid_label" if normalized else "empty"),
            }
            rec["stage_outputs"]["answer_generation"] = {
                "answer": result.answer,
                "explanation": result.explanation,
                "cited_passage_ids": list(result.cited_passage_ids),
                "evidence_passage_ids": [p["passage_id"] for p in evidence],
                "rerun_model": settings.llm_model,
                "rerun_input_tokens": result.input_tokens,
                "rerun_output_tokens": result.output_tokens,
                "errors": list(result.errors),
            }
            rec["versions"] = rec.get("versions", {})
            rec["versions"]["rerun_model"] = settings.llm_model
            rec["versions"]["rerun_api_base"] = settings.llm_api_base
            rec["cost"] = rec.get("cost", {})
            rec["cost"]["rerun_llm_calls"] = 1
            rec["cost"]["rerun_input_tokens"] = result.input_tokens
            rec["cost"]["rerun_output_tokens"] = result.output_tokens
            if result.errors:
                rec.setdefault("errors", []).extend(list(result.errors))

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{idx+1}/{len(lines)}] {rec.get('sample_id')} gold={rec.get('gold',{}).get('label')} -> {rec['output']['labels']} ({rec['output']['parse_status']})")

    print(f"Done. Wrote {len(lines)} records to {output_path}")


if __name__ == "__main__":
    main()
