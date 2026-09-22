"""LocalGraphEval: explanation claim support against a gold explanation graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from methods.dspy_runtime import build_lm, token_usage


@dataclass(frozen=True)
class Claim:
    claim_id: str
    sentence: str
    subject: str
    predicate: str
    object: str

    def as_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


def _claims(values: Any, prefix: str) -> list[Claim]:
    output: list[Claim] = []
    for index, value in enumerate(values or [], start=1):
        if not isinstance(value, dict):
            continue
        sentence = str(value.get("sentence", "")).strip()
        subject = str(value.get("subject", "")).strip()
        predicate = str(value.get("predicate", "")).strip()
        object_ = str(value.get("object", "")).strip()
        if sentence and subject and predicate and object_:
            output.append(Claim(f"{prefix}{index}", sentence, subject, predicate, object_))
    return output


def _entities(values: Any) -> list[dict[str, str]]:
    entities = []
    seen = set()
    for value in values or []:
        if isinstance(value, dict):
            text = str(value.get("text", "")).strip()
            entity_type = str(value.get("type", "concept")).strip() or "concept"
        else:
            text, entity_type = str(value).strip(), "concept"
        key = (text.casefold(), entity_type.casefold())
        if text and key not in seen:
            seen.add(key)
            entities.append({"text": text, "type": entity_type})
    return entities


class LocalGraphEvaluator:
    def __init__(self):
        import dspy

        class ExtractClaims(dspy.Signature):
            """Convert an explanation into a small knowledge graph.

            Extract the entities explicitly mentioned or needed by the reasoning,
            then extract every factual/clinical relation between those entities,
            including supporting facts used to reach the final diagnosis. Preserve
            negation and uncertainty in the predicate. Do not add outside knowledge.
            Every relation must have a faithful source sentence from the explanation.
            """

            explanation: str = dspy.InputField()
            entities: list[dict[str, str]] = dspy.OutputField(
                desc="Unique objects with text and type fields; type is concept, symptom, disorder, behavior, risk, treatment, or context"
            )
            relations: list[dict[str, str]] = dspy.OutputField(
                desc="Exhaustive factual graph edges with sentence, subject, predicate, and object fields"
            )

        class JudgeClaim(dspy.Signature):
            """Judge whether a generated claim is supported by the gold claims."""

            generated_claim: str = dspy.InputField()
            gold_claims: str = dspy.InputField()
            status: str = dspy.OutputField(
                desc="Exactly one of SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, UNSUPPORTED"
            )
            matched_gold_claim_ids: list[str] = dspy.OutputField()
            reasoning: str = dspy.OutputField(desc="One concise sentence")

        self.lm = build_lm()
        self.extractor = dspy.Predict(ExtractClaims)
        self.judge = dspy.Predict(JudgeClaim)

    def extract(self, explanation: str, prefix: str) -> tuple[list[Claim], dict[str, int]]:
        import dspy

        with dspy.context(lm=self.lm):
            prediction = self.extractor(explanation=explanation)
        input_tokens, output_tokens = token_usage(self.lm)
        return _entities(prediction.entities), _claims(prediction.relations, prefix), {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def evaluate_claim(self, claim: Claim, gold: list[Claim]) -> tuple[dict[str, Any], dict[str, int]]:
        import dspy

        with dspy.context(lm=self.lm):
            prediction = self.judge(
                generated_claim=json.dumps(claim.as_dict(), ensure_ascii=False),
                gold_claims=json.dumps([item.as_dict() for item in gold], ensure_ascii=False),
            )
        input_tokens, output_tokens = token_usage(self.lm)
        allowed = {"SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "UNSUPPORTED"}
        status = str(prediction.status).strip().upper()
        if status not in allowed:
            status = "UNSUPPORTED"
        matched = [str(item) for item in (prediction.matched_gold_claim_ids or [])]
        return {
            "claim": claim.as_dict(),
            "status": status,
            "matched_gold_claim_ids": matched,
            "reasoning": str(prediction.reasoning).strip(),
        }, {"input_tokens": input_tokens, "output_tokens": output_tokens}


def evaluate_records(evaluator: LocalGraphEvaluator, records: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    total_input = total_output = 0
    aggregate = {"generated": 0, "gold": 0, "strict_supported": 0, "soft_supported": 0, "covered_gold": 0}
    for record in records:
        gold_text = record.get("gold_explanation", "")
        generated_text = record.get("generated_explanation", "")
        gold_entities, gold, usage = evaluator.extract(gold_text, "G")
        total_input += usage["input_tokens"]; total_output += usage["output_tokens"]
        generated_entities, generated, usage = evaluator.extract(generated_text, "C")
        total_input += usage["input_tokens"]; total_output += usage["output_tokens"]
        judged = []
        for claim in generated:
            item, usage = evaluator.evaluate_claim(claim, gold)
            total_input += usage["input_tokens"]; total_output += usage["output_tokens"]
            judged.append(item)
        supported = sum(item["status"] == "SUPPORTED" for item in judged)
        partial = sum(item["status"] == "PARTIALLY_SUPPORTED" for item in judged)
        matched_strict = {
            gold_id
            for item in judged
            if item["status"] == "SUPPORTED"
            for gold_id in item["matched_gold_claim_ids"]
        }
        matched_soft = {
            gold_id
            for item in judged
            if item["status"] in {"SUPPORTED", "PARTIALLY_SUPPORTED"}
            for gold_id in item["matched_gold_claim_ids"]
        }
        generated_count = len(judged)
        gold_count = len(gold)
        strict_precision = supported / generated_count if generated_count else 0.0
        soft_precision = (supported + 0.5 * partial) / generated_count if generated_count else 0.0
        strict_recall = len(matched_strict) / gold_count if gold_count else 0.0
        soft_recall = len(matched_soft) / gold_count if gold_count else 0.0

        def f1(precision: float, recall: float) -> float:
            return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        aggregate["generated"] += generated_count
        aggregate["gold"] += gold_count
        aggregate["strict_supported"] += supported
        aggregate["soft_supported"] += supported + 0.5 * partial
        aggregate["covered_gold"] += len(matched_strict)
        results.append({
            "sample_id": record["sample_id"],
            "gold_entities": gold_entities,
            "generated_entities": generated_entities,
            "gold_claims": [item.as_dict() for item in gold],
            "generated_claims": judged,
            "summary": {
                "generated_claims": len(judged),
                "supported": supported,
                "partially_supported": partial,
                "contradicted": sum(item["status"] == "CONTRADICTED" for item in judged),
                "unsupported": sum(item["status"] == "UNSUPPORTED" for item in judged),
                "strict_precision": strict_precision,
                "strict_recall": strict_recall,
                "strict_f1": f1(strict_precision, strict_recall),
                "soft_precision": soft_precision,
                "soft_recall": soft_recall,
                "soft_f1": f1(soft_precision, soft_recall),
                "covered_gold_claim_ids": sorted(matched_strict),
            },
        })
    micro_precision = aggregate["strict_supported"] / aggregate["generated"] if aggregate["generated"] else 0.0
    micro_recall = aggregate["covered_gold"] / aggregate["gold"] if aggregate["gold"] else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    macro = lambda key: sum(float(item["summary"][key]) for item in results) / len(results) if results else 0.0
    return {
        "metric": "LocalGraphEval",
        "samples": len(results),
        "results": results,
        "aggregate": {
            "micro_strict_precision": micro_precision,
            "micro_strict_recall": micro_recall,
            "micro_strict_f1": micro_f1,
            "macro_strict_precision": macro("strict_precision"),
            "macro_strict_recall": macro("strict_recall"),
            "macro_strict_f1": macro("strict_f1"),
            "macro_soft_precision": macro("soft_precision"),
            "macro_soft_recall": macro("soft_recall"),
            "macro_soft_f1": macro("soft_f1"),
            "total_generated_claims": aggregate["generated"],
            "total_gold_claims": aggregate["gold"],
            "total_covered_gold_claims": aggregate["covered_gold"],
        },
        "cost": {"input_tokens": total_input, "output_tokens": total_output},
    }
