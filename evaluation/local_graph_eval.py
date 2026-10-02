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
    scope: str
    sentence: str
    subject: str
    predicate: str
    object: str

    def as_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


def _claims(values: Any, prefix: str, scope: str) -> list[Claim]:
    output: list[Claim] = []
    for index, value in enumerate(values or [], start=1):
        if not isinstance(value, dict):
            continue
        sentence = str(value.get("sentence", "")).strip()
        subject = str(value.get("subject", "")).strip()
        predicate = str(value.get("predicate", "")).strip()
        object_ = str(value.get("object", "")).strip()
        if sentence and subject and predicate and object_:
            output.append(Claim(
                f"{prefix}{scope.upper()}_{index}",
                scope,
                sentence,
                subject,
                predicate,
                object_,
            ))
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

            Extract the relations explicitly stated in the explanation and divide
            them into two separate scopes.

            case_relations are claims about the person in the case, post, or
            explanation: claims whose subject is the poster, patient, individual,
            writer, or a pronoun referring to them. This includes what they report,
            experience, suffer from, deny, or are being treated for.

            clinical_relations are general clinical or evidence facts that do not
            describe the case person: for example, a symptom being associated with
            a disorder, a disorder being characterized by a symptom, or a treatment
            being used for a disorder.

            Every relation must be atomic and fine-grained. Split coordinated
            lists, multiple symptoms, and multiple objects into separate triples.
            For example, do not return one triple whose object is "not wanting to
            grow up, not finding appeal in adult responsibilities, feeling
            disconnected from society, and having a low will to live". Return four
            separate triples, one for each fact. Each triple must have exactly one
            entity-like subject, one concise predicate, and one entity-like object.
            Do not put a list, conjunction, or full explanation sentence in subject
            or object.

            For a general, potentially ambiguous entity (especially a common
            emotion, behavior, experience, or colloquial phrase), append a
            brief parenthetical semantic gloss of two or three words to the
            entity text in the extracted relation, while preserving the
            original phrase. Use the form ``envy (wanting others' advantages)``
            or ``resentment (hidden lasting anger)``. Do this only when the
            gloss disambiguates the concept for semantic retrieval; do not add
            glosses to explicit disorder names, established clinical terms,
            named people, or entities whose meaning is already specific. Keep
            the gloss faithful to the explanation and do not introduce outside
            diagnoses or unsupported facts. Apply the same enriched text in
            the corresponding entity list.

            Resolve discourse references before extracting relations. Expressions
            such as "these observations", "these symptoms", "this", "they", "it",
            and "the above" must be linked to their explicitly stated antecedents
            in the surrounding text. When a later sentence says that such an
            expression aligns with, suggests, supports, or is consistent with a
            clinical condition, expand that relation to every atomic antecedent.
            For example, if the text says "The poster describes isolation and a
            low will to live. These observations align with depressive disorders",
            extract both:
            "isolation -> aligns with -> depressive disorders" and
            "low will to live -> aligns with -> depressive disorders".
            Do not replace these antecedents with a generic phrase such as
            "these observations", and do not omit the expanded relations.

            Keep the scopes separate. If one sentence contains both kinds of facts,
            extract separate relations into the appropriate lists. Preserve negation
            and uncertainty in the predicate. Do not add outside knowledge. Every
            relation must have a faithful source sentence from the explanation.
            """

            explanation: str = dspy.InputField()
            case_entities: list[dict[str, str]] = dspy.OutputField(
                desc="Unique case/person objects with text and type fields"
            )
            clinical_entities: list[dict[str, str]] = dspy.OutputField(
                desc="Unique clinical objects with text and type fields"
            )
            case_relations: list[dict[str, str]] = dspy.OutputField(
                desc="Claims about the case person; each has sentence, subject, predicate, and object"
            )
            clinical_relations: list[dict[str, str]] = dspy.OutputField(
                desc="General clinical/evidence claims; each has sentence, subject, predicate, and object"
            )

        class JudgeClaim(dspy.Signature):
            """Judge a generated claim against gold claims from the same scope.

            Case claims describe the person in the case. Clinical claims describe
            general clinical or evidence facts. Never match a case claim to a
            clinical claim, or a clinical claim to a case claim.
            """

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

    def extract(self, explanation: str, prefix: str) -> tuple[dict[str, list[dict[str, str]]], list[Claim], dict[str, int]]:
        import dspy

        with dspy.context(lm=self.lm):
            prediction = self.extractor(explanation=explanation)
        input_tokens, output_tokens = token_usage(self.lm)
        claims = (
            _claims(prediction.case_relations, prefix, "case")
            + _claims(prediction.clinical_relations, prefix, "clinical")
        )
        entities = {
            "case": _entities(prediction.case_entities),
            "clinical": _entities(prediction.clinical_entities),
        }
        return entities, claims, {
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
        gold_by_scope = {
            "case": [claim for claim in gold if claim.scope == "case"],
            "clinical": [claim for claim in gold if claim.scope == "clinical"],
        }
        judged = []
        for claim in generated:
            item, usage = evaluator.evaluate_claim(claim, gold_by_scope[claim.scope])
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

        by_scope = {}
        for scope in ("case", "clinical"):
            scoped_generated = [item for item in judged if item["claim"]["scope"] == scope]
            scoped_gold = gold_by_scope[scope]
            scoped_supported = sum(item["status"] == "SUPPORTED" for item in scoped_generated)
            scoped_partial = sum(item["status"] == "PARTIALLY_SUPPORTED" for item in scoped_generated)
            scoped_gold_ids = {claim.claim_id for claim in scoped_gold}
            scoped_matched = {
                gold_id
                for item in scoped_generated
                if item["status"] == "SUPPORTED"
                for gold_id in item["matched_gold_claim_ids"]
                if gold_id in scoped_gold_ids
            }
            scoped_precision = (
                scoped_supported / len(scoped_generated) if scoped_generated else 0.0
            )
            scoped_recall = (
                len(scoped_matched) / len(scoped_gold) if scoped_gold else 0.0
            )
            by_scope[scope] = {
                "generated_claims": len(scoped_generated),
                "gold_claims": len(scoped_gold),
                "supported": scoped_supported,
                "partially_supported": scoped_partial,
                "strict_precision": scoped_precision,
                "strict_recall": scoped_recall,
                "strict_f1": f1(scoped_precision, scoped_recall),
                "covered_gold_claim_ids": sorted(scoped_matched),
            }

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
                "by_scope": by_scope,
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
