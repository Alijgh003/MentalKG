"""Global factual explanation evaluation against the knowledge-graph facts.

Unlike LocalGraphEval, this evaluator does not compare a generated explanation
to a gold explanation. It evaluates only generated clinical triples: each triple
is embedded, the three nearest KG facts are retrieved from Milvus, and an LLM
judge decides whether at least one retrieved fact supports the triple.
"""

from __future__ import annotations

import json
from typing import Any

from kg_pipeline.embeddings import EmbeddingConfig, OpenAICompatibleEmbedder
from kg_pipeline.vector_store import KnowledgeGraphVectorStore, MilvusConfig
from methods.dspy_runtime import build_lm, token_usage


def render_triple(triple: dict[str, Any]) -> str:
    return " ".join(str(triple.get(key, "")).strip() for key in ("subject", "predicate", "object") if str(triple.get(key, "")).strip())


def clinical_triples(record: dict[str, Any]) -> list[dict[str, str]]:
    """Read clinical triples supplied directly by a caller.

    Accepts either ``clinical_claims`` or the LocalGraphEval-shaped
    ``clinical_relations`` field. The evaluator intentionally does not extract
    claims from prose; extraction remains a separate evaluation step.
    """
    values = record.get("clinical_claims", record.get("clinical_relations", []))
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, dict):
            continue
        triple = {key: str(value.get(key, "")).strip() for key in ("subject", "predicate", "object")}
        identity = tuple(triple[key] for key in ("subject", "predicate", "object"))
        if all(identity) and identity not in seen:
            seen.add(identity)
            output.append(triple)
    return output


class GlobalFactualEvaluator:
    def __init__(self, *, top_k: int = 3, dimension: int | None = None):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.milvus = MilvusConfig.from_env()
        self.collection = self.milvus.triple_collection
        self.dimension = dimension or self.milvus.triple_dimension
        self.store = KnowledgeGraphVectorStore(self.milvus)
        self.embedder = OpenAICompatibleEmbedder(EmbeddingConfig.from_env())
        import dspy

        class Judge(dspy.Signature):
            """Judge whether any retrieved knowledge-graph fact supports a claim."""

            generated_clinical_triple: str = dspy.InputField()
            retrieved_knowledge_graph_facts: str = dspy.InputField()
            support_score: str = dspy.OutputField(desc="An integer from 0 to 10: 0 means no entailment/support, 10 means the retrieved facts directly and completely support the claim")
            reasoning: str = dspy.OutputField(desc="Brief reasoning explaining the score, including partial or indirect support")

        self._Judge = Judge
        self.lm = None
        self.judge = None

    def _ensure_judge(self) -> None:
        if self.judge is None:
            self.lm = build_lm()
            self.judge = __import__("dspy").Predict(self._Judge)

    @staticmethod
    def _parse_support_score(value: Any) -> int:
        """Parse and clamp the judge's 0--10 support score."""
        import re
        match = re.search(r"(?:10|[0-9])", str(value))
        if not match:
            raise ValueError(f"Judge returned an invalid support score: {value!r}")
        return max(0, min(10, int(match.group(0))))

    def _retrieve(self, triples: list[dict[str, str]]) -> list[list[dict[str, Any]]]:
        texts = [render_triple(triple) for triple in triples]
        vectors = self.embedder.embed(texts, dimension=self.dimension)
        hits = self.store.search(
            self.collection,
            vectors.tolist(),
            limit=self.top_k,
            output_fields=("id", "text", "subject_text", "predicate", "object_text", "embedding_model"),
        )
        return [[dict(hit) for hit in group] for group in hits]

    def retrieve_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run only triple normalization, embedding, and Milvus retrieval."""
        results = []
        for record in records:
            triples = clinical_triples(record)
            retrieved = self._retrieve(triples) if triples else []
            results.append({
                "sample_id": record.get("sample_id"),
                "clinical_triples": [
                    {"triple": triple, "query_text": render_triple(triple), "retrieved_facts": facts}
                    for triple, facts in zip(triples, retrieved, strict=True)
                ],
            })
        return results

    def evaluate_records(self, records: list[dict[str, Any]], *, stop_after: str = "judge") -> dict[str, Any]:
        if stop_after not in {"clinical_triples", "fact_retrieval", "judge"}:
            raise ValueError("stop_after must be clinical_triples, fact_retrieval, or judge")
        if stop_after == "clinical_triples":
            return {"metric": "GlobalFactualExplanationEval", "stop_after": stop_after, "results": [
                {"sample_id": record.get("sample_id"), "clinical_triples": clinical_triples(record)}
                for record in records
            ]}
        if stop_after == "fact_retrieval":
            return {
                "metric": "GlobalFactualExplanationEval",
                "stop_after": stop_after,
                "top_k": self.top_k,
                "milvus_collection": self.collection,
                "results": self.retrieve_records(records),
            }
        results = []
        self._ensure_judge()
        total_input = total_output = 0
        supported_count = generated_count = 0
        total_support_score = 0
        for record in records:
            triples = clinical_triples(record)
            retrieved = self._retrieve(triples) if triples else []
            judged = []
            for triple, facts in zip(triples, retrieved, strict=True):
                import dspy
                with dspy.context(lm=self.lm):
                    prediction = self.judge(
                        generated_clinical_triple=json.dumps(triple, ensure_ascii=False),
                        retrieved_knowledge_graph_facts=json.dumps(facts, ensure_ascii=False),
                    )
                input_tokens, output_tokens = token_usage(self.lm)
                total_input += input_tokens
                total_output += output_tokens
                support_score = self._parse_support_score(prediction.support_score)
                # Keep the legacy field for consumers that still need a coarse
                # view, while making the continuous score the primary metric.
                is_supported = support_score >= 5
                supported_count += int(is_supported)
                generated_count += 1
                total_support_score += support_score
                judged.append({
                    "triple": triple,
                    "retrieved_facts": facts,
                    "supported": is_supported,
                    "support_score": support_score,
                    "reasoning": str(prediction.reasoning).strip(),
                })
            precision = sum(item["supported"] for item in judged) / len(judged) if judged else 0.0
            mean_support_score = sum(item["support_score"] for item in judged) / len(judged) if judged else 0.0
            results.append({"sample_id": record.get("sample_id"), "clinical_triples": judged, "summary": {"generated": len(judged), "supported": sum(item["supported"] for item in judged), "precision": precision, "mean_support_score": mean_support_score, "support_score_max": 10}})
        return {
            "metric": "GlobalFactualExplanationEval",
            "stop_after": stop_after,
            "top_k": self.top_k,
            "milvus_collection": self.collection,
            "samples": len(results),
            "aggregate": {"generated_clinical_triples": generated_count, "supported_clinical_triples": supported_count, "precision": supported_count / generated_count if generated_count else 0.0, "mean_support_score": total_support_score / generated_count if generated_count else 0.0, "support_score_max": 10},
            "cost": {"input_tokens": total_input, "output_tokens": total_output},
            "results": results,
        }
