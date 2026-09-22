from __future__ import annotations

import logging
from dataclasses import dataclass
import json
from typing import Any

from methods.dspy_runtime import build_lm, token_usage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactQueryGenerationResult:
    triples: list[dict[str, str]]
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    errors: tuple[str, ...] = ()


def render_fact_query(triple: dict[str, str]) -> str:
    """Render an LLM-produced triple like the text stored in the Milvus index."""
    return " ".join(
        value.strip()
        for value in (triple.get("subject", ""), triple.get("predicate", ""), triple.get("object", ""))
        if value and value.strip()
    )


def _clean_triples(values: Any, *, limit: int) -> list[dict[str, str]]:
    if not isinstance(values, list):
        return []
    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        triple = {key: str(value.get(key) or "").strip() for key in ("subject", "predicate", "object")}
        identity = tuple(triple.values())
        if not all(identity) or identity in seen:
            continue
        seen.add(identity)
        cleaned.append(triple)
        if len(cleaned) >= limit:
            break
    return cleaned


class FactRecognizer:
    """Turn a psychiatric self-report into fact-shaped semantic search queries."""

    def __init__(self, mode: str = "llm", *, max_queries: int = 5):
        if mode not in {"embedding", "llm"}:
            raise ValueError("recognition mode must be 'embedding' or 'llm'")
        self.mode = mode
        if max_queries <= 0:
            raise ValueError("max_queries must be positive")
        self.max_queries = max_queries
        self.lm = None
        self.program = None
        if mode == "llm":
            import dspy

            class GeneratePsychiatricFactQueries(dspy.Signature):
                """Extract meaningful knowledge-graph facts from the entire post.

                Return only concise subject-predicate-object triples. Use the main
                semantic entity as subject; use `poster` only for an action, experience,
                request, or belief explicitly attributed to the writer. Cover symptoms,
                disorders, medications, treatments, effects, questions, and relations.
                Preserve uncertainty and negation. Keep input placeholders such as
                `[deleted]` in provenance, but do not treat them as semantic entities
                or facts.
                Do not infer diagnoses or relations not stated, and do not force every
                allowed label into a triple. The appended label list is only the task
                label set; it is not evidence.
                """

                post: str = dspy.InputField(desc="Psychiatric post, self-report, or question")
                max_triples: int = dspy.InputField(desc="Maximum number of triples to return")
                triples: list[dict[str, str]] = dspy.OutputField(
                    desc="Objects with exactly subject, predicate, and object string fields"
                )

            self.lm = build_lm()
            self.program = dspy.Predict(GeneratePsychiatricFactQueries)

    def generate(self, post: str) -> FactQueryGenerationResult:
        if self.mode == "embedding":
            return FactQueryGenerationResult(triples=[{
                "subject": "patient self-report",
                "predicate": "is clinically related to",
                "object": post.strip(),
            }])

        import dspy

        try:
            with dspy.context(lm=self.lm):
                prediction = self.program(post=post, max_triples=self.max_queries)
            triples = _clean_triples(prediction.triples, limit=self.max_queries)
            input_tokens, output_tokens = token_usage(self.lm)
            return FactQueryGenerationResult(
                triples=triples,
                raw_response=repr(prediction),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as error:
            logger.exception("Fact-query generation failed")
            return FactQueryGenerationResult(
                triples=[], errors=(f"fact_generation_error: {error}",)
            )


@dataclass(frozen=True)
class FactFilteringResult:
    selected_ids: tuple[str, ...]
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    errors: tuple[str, ...] = ()


class FactRelevanceFilter:
    """Use one DSPy call to filter the merged hits from all fact queries."""

    def __init__(self):
        import dspy

        class FilterRetrievedFacts(dspy.Signature):
            """Select retrieved facts that are useful for answering the question.

            Judge meaning, not merely shared words. Keep facts that support, oppose,
            clarify, or provide a necessary bridge for any generated query and the
            user's question. Compare candidates across all queries and keep only the
            most useful facts overall. Preserve facts about the target disorder.
            Reject entity substitutions, unrelated case facts, and polarity conflicts
            such as `denies symptom` when a query asks for `reports symptom`, unless
            that contradiction itself is useful evidence. Return only candidate IDs
            from the supplied list. Returning no IDs is allowed when every hit is noise.
            """

            question: str = dspy.InputField(desc="Original post and the question to answer")
            queries_json: str = dspy.InputField(desc="JSON array of all triple-shaped retrieval queries")
            candidates_json: str = dspy.InputField(
                desc="JSON array of candidates and the query ranks that retrieved each one"
            )
            max_selected: int = dspy.InputField(desc="Maximum number of fact IDs to select")
            selected_ids: list[str] = dspy.OutputField(
                desc="At most max_selected candidate IDs, in descending usefulness"
            )

        self.lm = build_lm()
        self.program = dspy.Predict(FilterRetrievedFacts)

    def filter(
        self, question: str, queries: list[dict[str, str]], candidates: list[dict],
        *, max_selected: int,
    ) -> FactFilteringResult:
        import dspy

        allowed = {str(row["id"]) for row in candidates}
        payload = [
            {
                "candidate_id": str(row["id"]),
                "subject": row.get("subject_text", ""),
                "predicate": row.get("predicate", ""),
                "object": row.get("object_text", ""),
                "text": row.get("text", ""),
                "matched_query_ranks": sorted({
                    int(match["query_rank"]) for match in row.get("matched_queries", [])
                }),
                "cosine_similarity": row.get("raw_score", 0.0),
            }
            for row in candidates
        ]
        try:
            with dspy.context(lm=self.lm):
                prediction = self.program(
                    question=question,
                    queries_json=json.dumps(queries, ensure_ascii=False),
                    candidates_json=json.dumps(payload, ensure_ascii=False),
                    max_selected=max_selected,
                )
            selected: list[str] = []
            for candidate_id in prediction.selected_ids:
                candidate_id = str(candidate_id)
                if candidate_id in allowed and candidate_id not in selected:
                    selected.append(candidate_id)
                if len(selected) >= max_selected:
                    break
            input_tokens, output_tokens = token_usage(self.lm)
            return FactFilteringResult(
                selected_ids=tuple(selected), raw_response=repr(prediction),
                input_tokens=input_tokens, output_tokens=output_tokens,
            )
        except Exception as error:
            logger.exception("Fact relevance filtering failed")
            return FactFilteringResult(
                selected_ids=tuple(str(row["id"]) for row in candidates),
                errors=(f"fact_filtering_error: {error}",),
            )
