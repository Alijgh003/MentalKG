from __future__ import annotations

import logging
from dataclasses import dataclass

from methods.dspy_runtime import build_lm, token_usage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerResult:
    answer: str = ""
    explanation: str = ""
    cited_passage_ids: tuple[str, ...] = ()
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    errors: tuple[str, ...] = ()


class EvidenceAnswerer:
    """Generate a compact evidence-grounded answer from the top retrieved chunks."""

    def __init__(self):
        import dspy

        class AnswerFromPsychiatricEvidence(dspy.Signature):
            """Perform evidence-grounded screening, not formal diagnosis.

            Decide from the post and retrieved passages. Write a concise clinical
            argument: state the decisive poster observations, state what the materially
            relevant passages establish, explain why those rules apply or do not apply,
            address the strongest relevant alternative or limitation, and conclude.

            Do not invent facts or transfer details from a passage to the poster.
            Association, comorbidity, or one nonspecific observation is not sufficient.
            Return exactly one of the supplied valid labels. If evidence is insufficient,
            use the supplied label representing no disorder when one exists. Mention
            only passage IDs returned in
            `cited_passage_ids`, cite only passages that affect the decision, and keep
            the reasoning between 60 and 130 words.
            """

            question: str = dspy.InputField(desc="Original post and benchmark question")
            evidence: list[dict[str, str]] = dspy.InputField(
                desc="Retrieved evidence passages with IDs"
            )
            valid_labels: list[str] = dspy.InputField(
                desc="Complete set of labels allowed for this dataset"
            )
            answer: str = dspy.OutputField(desc="Exactly one value from valid_labels")
            cited_passage_ids: list[str] = dspy.OutputField(
                desc="Only supplied passage IDs that materially affect the decision"
            )

        self.lm = build_lm()
        self.program = dspy.ChainOfThought(
            AnswerFromPsychiatricEvidence,
            rationale_field=dspy.OutputField(
                desc=(
                    "Brief, faithful reasoning: state the decisive post observation, "
                    "connect only materially useful passages, and omit irrelevant passage IDs"
                )
            ),
        )
        self.binary_demo = dspy.Example(
            valid_labels=["no", "yes"],
            question=(
                "Post: I am not letting my depression defeat me today. I showered "
                "and did my laundry. Question: Does the poster suffer from depression?"
            ),
            evidence=[
                {
                    "id": "P1",
                    "text": (
                        "Depression may occur in some neurocognitive disorders and "
                        "requires differential diagnosis in that context."
                    ),
                }
            ],
            answer="yes",
            reasoning=(
                "The poster directly self-reports having depression. The retrieved "
                "passage concerns a different clinical context and does not affect "
                "this decision."
            ),
            cited_passage_ids=[],
        ).with_inputs("question", "evidence", "valid_labels")
        self.program.predict.demos = [self.binary_demo]

    def answer(
        self, question: str, passages: list[dict], valid_labels: tuple[str, ...]
    ) -> AnswerResult:
        import dspy

        id_map = {f"P{rank}": row["passage_id"] for rank, row in enumerate(passages, start=1)}
        evidence = [
            {"id": short_id, "text": row.get("text", "")}
            for short_id, row in zip(id_map, passages, strict=True)
        ]
        try:
            normalized_labels = {label.casefold(): label for label in valid_labels}
            self.program.predict.demos = (
                [self.binary_demo]
                if set(normalized_labels) == {"yes", "no"}
                else []
            )
            with dspy.context(lm=self.lm):
                prediction = self.program(
                    question=question,
                    evidence=evidence,
                    valid_labels=list(valid_labels),
                )
            citations = tuple(
                id_map[str(value)]
                for value in prediction.cited_passage_ids
                if str(value) in id_map
            )
            input_tokens, output_tokens = token_usage(self.lm)
            raw_answer = str(prediction.answer).strip().rstrip(".")
            answer = normalized_labels.get(raw_answer.casefold(), raw_answer)
            errors = () if answer in valid_labels else (
                f"answer_label_not_allowed: {raw_answer!r}; expected one of {list(valid_labels)!r}",
            )
            return AnswerResult(
                answer=answer,
                explanation=str(prediction.reasoning).strip(),
                cited_passage_ids=citations,
                raw_response=repr(prediction),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                errors=errors,
            )
        except Exception as error:
            logger.exception("Answer generation failed")
            return AnswerResult(errors=(f"answer_generation_error: {error}",))
