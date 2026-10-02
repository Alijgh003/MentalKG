from __future__ import annotations

import logging
from dataclasses import dataclass

from methods.base import BenchmarkSample, MethodResult
from methods.dspy_runtime import build_lm, token_usage
from methods.telemetry import StageRecorder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoTAnswerResult:
    answer: str = ""
    explanation: str = ""
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    errors: tuple[str, ...] = ()


class CoTMethod:
    """LLM-only Chain-of-Thought without any retrieval.

    Gives the post, question, and allowed labels directly to a ChainOfThought LLM.
    No evidence, no graph, no dense retrieval. Uses same LM as other methods
    (LLM_MODEL / RERUN_LLM_MODEL via build_lm).
    """

    name = "cot"

    def __init__(self, *, generate_answer: bool = True, config_overrides: dict | None = None, stop_after: str | None = None):
        # generate_answer kept for interface parity; CoT always generates
        self.generate_answer = generate_answer
        self.stop_after = stop_after
        # config_overrides unused but accepted for runner
        import dspy

        class AnswerWithCoT(dspy.Signature):
            """Perform evidence-free screening, not formal diagnosis.

            Decide from the post and allowed labels only. Think step by step:
            state the decisive poster observations, explain why the chosen label
            applies and why alternatives do not, and conclude.
            Return exactly one of the supplied valid labels (never invent a label).
            Keep reasoning between 60 and 130 words.
            """

            question: str = dspy.InputField(desc="Original post and benchmark question with allowed labels")
            valid_labels: list[str] = dspy.InputField(desc="Complete set of labels allowed for this dataset")
            answer: str = dspy.OutputField(desc="Exactly one value from valid_labels")
            # reasoning will be injected by ChainOfThought

        self.lm = build_lm()
        self.program = dspy.ChainOfThought(
            AnswerWithCoT,
            rationale_field=dspy.OutputField(
                desc="Brief reasoning: state decisive observations and why label applies"
            ),
        )

    def close(self) -> None:
        return None

    def run(self, sample: BenchmarkSample) -> MethodResult:
        recorder = StageRecorder(
            method=self.name,
            dataset=sample.dataset,
            split=sample.split,
            dataset_row_index=sample.dataset_row_index,
            source_row_index=sample.source_row_index,
            sample_id=sample.sample_id,
        )
        labeled_question = (
            f"{sample.text}\n\nAllowed answer labels (choose exactly one): "
            f"{', '.join(sample.valid_labels)}"
        )
        stage_outputs = {}
        # Keep trace similar to other methods
        with recorder.stage("cot_generation", valid_labels=len(sample.valid_labels)) as event:
            event["llm_calls"] = 1
            import dspy

            try:
                with dspy.context(lm=self.lm):
                    prediction = self.program(
                        question=labeled_question,
                        valid_labels=list(sample.valid_labels),
                    )
                input_tokens, output_tokens = token_usage(self.lm)
                event["input_tokens"] = input_tokens
                event["output_tokens"] = output_tokens
                raw_answer = str(prediction.answer).strip().rstrip(".")
                normalized_labels = {label.casefold(): label for label in sample.valid_labels}
                answer = normalized_labels.get(raw_answer.casefold(), raw_answer)
                errors = () if answer in sample.valid_labels else (
                    f"answer_label_not_allowed: {raw_answer!r}; expected one of {list(sample.valid_labels)!r}",
                )
                reasoning = str(getattr(prediction, "reasoning", "")).strip()
                event["details"].update({
                    "answer": answer,
                    "reasoning": reasoning,
                    "valid_labels": list(sample.valid_labels),
                    "raw_answer": raw_answer,
                })
                result = MethodResult(
                    labels=[answer] if answer and not errors else [],
                    label_scores={},
                    explanation=reasoning,
                    raw_response=repr(prediction),
                    parse_status="ok" if answer and not errors else ("invalid_label" if answer else "empty"),
                    stopping_reason="cot_complete" if not errors else "cot_error",
                    llm_calls=1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    errors=list(errors),
                    stage_outputs={
                        "cot_generation": {
                            "answer": answer,
                            "reasoning": reasoning,
                            "raw_answer": raw_answer,
                            "valid_labels": list(sample.valid_labels),
                        }
                    },
                    stages=recorder.events,
                    method_metadata={"cot": {"model": self.lm.model if hasattr(self.lm, "model") else str(self.lm)}},
                )
                # Also expose answer_generation stage for viewer compatibility
                result.stage_outputs["answer_generation"] = {
                    "answer": answer,
                    "explanation": reasoning,
                    "cited_passage_ids": [],
                    "evidence_passage_ids": [],
                }
                return result
            except Exception as error:
                logger.exception("CoT generation failed")
                event["details"]["error"] = str(error)
                return MethodResult(
                    labels=[],
                    explanation="",
                    raw_response="",
                    parse_status="empty",
                    stopping_reason="cot_error",
                    llm_calls=1,
                    errors=[f"cot_error: {error}"],
                    stage_outputs={"cot_generation": {"error": str(error)}},
                    stages=recorder.events,
                )
