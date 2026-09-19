"""DSPy-style pseudocode for the minimal linguistic bridge.

This file documents the prompt contract. It is intentionally not wired to a
specific model or production pipeline yet.
"""

import dspy
from pydantic import BaseModel, Field


class Observation(BaseModel):
    evidence_span: str = Field(
        description="An exact, verbatim span copied from the self-report."
    )
    observation: str = Field(
        description=(
            "A broad, everyday-language semantic topic inferred from the span "
            "and suitable as a retrieval query."
        )
    )


class ExtractMinimalObservations(dspy.Signature):
    """Extract minimal retrieval-oriented observations from a self-report.

    Rules:
    - Ground every output in information present in `self_report`; do not use
      external facts about the writer.
    - Every observation must quote one exact supporting `evidence_span`.
    - Convert the span by ONE semantic abstraction step into a broad topic useful
      for retrieval. Everyday inferences are allowed, for example prolonged
      crying -> intense sadness or emotional distress, and drinking until a
      blackout -> heavy alcohol use.
    - Resolve obvious figurative language from context when a typical reader
      would agree, for example "dying of boredom" -> extreme boredom.
    - Preserve the experiencer and any negation, uncertainty, modality,
      condition, or time that materially changes the meaning.
    - Do not diagnose, name a DSM criterion, infer a dataset label, add missing
      duration/severity, or invent a cause that the text does not support.
    - Prefer common language such as "loss of interest" over specialist labels
      such as "anhedonia" unless the specialist term appears in the input.
    - Split a span only when it supports genuinely distinct retrieval topics;
      otherwise avoid redundant observations.
    - Return [] when the text contains no substantive self-report.
    - A non-specialist must be able to verify the observation from the span.
    """

    self_report: str = dspy.InputField(
        description="The raw self-report only; no task question, labels, or evidence."
    )
    observations: list[Observation] = dspy.OutputField()


def _demo(self_report: str, observations: list[Observation]) -> dspy.Example:
    """Build a demo only when every evidence span is an exact input substring."""

    assert all(item.evidence_span in self_report for item in observations)
    return dspy.Example(
        self_report=self_report,
        observations=observations,
    ).with_inputs("self_report")


# These examples are synthetic and must not be replaced with evaluation/test
# examples. Together they teach one-step everyday semantic abstraction, not
# domain labels: behavior/emotion topics, time, negation, attribution,
# uncertainty, condition, figurative wording, and a non-self-report question.
FEW_SHOT_DEMOS = [
    _demo(
        "Last night I kept drinking until I blacked out, and I cried for hours.",
        [
            Observation(
                evidence_span="kept drinking until I blacked out",
                observation="Heavy alcohol use to the point of blacking out.",
            ),
            Observation(
                evidence_span="I cried for hours",
                observation=(
                    "Prolonged crying associated with intense sadness or "
                    "emotional distress."
                ),
            ),
        ],
    ),
    _demo(
        "I no longer enjoy painting or meeting my friends.",
        [
            Observation(
                evidence_span="I no longer enjoy painting or meeting my friends",
                observation=(
                    "Loss of interest or enjoyment in hobbies and social activity."
                ),
            )
        ],
    ),
    _demo(
        "For the past week, I have been waking up several times every night.",
        [
            Observation(
                evidence_span=(
                    "For the past week, I have been waking up several times "
                    "every night"
                ),
                observation=(
                    "Repeated nighttime awakenings over the past week."
                ),
            )
        ],
    ),
    _demo(
        "I am worried about tomorrow, but I am not afraid to leave the house.",
        [
            Observation(
                evidence_span="I am worried about tomorrow",
                observation="Worry about an upcoming event.",
            ),
            Observation(
                evidence_span="I am not afraid to leave the house",
                observation="No fear of leaving the house.",
            ),
        ],
    ),
    _demo(
        "My roommate says I seem withdrawn; I think I am just tired.",
        [
            Observation(
                evidence_span="My roommate says I seem withdrawn",
                observation=(
                    "Perceived social withdrawal according to the writer's "
                    "roommate."
                ),
            ),
            Observation(
                evidence_span="I think I am just tired",
                observation="Possible tiredness according to the writer.",
            ),
        ],
    ),
    _demo(
        "If my workload increases again, I might quit my job.",
        [
            Observation(
                evidence_span="If my workload increases again, I might quit my job",
                observation=(
                    "Possible job resignation if workload increases again."
                ),
            )
        ],
    ),
    _demo(
        "After the argument, I cried and did not answer my friends' messages.",
        [
            Observation(
                evidence_span=(
                    "After the argument, I cried and did not answer my friends' "
                    "messages"
                ),
                observation=(
                    "Crying and withdrawal from communication with friends after "
                    "an argument."
                ),
            )
        ],
    ),
    _demo(
        "This lecture is so boring I'm dying.",
        [
            Observation(
                evidence_span="This lecture is so boring I'm dying",
                observation="Extreme boredom with the lecture.",
            )
        ],
    ),
    _demo(
        (
            "Sometimes I wish I would not wake up, but I have no plan to hurt "
            "myself."
        ),
        [
            Observation(
                evidence_span="Sometimes I wish I would not wake up",
                observation="Occasional wish not to wake up.",
            ),
            Observation(
                evidence_span="I have no plan to hurt myself",
                observation="Stated absence of a plan to hurt oneself.",
            ),
        ],
    ),
    _demo(
        (
            "My father was diagnosed with depression. "
            "I have never been diagnosed with depression."
        ),
        [
            Observation(
                evidence_span="My father was diagnosed with depression",
                observation="A depression diagnosis reported for the writer's father.",
            ),
            Observation(
                evidence_span="I have never been diagnosed with depression",
                observation="No depression diagnosis reported for the writer.",
            ),
        ],
    ),
    _demo(
        "Can lack of sleep affect concentration?",
        [],
    ),
]


# Pseudocode usage. DSPy optimizers may later select a smaller subset of these
# demos, but the held-out evaluation data must never become a demo source.
# extractor = dspy.Predict(ExtractMinimalObservations)
# extractor.demos = FEW_SHOT_DEMOS
# result = extractor(self_report=raw_text)
# assert all(item.evidence_span in raw_text for item in result.observations)
# retrieval_queries = [item.observation for item in result.observations]
