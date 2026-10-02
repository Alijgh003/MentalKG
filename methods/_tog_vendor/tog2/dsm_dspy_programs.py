"""Typed DSPy calls for every LLM decision in the DSM ToG-2 adapter.

The retrieval scores themselves are computed by Jina/Milvus, not an LLM.
These Signatures cover extraction, topic prune, combined relation prune,
sufficiency/answer, query reformulation, and final best effort.
"""

from __future__ import annotations

import os

import dspy
from config.settings import Settings


def configure_dspy_lm():
    """Build ToG-2's DSPy LM from the project's global Settings/.env."""
    settings = Settings()
    key = settings.llm_api_key
    if not key:
        raise ValueError("LLM_API_KEY is missing from the project Settings/.env")
    os.environ["OPENROUTER_API_KEY"] = key
    model = settings.llm_model
    if not model.startswith(("openrouter/", "openai/", "anthropic/", "gemini/")):
        model = "openrouter/" + model
    # Leave the provider's output budget unrestricted by default. Reasoning
    # models can otherwise spend the entire budget before emitting JSON.
    token_limit = os.environ.get("LLM_MAX_TOKENS", "none").strip().lower()
    max_tokens = None if token_limit == "none" else int(token_limit)
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("LLM_MAX_TOKENS must be positive or 'none'.")
    # Do not hide provider retries from per-call token traces.
    lm = dspy.LM(model=model, api_base=settings.llm_api_base,
                 api_key=key, temperature=0.0, max_tokens=max_tokens,
                 timeout=settings.llm_timeout, cache=False, num_retries=0)
    dspy.configure(lm=lm, track_usage=True)
    return lm


class TopicEntityExtraction(dspy.Signature):
    """Extract up to k DSM graph entities from the post. Symptoms, disorders,
    behaviors, substances and life events are valid. Allowed class labels may
    also be emitted as entities if useful; emit only relevant labels."""

    post: str = dspy.InputField(desc="complete user post and classification question")
    allowed_labels: list[str] = dspy.InputField(desc="complete set of possible final classes")
    k: int = dspy.InputField(desc="maximum number of extracted entity names")
    entities: list[str] = dspy.OutputField(desc="up to k distinct entity surface forms")


class TopicPrune(dspy.Signature):
    """Choose useful starting entities for ToG-2 graph exploration. Only
    supplied entity IDs may be selected; return at most width IDs."""

    question: str = dspy.InputField()
    allowed_labels: list[str] = dspy.InputField()
    seed_nodes: str = dspy.InputField(desc="JSON mapping PostgreSQL UUID to canonical name")
    width: int = dspy.InputField()
    selected_ids: list[str] = dspy.OutputField(desc="subset of supplied UUIDs, at most width")


class CombinedRelationPrune(dspy.Signature):
    """ToG-2 combined relation pruning: consider all topic entities together,
    select useful directed relations, and score each from 0 to 1. Omit weak
    relations. Option IDs distinguish identical predicates/directions."""

    question: str = dspy.InputField()
    allowed_labels: list[str] = dspy.InputField()
    relation_options: str = dspy.InputField(desc="JSON array of option_id, topic entity, relation and direction")
    width: int = dspy.InputField(desc="maximum selected relations per topic entity")
    selected_option_ids: list[str] = dspy.OutputField(desc="option IDs copied exactly from input")
    scores: list[float] = dspy.OutputField(desc="one 0-1 relevance score per selected option, same order")
    explanation: str = dspy.OutputField(desc="why those relations help")


class SufficiencyAndAnswer(dspy.Signature):
    """ToG-2 reasoning with hybrid knowledge. Judge whether the post,
    selected triple paths, ranked entity-linked context chunks, previous
    clues and general knowledge suffice to choose exactly ONE allowed class.
    First write a general sufficiency explanation: state whether the supplied
    retrieved contexts and triple paths are enough to decide, and why. This
    explanation must be grounded in those supplied inputs. If the evidence is
    sufficient, write a second 2-4 sentence answer explanation that names the
    specific paths or context facts used and explains how they point to the
    selected class. Do not use the answer explanation when evidence is not
    sufficient, and do not treat general knowledge as a substitute for the
    supplied paths or contexts."""

    question: str = dspy.InputField(desc="original post/question, including all possible classes")
    allowed_labels: list[str] = dspy.InputField()
    previous_clue: str = dspy.InputField()
    selected_triple_paths: str = dspy.InputField(desc="full directed paths to top-width entities")
    ranked_contexts: str = dspy.InputField(desc="top-K chunks with entity and path association")
    sufficiency_explanation: str = dspy.OutputField(
        desc="general explanation of whether the supplied paths and contexts are sufficient, and why"
    )
    sufficient: bool = dspy.OutputField(desc="true only when evidence supports an allowed class")
    answer_explanation: str = dspy.OutputField(
        desc="if sufficient is true, 2-4 sentences naming the supplied paths or context facts and linking them to the chosen class; otherwise empty"
    )
    label: str = dspy.OutputField(desc="exact allowed class if sufficient; empty otherwise")
    clue: str = dspy.OutputField(desc="helpful insight or missing evidence if not sufficient; empty otherwise")


class QueryReformulation(dspy.Signature):
    """Given the question, clue and knowledge gained so far, predict missing
    evidence and write an optimized retrieval query. Do not answer."""

    question: str = dspy.InputField()
    allowed_labels: list[str] = dspy.InputField()
    clue: str = dspy.InputField()
    selected_triple_paths: str = dspy.InputField()
    ranked_contexts: str = dspy.InputField()
    needed_evidence: str = dspy.OutputField()
    query: str = dspy.OutputField(desc="retrieval query preserving the original classification target")


class BestEffortAnswer(dspy.Signature):
    """At ToG-2's maximum depth or when no more candidates exist, produce a
    best-effort allowed class only if defensible; otherwise return empty."""

    question: str = dspy.InputField()
    allowed_labels: list[str] = dspy.InputField()
    previous_clue: str = dspy.InputField()
    selected_triple_paths: str = dspy.InputField()
    ranked_contexts: str = dspy.InputField()
    label: str = dspy.OutputField(desc="exact allowed class or empty if evidence is insufficient")
    explanation: str = dspy.OutputField()


class DSMToG2Program(dspy.Module):
    def __init__(self):
        self.extractor = dspy.Predict(TopicEntityExtraction)
        self.topic_pruner = dspy.Predict(TopicPrune)
        self.relation_pruner = dspy.ChainOfThought(CombinedRelationPrune)
        self.checker = dspy.Predict(SufficiencyAndAnswer)
        self.rewriter = dspy.Predict(QueryReformulation)
        self.final_answerer = dspy.ChainOfThought(BestEffortAnswer)
