"""Reusable entity/relation extraction service, independent of Huey and storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class EntityProgram(Protocol):
    def __call__(self, *, text: str, context_hierarchy: str) -> Any: ...


class RelationProgram(Protocol):
    def __call__(
        self, *, text: str, entities: list[dict[str, Any]], passage_type: str,
        context_section: str,
    ) -> Any: ...


@dataclass(frozen=True)
class EntityExtraction:
    node_id: str
    entities: list[dict[str, Any]]
    passage_type: str | None
    accepted: bool


@dataclass(frozen=True)
class RelationExtraction:
    node_id: str
    triples: list[dict[str, Any]]


class KnowledgeGraphExtractor:
    """Coordinates the two extraction phases without deciding queue or storage."""

    def __init__(self, entity_program: EntityProgram, relation_program: RelationProgram):
        self.entity_program = entity_program
        self.relation_program = relation_program

    def extract_entities(self, content: str, hierarchy_path: str, node_id: str) -> EntityExtraction:
        prediction = self.entity_program(text=content, context_hierarchy=hierarchy_path)
        return EntityExtraction(
            node_id=node_id,
            entities=list(prediction.entities),
            passage_type=getattr(prediction, "passage_type", None),
            accepted=bool(prediction.get("alright", False)),
        )

    def extract_relations(
        self, content: str, hierarchy_path: str, entity_result: EntityExtraction,
    ) -> RelationExtraction:
        prediction = self.relation_program(
            text=content,
            entities=entity_result.entities,
            passage_type=entity_result.passage_type or "other",
            context_section=hierarchy_path,
        )
        return RelationExtraction(node_id=entity_result.node_id, triples=list(prediction.triples))


def build_dspy_extractor(settings) -> KnowledgeGraphExtractor:
    """Build the DSPy implementation only when a worker actually needs it."""
    import dspy
    from dspy import BootstrapFewShot

    from signatures import entity_example, entity_extraction, relation_examples, relation_extraction

    language_model = dspy.LM(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
    )
    dspy.settings.configure(lm=language_model, temperature=0.0)
    optimizer = BootstrapFewShot()
    entity_program = optimizer.compile(
        dspy.ChainOfThought(entity_extraction.DisorderEntityExtractor),
        trainset=[
            entity_example.case_study_para1_entities,
            entity_example.case_study_para2_entities,
            entity_example.example_entities,
        ],
    )
    relation_program = optimizer.compile(
        dspy.ChainOfThought(relation_extraction.DisorderRelationExtractor),
        trainset=[
            relation_examples.case_study_para1_relations,
            relation_examples.case_study_para2_relations,
            relation_examples.example_relations,
        ],
    )
    return KnowledgeGraphExtractor(entity_program, relation_program)
