"""ToG-2 context-based candidate-entity pruning on DSM's graph and chunks.

This implements equations (5) and (6) of the paper. It is separate from the
initial seed-context retrieval in ``dsm_seed_context``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from dsm_seed_context import _tog1_backends, get_seed_chunks


def _triple_sentence(candidate: dict) -> str:
    """Render the traversed directed edge, not an unrelated supporting fact."""
    if candidate.get("triple_sentence"):
        return str(candidate["triple_sentence"]).strip()
    topic = str(candidate["topic_entities"]).strip()
    relation = str(candidate["relation"]).strip()
    name = str(candidate["name"]).strip()
    if candidate.get("head", True):
        return f"{topic} {relation} {name}."
    return f"{name} {relation} {topic}."


def rank_candidate_entities(
    question: str,
    candidates: Sequence[dict],
    *,
    width: int = 5,
    context_number: int = 10,
    alpha: float = 0.8,
    embedding_batch_size: int = 64,
    return_all: bool = False,
) -> tuple[list[dict], list[dict]] | tuple[list[dict], list[dict], dict]:
    """Return (top-W entities, globally top-K scored candidate/chunk pairs).

    Each candidate needs ``id``, ``name``, ``topic_entities``, ``relation`` and
    ``head`` (edge direction). The chunk pool comes from PostgreSQL's
    fact/mention/chunk links. A chunk linked to two candidate entities is
    scored separately with each candidate's own traversed triple. Equation
    (5) requires embedding the *combined* triple sentence and chunk; a stored
    bare-chunk vector cannot reproduce that score.
    """
    if width < 1 or context_number < 1 or embedding_batch_size < 1:
        raise ValueError("width, context_number and embedding_batch_size must be positive")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite non-negative number")
    if not candidates:
        return ([], [], {"all_scored_contexts": [], "entity_ranking": []}) if return_all else ([], [])

    # The same entity can be reached through different triples; keep each path.
    candidates_by_id: dict[str, list[tuple[int, dict]]] = {}
    for index, candidate in enumerate(candidates):
        candidates_by_id.setdefault(str(candidate["id"]), []).append((index, candidate))
    chunks = get_seed_chunks({entity_id: paths[0][1]["name"] for entity_id, paths in candidates_by_id.items()})
    passages: list[dict] = []
    for chunk in chunks:
        for entity_id in chunk["seed_ids"]:
            for candidate_index, candidate in candidates_by_id[entity_id]:
                triple = _triple_sentence(candidate)
                passages.append({
                    "candidate_index": candidate_index,
                    "entity_id": entity_id,
                    "entity_name": candidate["name"],
                    "chunk_id": chunk["chunk_id"],
                    "chunk": chunk["text"],
                    "triple_sentence": triple,
                    "scoring_text": f"{triple} : {chunk['text']}",
                    "facts": chunk["facts"],
                })
    if not passages:
        missing = {str(candidate["id"]): candidate["name"] for candidate in candidates}
        details = {"all_scored_contexts": [], "entity_ranking": [
            {"id": entity_id, "name": name, "entity_score": 0.0,
             "selected": False, "reason": "no_document_chunk"}
            for entity_id, name in missing.items()
        ]}
        return ([], [], details) if return_all else ([], [])

    _, ml = _tog1_backends()
    query_vector = np.asarray(ml.embed([question])[0], dtype=np.float32)
    query_norm = float(np.linalg.norm(query_vector))
    if not query_norm:
        raise ValueError("Question embedding has zero norm")
    for start in range(0, len(passages), embedding_batch_size):
        batch = passages[start:start + embedding_batch_size]
        vectors = np.asarray(ml.embed([item["scoring_text"] for item in batch]), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != len(query_vector):
            raise ValueError("Question and candidate-context embedding dimensions differ")
        norms = np.linalg.norm(vectors, axis=1)
        scores = (vectors @ query_vector) / np.maximum(norms * query_norm, 1e-12)
        for item, score in zip(batch, scores):
            item["score"] = float(score)

    # K is global across every candidate, never a separate top-K per entity.
    passages.sort(key=lambda item: (-item["score"], item["candidate_index"], item["chunk_id"]))
    top_context = passages[:context_number]
    entity_scores: dict[str, float] = {}
    best_path: dict[str, int] = {}
    for rank, item in enumerate(top_context, start=1):
        entity_id = item["entity_id"]
        best_path.setdefault(entity_id, item["candidate_index"])
        entity_scores[entity_id] = (
            entity_scores.get(entity_id, 0.0)
            + item["score"] * math.exp(-alpha * rank)
        )

    ranked_ids = sorted(entity_scores, key=lambda entity_id: (-entity_scores[entity_id], entity_id))[:width]
    selected = []
    for entity_id in ranked_ids:
        result = dict(candidates[best_path[entity_id]])
        result["entity_score"] = entity_scores[entity_id]
        result["context_chunks"] = [
            item for item in top_context if item["entity_id"] == entity_id
        ]
        selected.append(result)
    if not return_all:
        return selected, top_context
    names_by_id = {str(candidate["id"]): candidate["name"] for candidate in candidates}
    all_ranked_ids = sorted(names_by_id, key=lambda entity_id: (-entity_scores.get(entity_id, 0.0), entity_id))
    details = {
        "all_scored_contexts": passages,
        "entity_ranking": [
            {"id": entity_id, "name": names_by_id[entity_id],
             "entity_score": entity_scores.get(entity_id, 0.0),
             "selected": entity_id in set(ranked_ids),
             "reason": ("selected" if entity_id in ranked_ids else
                        "below_width" if entity_id in entity_scores else "no_chunk_in_top_k")}
            for entity_id in all_ranked_ids
        ],
    }
    return selected, top_context, details
