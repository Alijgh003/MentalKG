"""Retrieve and rank DSM source chunks for ToG-2's initial topic entities.

PostgreSQL supplies the exact seed -> fact -> mention -> chunk relationship.
Milvus supplies the already indexed chunk vectors. The returned evidence keeps
the fact triplet and both database IDs alongside the source text.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def _tog1_backends():
    # The original checkout kept ToG-1 beside ToG-2 under separate trees.
    # The project adapter vendors both implementations under one package while
    # preserving their APIs and search logic.
    source = Path(__file__).resolve().parents[1] / "tog1"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import milvus_linker as ml
    import postgres_func as pg

    return pg, ml


def get_seed_chunks(
    seed_nodes: Mapping[str, str],
) -> list[dict]:
    """Fetch unique chunks supported by facts touching the supplied seeds.

Each result contains all incident facts that mention that chunk. The seed ID
is always an entity UUID; the chunk ID is ``chunks.id`` in PostgreSQL.
    """
    if not seed_nodes:
        return []
    pg, _ = _tog1_backends()
    connection = pg._conn()
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT DISTINCT f.id, f.subject_id, s.text, f.predicate, "
            "f.object_id, o.text, c.id, c.content "
            "FROM facts AS f "
            "JOIN mentions AS m ON m.fact_id = f.id "
            "JOIN chunks AS c ON c.id = m.chunk_id "
            "LEFT JOIN entities AS s ON s.id = f.subject_id "
            "LEFT JOIN entities AS o ON o.id = f.object_id "
            "WHERE f.subject_id = ANY(%s::uuid[]) "
            "OR f.object_id = ANY(%s::uuid[]) "
            "ORDER BY f.id, c.id",
            (list(seed_nodes), list(seed_nodes)),
        )
        rows = cursor.fetchall()
    finally:
        pg._put(connection)

    by_chunk: dict[str, dict] = {}
    seed_ids = set(seed_nodes)
    for fact_id, subject_id, subject, predicate, object_id, obj, chunk_id, content in rows:
        if not content or not str(content).strip():
            continue
        subject_id, object_id, chunk_id = str(subject_id), str(object_id), str(chunk_id)
        item = by_chunk.setdefault(
            chunk_id,
            {"chunk_id": chunk_id, "text": content, "seed_ids": set(), "facts": []},
        )
        item["seed_ids"].update({subject_id, object_id} & seed_ids)
        fact = {
            "fact_id": str(fact_id),
            "subject_id": subject_id,
            "subject": subject or subject_id,
            "predicate": predicate,
            "object_id": object_id,
            "object": obj or object_id,
        }
        if fact not in item["facts"]:
            item["facts"].append(fact)

    return [
        {**item, "seed_ids": sorted(item["seed_ids"])}
        for item in by_chunk.values()
    ]


def get_chunk_vectors(
    chunk_ids: Sequence[str],
    *,
    collection: str = "source_chunks_v1",
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Read stored vectors whose Milvus primary key equals ``chunks.id``.

    If the collection uses another primary key, the missing IDs remain absent
    and the ranking function reports the mismatch explicitly.
    """
    if not chunk_ids:
        return {}
    _, ml = _tog1_backends()
    client = ml._client_or_raise()
    client.load_collection(collection)
    vectors: dict[str, np.ndarray] = {}
    for start in range(0, len(chunk_ids), batch_size):
        batch = list(chunk_ids[start : start + batch_size])
        for hit in client.get(
            collection_name=collection, ids=batch, output_fields=["vector"]
        ):
            if hit.get("id") is not None and hit.get("vector") is not None:
                vectors[str(hit["id"])] = np.asarray(hit["vector"], dtype=np.float32)
    return vectors


def rank_seed_chunks(
    question: str,
    seed_nodes: Mapping[str, str],
    *,
    top_k_per_seed: int = 3,
    collection: str = "source_chunks_v1",
    return_all: bool = False,
) -> list[dict] | tuple[list[dict], list[dict]]:
    """Rank eligible chunks by cosine(question vector, stored chunk vector).

    The question embedding must use the same model that produced the stored
    chunk vectors. Candidate chunk vectors are read from Milvus, not
    recomputed. The fact triples are retained as evidence provenance.
    """
    if top_k_per_seed < 1:
        raise ValueError("top_k_per_seed must be positive")
    chunks = get_seed_chunks(seed_nodes)
    if not chunks:
        return ([], []) if return_all else []
    vectors = get_chunk_vectors(
        [item["chunk_id"] for item in chunks], collection=collection
    )
    if not vectors:
        raise ValueError(
            "No PostgreSQL chunk IDs matched Milvus vector IDs in "
            f"{collection!r}; verify the chunk identity mapping"
        )

    _, ml = _tog1_backends()
    question_vector = np.asarray(ml.embed([question])[0], dtype=np.float32)
    question_norm = float(np.linalg.norm(question_vector))
    if not question_norm:
        raise ValueError("Question embedding has zero norm")

    ranked = []
    for item in chunks:
        chunk_vector = vectors.get(item["chunk_id"])
        if chunk_vector is None:
            continue
        if len(chunk_vector) != len(question_vector):
            raise ValueError(
                f"Chunk {item['chunk_id']} has vector dimension {len(chunk_vector)} "
                f"but question has dimension {len(question_vector)}"
            )
        chunk_norm = float(np.linalg.norm(chunk_vector))
        if not chunk_norm:
            continue
        score = float(np.dot(question_vector, chunk_vector) / (question_norm * chunk_norm))
        ranked.append({**item, "score": score})

    ranked.sort(key=lambda item: (-item["score"], item["chunk_id"]))
    seen_per_seed: dict[str, int] = defaultdict(int)
    selected: list[dict] = []
    for item in ranked:
        useful_seeds = [
            seed_id
            for seed_id in item["seed_ids"]
            if seen_per_seed[seed_id] < top_k_per_seed
        ]
        if not useful_seeds:
            continue
        selected.append({**item, "selected_seed_ids": useful_seeds})
        for seed_id in useful_seeds:
            seen_per_seed[seed_id] += 1
    return (selected, ranked) if return_all else selected


def retrieve_initial_context(
    question: str,
    seed_nodes: Mapping[str, str],
    *,
    top_k_paragraphs: int = 3,
    top_k_sentences: int = 3,
    collection: str = "source_chunks_v1",
) -> list[dict]:
    """Mirror ToG-2's initial paragraph-then-sentence retrieval.

    For each topic entity, rank its PostgreSQL-linked chunks using their
    stored Milvus vectors, retain the top paragraphs, split them with ToG-2's
    existing sentence-window function, and rank the windows against the
    question. The result has ToG-2's ``text``/``score`` fields plus provenance.
    """
    if top_k_paragraphs < 1 or top_k_sentences < 1:
        raise ValueError("Both retrieval limits must be positive")
    ranked_chunks = rank_seed_chunks(
        question,
        seed_nodes,
        top_k_per_seed=top_k_paragraphs,
        collection=collection,
    )
    if not ranked_chunks:
        return []

    from search import split_sentences_windows

    _, ml = _tog1_backends()
    question_vector = np.asarray(ml.embed([question])[0], dtype=np.float32)
    question_norm = float(np.linalg.norm(question_vector))
    if not question_norm:
        raise ValueError("Question embedding has zero norm")

    related_sentences: list[dict] = []
    for seed_id in seed_nodes:
        top_chunks = [
            item for item in ranked_chunks if seed_id in item["selected_seed_ids"]
        ][:top_k_paragraphs]
        if not top_chunks:
            continue
        # This is the paragraph concatenation performed by
        # search.pages_embedding_search in the original ToG-2.
        paragraph = "\n".join(item["text"] for item in top_chunks)
        windows = [
            text.strip() for text in split_sentences_windows(paragraph)
            if text.strip()
        ]
        if not windows:
            continue
        sentence_vectors = np.asarray(ml.embed(windows), dtype=np.float32)
        if sentence_vectors.ndim != 2 or sentence_vectors.shape[1] != len(question_vector):
            raise ValueError("Sentence and question embedding dimensions differ")
        norms = np.linalg.norm(sentence_vectors, axis=1)
        scores = (sentence_vectors @ question_vector) / np.maximum(
            norms * question_norm, 1e-12
        )
        indices = np.argsort(-scores, kind="stable")[:top_k_sentences]
        source_ids = [item["chunk_id"] for item in top_chunks]
        source_facts = list({
            fact["fact_id"]: fact
            for item in top_chunks
            for fact in item["facts"]
        }.values())
        for index in indices:
            related_sentences.append(
                {
                    "text": windows[int(index)],
                    "score": float(scores[index]),
                    "seed_id": seed_id,
                    "source_chunk_ids": source_ids,
                    "source_facts": source_facts,
                }
            )
    return related_sentences
