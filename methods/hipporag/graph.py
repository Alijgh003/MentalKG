from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .ppr import transition_matrix


@dataclass
class GraphProjection:
    node_ids: list[str]
    node_kinds: list[str]
    node_index: dict[str, int]
    entity_types: dict[str, str | None]
    entity_chunk_counts: dict[str, int]
    chunk_indices: np.ndarray
    transition: sparse.csr_matrix
    dangling: np.ndarray


def load_graph_projection(connection) -> GraphProjection:
    """Project facts and mentions into HippoRAG's undirected entity/passage graph."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT id::text,type FROM entities ORDER BY id")
        entities = [(row[0], row[1]) for row in cursor]
        cursor.execute("SELECT id FROM chunks ORDER BY id")
        chunks = [row[0] for row in cursor]

    entity_node_ids = [f"entity:{entity_id}" for entity_id, _ in entities]
    chunk_node_ids = [f"chunk:{chunk_id}" for chunk_id in chunks]
    node_ids = entity_node_ids + chunk_node_ids
    node_kinds = ["entity"] * len(entity_node_ids) + ["chunk"] * len(chunk_node_ids)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    entity_types = {entity_id: entity_type for entity_id, entity_type in entities}

    fact_counts: dict[tuple[int, int], float] = defaultdict(float)
    passage_edges: set[tuple[int, int]] = set()
    entity_chunks: dict[str, set[str]] = defaultdict(set)
    with connection.cursor(name="hipporag_graph_edges") as cursor:
        cursor.execute(
            "SELECT f.subject_id::text,f.object_id::text,m.chunk_id "
            "FROM facts f JOIN mentions m ON m.fact_id=f.id"
        )
        for subject_id, object_id, chunk_id in cursor:
            subject_index = node_index[f"entity:{subject_id}"]
            object_index = node_index[f"entity:{object_id}"]
            chunk_index = node_index[f"chunk:{chunk_id}"]
            if subject_index != object_index:
                pair = tuple(sorted((subject_index, object_index)))
                fact_counts[pair] += 1.0
            passage_edges.add(tuple(sorted((subject_index, chunk_index))))
            passage_edges.add(tuple(sorted((object_index, chunk_index))))
            entity_chunks[subject_id].add(chunk_id)
            entity_chunks[object_id].add(chunk_id)

    # HippoRAG combines typed contributions without parallel physical edges and
    # uses the maximum contribution as the final edge weight.
    weights = dict(fact_counts)
    for edge in passage_edges:
        weights[edge] = max(weights.get(edge, 0.0), 1.0)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for (left, right), weight in weights.items():
        rows.extend((left, right))
        columns.extend((right, left))
        values.extend((weight, weight))
    adjacency = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(node_ids), len(node_ids)),
        dtype=np.float64,
    )
    transition, dangling = transition_matrix(adjacency)
    return GraphProjection(
        node_ids=node_ids,
        node_kinds=node_kinds,
        node_index=node_index,
        entity_types=entity_types,
        entity_chunk_counts={entity_id: len(chunk_ids) for entity_id, chunk_ids in entity_chunks.items()},
        chunk_indices=np.arange(len(entity_node_ids), len(node_ids), dtype=np.int64),
        transition=transition,
        dangling=dangling,
    )
