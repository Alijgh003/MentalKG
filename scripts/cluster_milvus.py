#!/usr/bin/env python3
"""Cluster pre-consolidation Milvus vectors by entity type and predicate."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from kg_pipeline.clustering import (
    ENTITY_CLUSTER_POLICIES,
    PREDICATE_CLUSTER_POLICY,
    SEED_ELIGIBLE_ENTITY_TYPES,
    choose_cluster_count,
    cluster_vectors,
)
from kg_pipeline.vector_store import KnowledgeGraphVectorStore, MilvusConfig

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create type-specific consolidation candidate clusters from Milvus"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/clustering"),
        help="Directory for JSONL assignments and summaries",
    )
    parser.add_argument(
        "--entity-type",
        action="append",
        choices=SEED_ELIGIBLE_ENTITY_TYPES,
        help="Cluster only selected seed type(s); repeat as needed",
    )
    parser.add_argument("--skip-predicates", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _read_collection(store, collection: str, output_fields: list[str], filter_: str = ""):
    iterator = store.client.query_iterator(
        collection_name=collection,
        batch_size=2000,
        filter=filter_,
        output_fields=output_fields,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            yield from batch
    finally:
        iterator.close()


def _unique_vectors(records):
    """Keep one vector per normalized text while retaining mention frequency."""
    unique = {}
    counts = Counter()
    for record in records:
        key = record["normalized_text"]
        counts[key] += 1
        unique.setdefault(key, record)
    rows = list(unique.values())
    matrix = np.asarray([row["vector"] for row in rows], dtype=np.float32)
    return rows, matrix, counts


def _write_clusters(
    path: Path,
    group_name: str,
    rows: list[dict],
    labels: np.ndarray,
    similarities: np.ndarray,
    mention_counts: Counter,
) -> dict:
    members_by_cluster = Counter(map(int, labels))
    representative_index = {}
    for index, (label, similarity) in enumerate(zip(labels, similarities, strict=True)):
        label = int(label)
        if label not in representative_index or similarity > similarities[representative_index[label]]:
            representative_index[label] = index

    with path.open("w", encoding="utf-8") as handle:
        for index, (row, label, similarity) in enumerate(
            zip(rows, labels, similarities, strict=True)
        ):
            label = int(label)
            representative = rows[representative_index[label]]["text"]
            output = {
                "group": group_name,
                "cluster_id": f"{group_name}:{label}",
                "text": row["text"],
                "normalized_text": row["normalized_text"],
                "source_record_id": str(row["id"]),
                "mention_count": mention_counts[row["normalized_text"]],
                "cluster_unique_text_count": members_by_cluster[label],
                "representative": representative,
                "cosine_to_centroid": round(float(similarity), 8),
                "is_representative": index == representative_index[label],
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    return {
        "group": group_name,
        "unique_texts": len(rows),
        "mentions": sum(mention_counts.values()),
        "clusters": len(members_by_cluster),
        "mean_unique_texts_per_cluster": round(len(rows) / len(members_by_cluster), 2),
        "mean_cosine_to_centroid": round(float(np.mean(similarities)), 6),
        "minimum_cosine_to_centroid": round(float(np.min(similarities)), 6),
        "output": str(path),
    }


def _cluster_group(store, collection, group_name, filter_, policy, output_dir, random_state):
    logger.info("Reading vectors for %s", group_name)
    records = _read_collection(
        store,
        collection,
        ["id", "vector", "text", "normalized_text"],
        filter_,
    )
    rows, matrix, counts = _unique_vectors(records)
    if not rows:
        logger.warning("No vectors found for %s", group_name)
        return None
    cluster_count = choose_cluster_count(len(rows), policy)
    logger.info(
        "Clustering %s: mentions=%s unique_texts=%s requested_clusters=%s",
        group_name,
        sum(counts.values()),
        len(rows),
        cluster_count,
    )
    labels, _, similarities = cluster_vectors(
        matrix, cluster_count, random_state=random_state
    )
    return _write_clusters(
        output_dir / f"{group_name}_clusters.jsonl",
        group_name,
        rows,
        labels,
        similarities,
        counts,
    )


def main() -> int:
    args = _parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = KnowledgeGraphVectorStore(MilvusConfig())
    entity_types = args.entity_type or list(SEED_ELIGIBLE_ENTITY_TYPES)
    summaries = []

    for entity_type in entity_types:
        summary = _cluster_group(
            store,
            store.config.entity_collection,
            entity_type,
            f'entity_type == "{entity_type}"',
            ENTITY_CLUSTER_POLICIES[entity_type],
            args.output_dir,
            args.random_state,
        )
        if summary:
            summaries.append(summary)

    if not args.skip_predicates:
        summary = _cluster_group(
            store,
            store.config.predicate_collection,
            "predicate",
            "",
            PREDICATE_CLUSTER_POLICY,
            args.output_dir,
            args.random_state,
        )
        if summary:
            summaries.append(summary)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "purpose": "consolidation_candidate_generation_not_clinical_classes",
                "seed_eligible_entity_types": list(SEED_ELIGIBLE_ENTITY_TYPES),
                "groups": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
