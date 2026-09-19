#!/usr/bin/env python3
"""Embed a few queries and inspect nearest KG mentions in Milvus."""

from __future__ import annotations

import argparse
import logging

from kg_pipeline.embeddings import EmbeddingConfig, OpenAICompatibleEmbedder
from kg_pipeline.vector_store import KnowledgeGraphVectorStore, MilvusConfig


DEFAULT_ENTITY_QUERIES = (
    "persistent sadness and low mood",
    "loss of interest or pleasure in activities",
    "difficulty sleeping through the night",
    "thoughts of suicide or self-harm",
)

DEFAULT_PREDICATE_QUERIES = (
    "has symptom",
    "is associated with",
    "is diagnosed as",
    "lasts for a duration",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run small semantic-search smoke tests against KG Milvus collections"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Results shown per query")
    parser.add_argument(
        "--entity-query",
        action="append",
        dest="entity_queries",
        help="Custom entity query; repeat up to four times",
    )
    parser.add_argument(
        "--predicate-query",
        action="append",
        dest="predicate_queries",
        help="Custom predicate query; repeat up to four times",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _print_results(title: str, queries: list[str], results: list[list[dict]]) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")
    for query, hits in zip(queries, results, strict=True):
        print(f"\nQUERY: {query}")
        if not hits:
            print("  No results")
            continue
        for rank, hit in enumerate(hits, start=1):
            entity = hit.get("entity", {})
            score = float(hit.get("distance", 0.0))
            metadata = []
            if entity.get("entity_type"):
                metadata.append(f"type={entity['entity_type']}")
            if entity.get("occurrence_count") is not None:
                metadata.append(f"occurrences={entity['occurrence_count']}")
            if entity.get("node_id"):
                metadata.append(f"node={entity['node_id']}")
            suffix = f"  [{' | '.join(metadata)}]" if metadata else ""
            print(
                f"  {rank:>2}. cosine_similarity={score:.6f}  "
                f"text={entity.get('text', '<missing>')!r}{suffix}"
            )


def main() -> int:
    args = _parser().parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.entity_queries and len(args.entity_queries) > 4:
        raise SystemExit("Pass at most four --entity-query values")
    if args.predicate_queries and len(args.predicate_queries) > 4:
        raise SystemExit("Pass at most four --predicate-query values")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    entity_queries = list(args.entity_queries or DEFAULT_ENTITY_QUERIES)
    predicate_queries = list(args.predicate_queries or DEFAULT_PREDICATE_QUERIES)
    embedder = OpenAICompatibleEmbedder(EmbeddingConfig())
    store = KnowledgeGraphVectorStore(MilvusConfig())

    entity_collection = store.config.entity_collection
    predicate_collection = store.config.predicate_collection
    for collection in (entity_collection, predicate_collection):
        if not store.client.has_collection(collection_name=collection):
            raise SystemExit(f"Milvus collection does not exist: {collection}")
        store.client.load_collection(collection_name=collection)

    entity_count = store.count(entity_collection)
    predicate_count = store.count(predicate_collection)
    print("Milvus smoke test")
    print(f"  entity collection:    {entity_collection} ({entity_count} rows)")
    print(f"  predicate collection: {predicate_collection} ({predicate_count} rows)")

    entity_vectors = embedder.embed(
        entity_queries, dimension=store.config.entity_dimension
    )
    predicate_vectors = embedder.embed(
        predicate_queries, dimension=store.config.predicate_dimension
    )

    entity_results = store.search(
        entity_collection,
        entity_vectors.tolist(),
        limit=args.top_k,
        output_fields=("text", "entity_type", "node_id", "source_path", "ordinal"),
    )
    predicate_results = store.search(
        predicate_collection,
        predicate_vectors.tolist(),
        limit=args.top_k,
        output_fields=("text", "occurrence_count"),
    )

    _print_results("ENTITY SEARCH", entity_queries, entity_results)
    _print_results("PREDICATE SEARCH", predicate_queries, predicate_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
