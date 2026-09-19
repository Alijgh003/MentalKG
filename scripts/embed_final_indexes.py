#!/usr/bin/env python3
"""Populate final canonical/triple/chunk retrieval collections in Milvus."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import DatabaseSettings
from kg_pipeline.embeddings import EmbeddingConfig, OpenAICompatibleEmbedder
from kg_pipeline.final_index_ingestion import FinalIndexIngestion
from kg_pipeline.vector_store import KnowledgeGraphVectorStore, MilvusConfig


KINDS = ("entities", "predicates", "triples", "chunks")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build final Milvus indexes from consolidated PostgreSQL KG tables"
    )
    parser.add_argument(
        "--kind",
        choices=(*KINDS, "all"),
        default="all",
        help="Which final collection to populate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit rows per selected kind for a smoke test",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    database_config = DatabaseSettings()
    embedding_config = EmbeddingConfig()
    milvus_config = MilvusConfig()
    logging.getLogger(__name__).info(
        "Embedding configuration: endpoint=%s model=%s batch_size=%s timeout=%ss "
        "max_attempts=%s",
        embedding_config.base_url.rstrip("/") + "/embeddings",
        embedding_config.model,
        embedding_config.batch_size,
        embedding_config.timeout_seconds,
        embedding_config.max_retries + 1,
    )
    logging.getLogger(__name__).info(
        "Final Milvus collections: entities=%s/%sd predicates=%s/%sd triples=%s/%sd chunks=%s/%sd",
        milvus_config.canonical_entity_collection,
        milvus_config.canonical_entity_dimension,
        milvus_config.canonical_predicate_collection,
        milvus_config.canonical_predicate_dimension,
        milvus_config.triple_collection,
        milvus_config.triple_dimension,
        milvus_config.source_chunk_collection,
        milvus_config.source_chunk_dimension,
    )

    embedder = OpenAICompatibleEmbedder(embedding_config)
    store = KnowledgeGraphVectorStore(milvus_config)
    store.ensure_final_collections()

    selected = KINDS if args.kind == "all" else (args.kind,)
    with psycopg.connect(database_config.database_url) as connection:
        ingestion = FinalIndexIngestion(connection, embedder, store)
        if "entities" in selected:
            inserted = ingestion.ingest_canonical_entities(args.limit)
            print(f"canonical entity vectors inserted: {inserted}")
        if "predicates" in selected:
            inserted = ingestion.ingest_canonical_predicates(args.limit)
            print(f"canonical predicate vectors inserted: {inserted}")
        if "triples" in selected:
            inserted = ingestion.ingest_raw_triples(args.limit)
            print(f"raw triple vectors inserted: {inserted}")
        if "chunks" in selected:
            inserted = ingestion.ingest_source_chunks(args.limit)
            print(f"source chunk vectors inserted: {inserted}")

    print(f"canonical entity collection count: {store.count(store.config.canonical_entity_collection)}")
    print(f"canonical predicate collection count: {store.count(store.config.canonical_predicate_collection)}")
    print(f"raw triple collection count: {store.count(store.config.triple_collection)}")
    print(f"source chunk collection count: {store.count(store.config.source_chunk_collection)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
