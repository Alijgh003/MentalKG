#!/usr/bin/env python3
"""Populate unconsolidated entity and predicate collections in Milvus."""

from __future__ import annotations

import argparse
import logging

import psycopg

from config.settings import DatabaseSettings
from kg_pipeline.embed_ingestion import EmbeddingIngestion
from kg_pipeline.embeddings import EmbeddingConfig, OpenAICompatibleEmbedder
from kg_pipeline.vector_store import KnowledgeGraphVectorStore, MilvusConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("entities", "predicates", "all"), default="all")
    parser.add_argument("--limit", type=int, help="Limit entity mentions for a smoke test")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for embedding and vector ingestion",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    database_config = DatabaseSettings()
    embedding_config = EmbeddingConfig()
    logging.getLogger(__name__).info(
        "Embedding configuration: endpoint=%s model=%s batch_size=%s timeout=%ss "
        "max_attempts=%s",
        embedding_config.base_url.rstrip("/") + "/embeddings",
        embedding_config.model,
        embedding_config.batch_size,
        embedding_config.timeout_seconds,
        embedding_config.max_retries + 1,
    )
    embedder = OpenAICompatibleEmbedder(embedding_config)
    store = KnowledgeGraphVectorStore(MilvusConfig())
    store.ensure_collections()

    with psycopg.connect(database_config.database_url) as connection:
        ingestion = EmbeddingIngestion(connection, embedder, store)
        if args.kind in ("entities", "all"):
            inserted = ingestion.ingest_entities(args.limit)
            print(f"entity vectors inserted: {inserted}")
        if args.kind in ("predicates", "all"):
            inserted = ingestion.ingest_predicates()
            print(f"predicate vectors inserted: {inserted}")

    print(f"entity collection count: {store.count(store.config.entity_collection)}")
    print(f"predicate collection count: {store.count(store.config.predicate_collection)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
