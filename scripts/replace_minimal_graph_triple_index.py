#!/usr/bin/env python3
"""Replace the same-name Milvus triple collection from the minimal graph."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import DatabaseSettings
from kg_pipeline.embeddings import EmbeddingConfig, OpenAICompatibleEmbedder
from kg_pipeline.minimal_graph_export import database_name_from_env, target_dsn
from kg_pipeline.vector_store import KnowledgeGraphVectorStore, MilvusConfig

LOGGER = logging.getLogger(__name__)


def batches(rows, size: int):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def clean(value) -> str:
    return str(value or "").strip()


def clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit]


def load_facts(connection):
    with connection.cursor(name="minimal_graph_triple_source") as cursor:
        cursor.execute(
            """
            SELECT f.id::text, se.text, f.predicate, oe.text
            FROM facts AS f
            JOIN entities AS se ON se.id = f.subject_id
            JOIN entities AS oe ON oe.id = f.object_id
            ORDER BY f.id
            """
        )
        yield from cursor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace raw_triples_v1 with canonical facts from dsm5_minimal_graph"
    )
    parser.add_argument(
        "--database-name",
        default=database_name_from_env(),
        help="Minimal graph database name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source facts without changing Milvus",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    database_config = DatabaseSettings()
    embedding_config = EmbeddingConfig()
    milvus_config = MilvusConfig()
    collection = milvus_config.triple_collection
    source_dsn = target_dsn(database_config.database_url, args.database_name)

    with psycopg.connect(source_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM facts")
            fact_count = cursor.fetchone()[0]
            cursor.execute(
                "SELECT count(*) FROM facts f "
                "JOIN entities se ON se.id=f.subject_id "
                "JOIN entities oe ON oe.id=f.object_id"
            )
            embeddable_count = cursor.fetchone()[0]
        if fact_count != embeddable_count:
            raise RuntimeError(
                f"Only {embeddable_count}/{fact_count} facts have resolvable entity text"
            )

        if args.dry_run:
            print(
                f"mode=dry-run database={args.database_name} "
                f"collection={collection} facts={fact_count}"
            )
            return 0

        store = KnowledgeGraphVectorStore(milvus_config)
        if store.client.has_collection(collection_name=collection):
            LOGGER.info("Dropping existing Milvus collection %s", collection)
            store.client.drop_collection(collection_name=collection)
        store.ensure_final_collections()

        embedder = OpenAICompatibleEmbedder(embedding_config)
        inserted = 0
        for batch in batches(load_facts(connection), milvus_config.insert_batch_size):
            texts = [
                f"{clean(row[1])} {clean(row[2])} {clean(row[3])}"
                for row in batch
            ]
            vectors = embedder.embed(texts, dimension=milvus_config.triple_dimension)
            payload = [
                {
                    "id": row[0],
                    "vector": vector.tolist(),
                    "text": text,
                    "subject_text": clip(clean(row[1]), 8192),
                    "predicate": clip(clean(row[2]), 2048),
                    "object_text": clip(clean(row[3]), 8192),
                    "node_id": "",
                    "extraction_set": "minimal_graph",
                    "source_path": "",
                    "ordinal": 0,
                    "embedding_model": embedder.config.model,
                }
                for row, text, vector in zip(batch, texts, vectors, strict=True)
            ]
            store.upsert(collection, payload)
            inserted += len(payload)
            LOGGER.info("Inserted %s/%s canonical fact vectors", inserted, fact_count)

        store.client.flush(collection_name=collection)
        final_count = store.count(collection)

    if final_count != fact_count:
        raise RuntimeError(
            f"Milvus replacement incomplete: expected {fact_count}, found {final_count}"
        )
    print(
        f"mode=applied database={args.database_name} collection={collection} "
        f"facts={fact_count} inserted={inserted} collection_count={final_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
