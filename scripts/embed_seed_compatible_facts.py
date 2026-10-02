#!/usr/bin/env python3
"""Embed facts whose subject and object use policy-approved seed types.

The target is the existing HippoRAG fact collection (raw_triples_v1 by default).
Use --replace when the collection should contain only this filtered vocabulary.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kg_pipeline.embeddings import EmbeddingConfig, OpenAICompatibleEmbedder
from kg_pipeline.minimal_graph_export import target_dsn
from kg_pipeline.vector_store import KnowledgeGraphVectorStore, MilvusConfig

SEED_TYPES = ("symptom", "behavior", "disorder", "concept")


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true", help="delete all existing rows in the target collection first")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    source = os.environ["DSM_KG_DATABASE_URL"]
    database = os.getenv("DSM_MINIMAL_GRAPH_DATABASE_NAME", "dsm5_minimal_graph")
    config = MilvusConfig.from_env()
    store = KnowledgeGraphVectorStore(config)
    collection = config.triple_collection
    embedder = OpenAICompatibleEmbedder(EmbeddingConfig.from_env())

    with psycopg.connect(target_dsn(source, database)) as conn, conn.cursor() as cur:
        query = """
            SELECT f.id::text, es.text, f.predicate, eo.text,
                   es.type, eo.type
            FROM facts f
            JOIN entities es ON es.id = f.subject_id
            JOIN entities eo ON eo.id = f.object_id
            WHERE es.type = ANY(%s) AND eo.type = ANY(%s)
            ORDER BY f.id
        """
        params = (list(SEED_TYPES), list(SEED_TYPES))
        if args.limit:
            query += " LIMIT %s"
            params = (*params, args.limit)
        cur.execute(query, params)
        rows = list(cur)

    if args.replace:
        store.client.delete(collection_name=collection, filter='id != ""')
        print(f"cleared collection: {collection}")

    inserted = 0
    for batch in batches(rows, config.insert_batch_size):
        texts = [f"{subject} {predicate} {object_}" for _, subject, predicate, object_, _, _ in batch]
        vectors = embedder.embed(texts, dimension=config.triple_dimension)
        payload = [
            {
                "id": fact_id,
                "vector": vector.tolist(),
                "text": text,
                "subject_text": subject,
                "predicate": predicate,
                "object_text": object_,
                "node_id": "",
                "extraction_set": "minimal_seed_compatible",
                "source_path": "",
                "ordinal": index,
                "embedding_model": embedder.config.model,
            }
            for index, ((fact_id, subject, predicate, object_, _, _), text, vector)
            in enumerate(zip(batch, texts, vectors, strict=True), start=inserted)
        ]
        store.upsert(collection, payload)
        inserted += len(payload)
        print(f"embedded {inserted}/{len(rows)}")
    print({"collection": collection, "seed_types": SEED_TYPES, "facts_selected": len(rows),
           "facts_upserted": inserted, "collection_count": store.count(collection), "replaced": args.replace})


if __name__ == "__main__":
    main()
