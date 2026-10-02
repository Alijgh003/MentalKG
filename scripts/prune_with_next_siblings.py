#!/usr/bin/env python3
"""Remove bare siblings of with_next_page chunks and synchronize Milvus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg

from config.settings import DatabaseSettings
from kg_pipeline.minimal_graph_export import database_name_from_env, target_dsn
from kg_pipeline.vector_store import MilvusConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    dst = target_dsn(DatabaseSettings().database_url, database_name_from_env())
    mc = MilvusConfig.from_env()
    from pymilvus import MilvusClient
    milvus = MilvusClient(uri=mc.uri, token=mc.token)

    with psycopg.connect(dst) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH wn AS (
                    SELECT id, replace(id, 'with_next_page_', '') AS sibling_id
                    FROM chunks WHERE id LIKE 'with_next_page_%'
                )
                SELECT c.id FROM chunks c JOIN wn ON wn.sibling_id=c.id
            """)
            sibling_ids = sorted(row[0] for row in cur.fetchall())
            cur.execute("""
                SELECT DISTINCT m.fact_id::text
                FROM mentions m JOIN unnest(%s::text[]) x(id) ON x.id=m.chunk_id
            """, (sibling_ids,))
            fact_ids = sorted(row[0] for row in cur.fetchall())
            cur.execute("""
                SELECT DISTINCT e.id::text
                FROM entities e
                WHERE e.id IN (
                    SELECT subject_id FROM facts WHERE id = ANY(%s::uuid[])
                    UNION
                    SELECT object_id FROM facts WHERE id = ANY(%s::uuid[])
                )
            """, (fact_ids, fact_ids))
            touched_entities = {row[0] for row in cur.fetchall()}
            summary = {
                "database": database_name_from_env(),
                "sibling_chunks": len(sibling_ids),
                "facts_attached_to_siblings": len(fact_ids),
                "endpoint_entities_of_those_facts": len(touched_entities),
                "execute": args.execute,
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            if not args.execute:
                return 0

            # Delete facts explicitly; FK cascade removes all their mentions.
            cur.execute("DELETE FROM facts WHERE id = ANY(%s::uuid[])", (fact_ids,))
            deleted_facts = cur.rowcount
            cur.execute("DELETE FROM chunks WHERE id = ANY(%s::text[])", (sibling_ids,))
            deleted_chunks = cur.rowcount
            cur.execute("""
                DELETE FROM entities e
                WHERE NOT EXISTS (SELECT 1 FROM facts f WHERE f.subject_id=e.id OR f.object_id=e.id)
            """)
            deleted_entities = cur.rowcount
        conn.commit()

    # Source relation mention IDs whose node is a deleted sibling identify raw-triple vectors.
    with psycopg.connect(DatabaseSettings().database_url) as src:
        with src.cursor() as cur:
            cur.execute("SELECT relation_mention_id::text FROM relation_mentions WHERE node_id = ANY(%s::text[])", (sibling_ids,))
            raw_ids = sorted(row[0] for row in cur.fetchall())

    def delete_ids(collection: str, ids: list[str]) -> int:
        if not ids or not milvus.has_collection(collection_name=collection):
            return 0
        milvus.delete(collection_name=collection, ids=ids)
        milvus.flush(collection_name=collection)
        return len(ids)

    deleted_source_vectors = delete_ids(mc.source_chunk_collection, sibling_ids)
    deleted_raw_vectors = delete_ids(mc.triple_collection, raw_ids)
    deleted_entity_vectors = delete_ids(mc.canonical_entity_collection, sorted(touched_entities))

    with psycopg.connect(dst) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE id = ANY(%s::text[])", (sibling_ids,))
            remaining_chunks = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM entities e WHERE NOT EXISTS (SELECT 1 FROM facts f WHERE f.subject_id=e.id OR f.object_id=e.id)")
            orphan_entities = cur.fetchone()[0]
        conn.commit()
    result = {
        "postgres_deleted_facts": deleted_facts,
        "postgres_deleted_chunks": deleted_chunks,
        "postgres_deleted_orphan_entities": deleted_entities,
        "milvus_deleted_source_chunks": deleted_source_vectors,
        "milvus_deleted_raw_triples": deleted_raw_vectors,
        "milvus_deleted_canonical_entities": deleted_entity_vectors,
        "remaining_deleted_chunk_ids": remaining_chunks,
        "orphan_entities_after": orphan_entities,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if remaining_chunks == 0 and orphan_entities == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
