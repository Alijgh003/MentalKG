#!/usr/bin/env python3
"""Remove chunks without fact mentions from the minimal DB and Milvus."""

from __future__ import annotations

import argparse
import json

import psycopg

from config.settings import DatabaseSettings
from kg_pipeline.minimal_graph_export import database_name_from_env, target_dsn
from kg_pipeline.vector_store import MilvusConfig


def milvus_chunk_ids(client, collection_name: str) -> set[str]:
    iterator = client.query_iterator(
        collection_name=collection_name,
        filter="",
        output_fields=["id"],
        batch_size=1_000,
    )
    ids: set[str] = set()
    try:
        while rows := iterator.next():
            ids.update(str(row["id"]) for row in rows)
    finally:
        iterator.close()
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Apply deletions; default is dry-run")
    args = parser.parse_args()

    source_dsn = DatabaseSettings().database_url
    database_name = database_name_from_env()
    destination_dsn = target_dsn(source_dsn, database_name)
    milvus = MilvusConfig.from_env()

    from pymilvus import MilvusClient

    client = MilvusClient(uri=milvus.uri, token=milvus.token)
    collection = milvus.source_chunk_collection

    with psycopg.connect(destination_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT chunk_id FROM mentions")
            mentioned_ids = {row[0] for row in cursor}
            cursor.execute("SELECT id FROM chunks")
            postgres_ids = {row[0] for row in cursor}

        milvus_ids = milvus_chunk_ids(client, collection)
        postgres_delete_ids = postgres_ids - mentioned_ids
        milvus_delete_ids = milvus_ids - mentioned_ids

        summary = {
            "database": database_name,
            "milvus_collection": collection,
            "mentioned_chunk_ids": len(mentioned_ids),
            "postgres_chunks_before": len(postgres_ids),
            "postgres_chunks_to_delete": len(postgres_delete_ids),
            "milvus_chunks_before": len(milvus_ids),
            "milvus_chunks_to_delete": len(milvus_delete_ids),
            "execute": args.execute,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        if not args.execute:
            connection.rollback()
            return 0

        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM chunks WHERE NOT EXISTS "
                "(SELECT 1 FROM mentions WHERE mentions.chunk_id=chunks.id)"
            )
            postgres_deleted = cursor.rowcount

        delete_list = sorted(milvus_delete_ids)
        for offset in range(0, len(delete_list), 1_000):
            client.delete(
                collection_name=collection,
                ids=delete_list[offset : offset + 1_000],
            )
        if delete_list:
            client.flush(collection_name=collection)

        connection.commit()

    with psycopg.connect(destination_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM chunks")
            postgres_after = cursor.fetchone()[0]
            cursor.execute(
                "SELECT count(*) FROM chunks c WHERE NOT EXISTS "
                "(SELECT 1 FROM mentions m WHERE m.chunk_id=c.id)"
            )
            postgres_orphans_after = cursor.fetchone()[0]

    milvus_after_ids = milvus_chunk_ids(client, collection)
    remaining_unmentioned_milvus = milvus_after_ids - mentioned_ids
    result = {
        "postgres_deleted": postgres_deleted,
        "milvus_deleted": len(delete_list),
        "postgres_chunks_after": postgres_after,
        "milvus_chunks_after": len(milvus_after_ids),
        "postgres_unmentioned_after": postgres_orphans_after,
        "milvus_unmentioned_after": len(remaining_unmentioned_milvus),
        "postgres_milvus_id_sets_equal": (
            postgres_after == len(milvus_after_ids) and not (milvus_after_ids ^ mentioned_ids)
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if postgres_orphans_after or remaining_unmentioned_milvus:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
