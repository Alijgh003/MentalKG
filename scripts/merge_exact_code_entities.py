#!/usr/bin/env python3
"""Merge exact-text duplicate code entities in the minimal graph database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import DatabaseSettings
from kg_pipeline.minimal_graph_export import database_name_from_env, target_dsn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Commit the merge; default is dry-run")
    args = parser.parse_args()

    dsn = target_dsn(DatabaseSettings().database_url, database_name_from_env())
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT text, array_agg(id ORDER BY id::text), count(*)
                FROM entities
                WHERE type = 'code'
                GROUP BY text
                HAVING count(*) > 1
                ORDER BY text
                """
            )
            groups = cursor.fetchall()
            duplicate_ids = sum(count - 1 for _, _, count in groups)
            subject_updates = 0
            object_updates = 0
            for _, entity_ids, _ in groups:
                representative, *duplicates = entity_ids
                cursor.execute(
                    "UPDATE facts SET subject_id=%s WHERE subject_id = ANY(%s::uuid[])",
                    (representative, duplicates),
                )
                subject_updates += cursor.rowcount
                cursor.execute(
                    "UPDATE facts SET object_id=%s WHERE object_id = ANY(%s::uuid[])",
                    (representative, duplicates),
                )
                object_updates += cursor.rowcount
                cursor.execute("DELETE FROM entities WHERE id = ANY(%s::uuid[])", (duplicates,))

        if args.apply:
            connection.commit()
        else:
            connection.rollback()

    mode = "applied" if args.apply else "dry-run"
    print(
        f"mode={mode} groups={len(groups)} duplicate_entities={duplicate_ids} "
        f"subject_refs_updated={subject_updates} object_refs_updated={object_updates}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
