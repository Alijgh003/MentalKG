#!/usr/bin/env python3
"""Replace minimal-graph fact predicates with their canonical names."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import DatabaseSettings
from kg_pipeline.minimal_graph_export import database_name_from_env, target_dsn


def load_canonical_predicates(source_dsn: str) -> list[tuple[str, str]]:
    """Load the relation-mention ID to canonical predicate-name mapping."""
    with psycopg.connect(source_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT r.relation_mention_id::text, cp.canonical_name
                FROM relation_mentions AS r
                JOIN canonical_predicates AS cp
                  ON cp.canonical_predicate_id = r.canonical_predicate_id
                ORDER BY r.relation_mention_id
                """
            )
            return [(str(fact_id), canonical_name) for fact_id, canonical_name in cursor]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace facts.predicate with canonical predicate names"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit the replacements; default is a dry-run",
    )
    parser.add_argument(
        "--database-name",
        default=database_name_from_env(),
        help="Target minimal graph database name",
    )
    args = parser.parse_args()

    source_dsn = DatabaseSettings().database_url
    target_database_dsn = target_dsn(source_dsn, args.database_name)
    mapping = load_canonical_predicates(source_dsn)

    with psycopg.connect(target_database_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM facts")
            fact_count = cursor.fetchone()[0]
            if fact_count != len(mapping):
                raise RuntimeError(
                    "Refusing to update: source canonical mapping has "
                    f"{len(mapping)} rows but target facts has {fact_count} rows"
                )

            cursor.execute(
                """
                CREATE TEMP TABLE canonical_predicate_map (
                    fact_id UUID PRIMARY KEY,
                    predicate TEXT NOT NULL
                ) ON COMMIT DROP
                """
            )
            with cursor.copy(
                "COPY canonical_predicate_map(fact_id, predicate) FROM STDIN"
            ) as copy:
                for fact_id, predicate in mapping:
                    copy.write_row((fact_id, predicate))

            cursor.execute(
                """
                SELECT count(*)
                FROM facts AS f
                LEFT JOIN canonical_predicate_map AS m ON m.fact_id = f.id
                WHERE m.fact_id IS NULL
                """
            )
            missing_target_ids = cursor.fetchone()[0]
            if missing_target_ids:
                raise RuntimeError(
                    f"Refusing to update: {missing_target_ids} target facts have no "
                    "canonical predicate mapping"
                )

            cursor.execute(
                """
                UPDATE facts AS f
                   SET predicate = m.predicate
                  FROM canonical_predicate_map AS m
                 WHERE f.id = m.fact_id
                   AND f.predicate IS DISTINCT FROM m.predicate
                """
            )
            changed_rows = cursor.rowcount

            cursor.execute("SELECT count(DISTINCT predicate) FROM facts")
            distinct_predicates = cursor.fetchone()[0]

        if args.apply:
            connection.commit()
        else:
            connection.rollback()

    mode = "applied" if args.apply else "dry-run"
    print(
        f"mode={mode} facts={fact_count} mappings={len(mapping)} "
        f"changed_rows={changed_rows} distinct_predicates={distinct_predicates}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
