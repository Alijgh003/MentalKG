#!/usr/bin/env python3
"""Create and populate the minimal four-table graph database."""

from __future__ import annotations

import argparse
import json

import psycopg

from config.settings import DatabaseSettings
from kg_pipeline.minimal_graph_export import (
    MinimalGraphExporter,
    database_name_from_env,
    ensure_database,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-name", default=database_name_from_env())
    args = parser.parse_args()
    source_dsn = DatabaseSettings().database_url
    destination_dsn = ensure_database(source_dsn, args.database_name)
    with psycopg.connect(source_dsn) as source, psycopg.connect(destination_dsn) as target:
        source.execute("SET TRANSACTION READ ONLY")
        exporter = MinimalGraphExporter(source, target)
        summary = exporter.rebuild()
        target.commit()
    print(f"Minimal graph database: {args.database_name}")
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
