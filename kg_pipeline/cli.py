"""Command line interface for validating and importing the DSM knowledge graph."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .paths import DatasetPaths
from .validation import validate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and load the DSM PostgreSQL knowledge graph")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root containing books/")
    parser.add_argument("--validate-only", action="store_true", help="Validate files and print counts; no DB needed")
    parser.add_argument("--audit-only", action="store_true", help="Print a compact read-only database audit")
    parser.add_argument("--dsn", default=os.getenv("DSM_KG_DATABASE_URL"), help="Target PostgreSQL DSN")
    parser.add_argument("--create-database", action="store_true", help="Create the target database before loading")
    parser.add_argument("--admin-dsn", default=os.getenv("POSTGRES_ADMIN_DSN"), help="Administrative DSN, normally connected to postgres")
    parser.add_argument("--database-name", default=os.getenv("DSM_KG_DATABASE_NAME", "dsm5_kg"), help="Database name to create")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    paths = DatasetPaths(args.root.resolve())
    if args.audit_only:
        if not args.dsn:
            raise SystemExit("--dsn (or DSM_KG_DATABASE_URL) is required to audit PostgreSQL")
        try:
            import psycopg
        except ImportError as error:
            raise SystemExit("Install dependencies first: pip install -r requirements.txt") from error
        from .audit import print_audit
        with psycopg.connect(args.dsn) as connection:
            print_audit(connection)
        return 0
    logging.getLogger(__name__).info("Validating source artifacts under %s", paths.root)
    report = validate_dataset(paths)
    for key, value in vars(report).items():
        print(f"{key}: {value}")
    if args.validate_only:
        return 0
    if not args.dsn:
        raise SystemExit("--dsn (or DSM_KG_DATABASE_URL) is required to load PostgreSQL")

    try:
        import psycopg
    except ImportError as error:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from error
    from .importer import PostgresImporter, create_database

    if args.create_database:
        if not args.admin_dsn:
            raise SystemExit("--admin-dsn (or POSTGRES_ADMIN_DSN) is required with --create-database")
        create_database(args.admin_dsn, args.database_name)
    logging.getLogger(__name__).info("Connecting to PostgreSQL and starting import")
    with psycopg.connect(args.dsn) as connection:
        PostgresImporter(connection, paths).load()
        connection.commit()
    logging.getLogger(__name__).info("PostgreSQL transaction committed")
    print("PostgreSQL import completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
